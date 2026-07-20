#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemRunCommitCoordinator: journaled cross-store commit for File backends.

File storage has no cross-store transaction, so pause and complete write their
stores sequentially. A crash mid-sequence could leave a run half-committed, so
each commit is journaled (``TransactionJournal`` under ``{root}/transactions``):
the coordinator advances the journal's state machine after every successful
write, and ``recover_incomplete_commits()`` runs at startup to make every
in-flight commit consistent with its run's authoritative status -- either the
run already reached its target state (commit point) and only best-effort events
are re-appended, or the run did NOT reach it and is marked FAILED (fail-closed:
no silent half-committed state)."""

import base64
import logging
from dataclasses import replace
from typing import Any, Mapping

from ...events.context import EventContext, append_event
from ...events.payloads import (
    ApprovalRequested,
    RunCompleted,
    RunPaused as RunPausedEvent,
)
from ...events.registry import default_codec
from ...run.commit import (
    CompleteRunCommand,
    CompletedRunCommit,
    PauseRunCommand,
    PausedRunCommit,
)
from ...run.lifecycle import mark_completed, mark_failed
from ...run.models import NewRunCheckpoint, RunErrorInfo, RunStatus
from ...session.models import MessageRole, NewSessionMessage
from .journal import TransactionJournal, TransactionKind, TransactionState

_LOGGER = logging.getLogger(__name__)


class FilesystemRunCommitCoordinator:
    """Journaled sequential commit for File-backed Storage."""

    def __init__(
        self,
        *,
        approval_store: Any,
        checkpoint_store: Any,
        run_store: Any,
        session_store: Any,
        event_store: Any,
        transactions_root: Any = None,
        metrics: Any = None,
    ) -> None:
        self._approvals = approval_store
        self._checkpoints = checkpoint_store
        self._runs = run_store
        self._sessions = session_store
        self._events = event_store
        # Optional ObservabilityMetrics sink. When wired, a critical-event
        # persist failure during recovery increments
        # ``critical_event_persist_failure_total``. Default None keeps existing
        # callers no-op.
        self._metrics = metrics
        # Production (Runtime.build) always passes storage.root / "transactions"
        # so recovery survives restarts. Tests that don't exercise recovery can
        # omit it and get an ephemeral temp dir (a real journal, just not shared
        # across coordinator instances).
        if transactions_root is None:
            import tempfile

            transactions_root = tempfile.mkdtemp(prefix="lt-txns-")
        self._journal = TransactionJournal(transactions_root)

    async def pause(self, command: PauseRunCommand) -> PausedRunCommit:
        """Persist approval + checkpoint + transition + events sequentially,
        advancing the journal after each write so a crash is recoverable."""
        approval_id = command.approval_request.get("approval_id", "")
        commit_id = command.commit_id or f"pause:{command.run_id}:{approval_id}"
        record = self._journal.find_incomplete(commit_id)
        existing_run = await self._runs.get(command.run_id)
        if (
            record is None
            and existing_run is not None
            and existing_run.status is RunStatus.WAITING_APPROVAL
        ):
            latest = await self._checkpoints.latest(command.run_id)
            stored_commit_id = (
                latest.metadata.get("commit_id") if latest is not None else None
            )
            if stored_commit_id and stored_commit_id != commit_id:
                raise RuntimeError(f"conflicting pause commit {commit_id}")
            return PausedRunCommit(
                approval_id=(
                    latest.metadata.get("approval_id", approval_id)
                    if latest is not None
                    else approval_id
                ),
                checkpoint_id=latest.id if latest is not None else "",
            )
        if record is None:
            record = self._journal.begin(
                kind=TransactionKind.PAUSE,
                run_id=command.run_id,
                target_run_status=RunStatus.WAITING_APPROVAL.value,
                commit_id=commit_id,
                command={
                    "approval_id": approval_id,
                    "tool_call_id": command.approval_request.get("tool_call_id"),
                    "tool_name": command.approval_request.get("tool_name", ""),
                    "reason": command.approval_request.get("reason") or "",
                    "arguments": command.approval_request.get("arguments", {}),
                    "checkpoint_payload_b64": _b64(command.checkpoint_payload),
                    "event_context": _ctx_to_dict(command.event_context),
                },
            )

        if record.state is TransactionState.PREPARED and (
            self._approvals is not None
            and command.approval_request.get("tool_call_id") is not None
        ):
            approval = await self._approvals.create_or_get_pending(
                tenant_id=command.approval_request.get("tenant_id") or "",
                run_id=command.run_id,
                tool_call_id=command.approval_request["tool_call_id"],
                tool_name=command.approval_request.get("tool_name", ""),
                reason=command.approval_request.get("reason"),
                arguments=command.approval_request.get("arguments", {}),
                approval_id=approval_id,
                binding={k: command.approval_request[k] for k in (
                    "descriptor_fingerprint", "handler_revision", "provider_revision",
                    "policy_revision", "capability_revision", "result_processor_revision",
                    "arguments_hash"
                ) if k in command.approval_request},
            )
            approval_id = approval.id
            record = self._journal.update(
                record, state="APPROVAL_WRITTEN", approval_id=approval_id
            )

        checkpoint = NewRunCheckpoint(
            run_id=command.run_id,
            format="pydantic-ai-v1",
            schema_version=1,
            payload=command.checkpoint_payload,
            metadata={"approval_id": approval_id, "commit_id": commit_id},
        )
        if record.state in (
            TransactionState.PREPARED,
            TransactionState.APPROVAL_WRITTEN,
        ):
            persisted = await self._checkpoints.latest(command.run_id)
            if persisted is None or persisted.metadata.get("commit_id") != commit_id:
                persisted = await self._checkpoints.append(checkpoint)
            record = self._journal.update(
                record, state="CHECKPOINT_WRITTEN", checkpoint_id=persisted.id
            )
        else:
            persisted = await self._checkpoints.get(record.checkpoint_id or "")
            if persisted is None:
                raise RuntimeError(f"pause checkpoint missing for commit {commit_id}")

        # The transition is the commit point: once the run is WAITING_APPROVAL,
        # the pause is durable. A crash before this leaves the run RUNNING and
        # recovery will mark it FAILED.
        if record.state in (
            TransactionState.PREPARED,
            TransactionState.APPROVAL_WRITTEN,
            TransactionState.CHECKPOINT_WRITTEN,
        ):
            await self._runs.transition(
                command.run_id,
                RunStatus.WAITING_APPROVAL,
                expected_version=command.expected_version,
            )
            record = self._journal.update(record, state="RUN_TRANSITIONED")

        if record.state is not TransactionState.EVENTS_WRITTEN:
            for payload in (
                ApprovalRequested(
                    approval_id=approval_id,
                    tool_name=command.approval_request.get("tool_name", ""),
                    reason=command.approval_request.get("reason") or "",
                ),
                RunPausedEvent(
                    run_id=command.run_id,
                    reason=f"approval required: {approval_id}",
                ),
            ):
                await self._append_critical_event_once(
                    context=command.event_context,
                    commit_id=commit_id,
                    payload=payload,
                )
            record = self._journal.update(record, state="EVENTS_WRITTEN")

        self._journal.commit(record)
        return PausedRunCommit(approval_id=approval_id, checkpoint_id=persisted.id)

    async def complete(self, command: CompleteRunCommand) -> CompletedRunCommit:
        """Persist session messages + checkpoint + SUCCEEDED transition + event,
        advancing the journal after each write so a crash is recoverable.

        Idempotent: a retried complete (the caller did not see the first
        response) finds the run already SUCCEEDED and returns its persisted
        result WITHOUT rewriting session/checkpoint/event -- no duplicates."""
        commit_id = (
            command.commit_id or f"complete:{command.run_id}:{command.expected_version}"
        )
        record = self._journal.find_incomplete(commit_id)
        existing = await self._runs.get(command.run_id)
        if (
            record is None
            and existing is not None
            and existing.status is RunStatus.SUCCEEDED
        ):
            latest = await self._checkpoints.latest(command.run_id)
            stored_commit_id = (
                latest.metadata.get("commit_id") if latest is not None else None
            )
            if stored_commit_id and stored_commit_id != commit_id:
                raise RuntimeError(f"conflicting complete commit {commit_id}")
            return CompletedRunCommit(result=existing.result)
        if record is None:
            record = self._journal.begin(
                kind=TransactionKind.COMPLETE,
                run_id=command.run_id,
                target_run_status=RunStatus.SUCCEEDED.value,
                commit_id=commit_id,
                command={
                    # Persist the full message content so recovery, after the
                    # commit point, can republish exactly these messages. They
                    # are NOT written to the session store until after the run
                    # reaches SUCCEEDED, so a pre-commit crash leaves the
                    # conversation history clean.
                    "session_id": command.session_id,
                    "messages": [_new_message_to_dict(m) for m in command.messages],
                    "event_context": _ctx_to_dict(command.event_context),
                },
            )

        checkpoint = NewRunCheckpoint(
            run_id=command.run_id,
            format="pydantic-ai-v1",
            schema_version=1,
            payload=command.checkpoint_payload,
        )

        # 1. Checkpoint (pre-commit). A crash here leaves the run not-yet-
        #    SUCCEEDED, so recovery marks it FAILED and publishes no messages.
        if record.state is TransactionState.PREPARED:
            persisted = await self._checkpoints.latest(command.run_id)
            if persisted is None or persisted.metadata.get("commit_id") != commit_id:
                persisted = await self._checkpoints.append(
                    replace(checkpoint, metadata={"commit_id": commit_id})
                )
            record = self._journal.update(
                record, state="CHECKPOINT_WRITTEN", checkpoint_id=persisted.id
            )

        # 2. Run -> SUCCEEDED is the commit point (it persists the result).
        if record.state is TransactionState.CHECKPOINT_WRITTEN:
            await mark_completed(
                self._runs,
                command.run_id,
                expected_version=command.expected_version,
                result=command.result,
            )
            record = self._journal.update(record, state="RUN_TRANSITIONED")

        # 3. Publish the session messages only AFTER the commit point. A crash
        #    here leaves the run SUCCEEDED, so recovery republishes from the
        #    journal (idempotent via commit_id) -- the conversation history ends
        #    up with exactly one copy of the answer.
        if record.state is TransactionState.RUN_TRANSITIONED:
            messages = [
                _message_from_dict(m) for m in record.command.get("messages") or ()
            ]
            if messages:
                await self._append_messages_once(
                    session_id=record.command["session_id"],
                    commit_id=commit_id,
                    messages=messages,
                )
            record = self._journal.update(record, state="SESSION_WRITTEN")

        # 4. RunCompleted event (post-commit).
        if record.state is TransactionState.SESSION_WRITTEN:
            await self._append_critical_event_once(
                context=command.event_context,
                commit_id=commit_id,
                payload=RunCompleted(run_id=command.run_id),
            )
            record = self._journal.update(record, state="EVENTS_WRITTEN")

        self._journal.commit(record)
        return CompletedRunCommit(result=command.result)

    async def _append_critical_event_once(
        self, *, context: EventContext, commit_id: str, payload: Any
    ) -> None:
        """Append a critical pause/complete event exactly once per
        (run_id, commit_id, event_type).

        Both the normal pause/complete path and crash recovery route through
        here, so they share one dedup rule and one tagging convention. The
        event is tagged with ``commit_id`` metadata so a run that legitimately
        pauses more than once -- each pause has its own commit_id -- keeps one
        ApprovalRequested/RunPaused per approval instead of the second being
        wrongly skipped (or the first duplicated on recovery).

        A failure is NOT swallowed: the caller leaves the journal at the
        pre-event state (EVENTS_WRITTEN is set only after this returns), so the
        journal is retained and the next retry or recovery re-attempts the
        write instead of permanently losing the audit event."""
        if await self._event_exists(
            context, commit_id, default_codec.registry.event_type_for(payload)
        ):
            return
        await append_event(
            self._events,
            context,
            payload,
            metadata={"commit_id": commit_id},
        )

    async def _event_exists(
        self, context: EventContext, commit_id: str, event_type: str
    ) -> bool:
        """True if a critical event of ``event_type`` tagged with ``commit_id``
        is already persisted for the run. Critical events carry their commit_id
        in metadata, so this distinguishes a re-run of the SAME commit (skip)
        from a NEW commit's event (keep) -- dedup by event type alone cannot."""
        registry = default_codec.registry
        page = await self._events.list(context.stream_id, after_sequence=0, limit=10000)
        return any(
            event.run_id == context.run_id
            and registry.event_type_for(event.payload) == event_type
            and event.metadata.get("commit_id") == commit_id
            for event in page.items
        )

    async def _append_messages_once(
        self,
        *,
        session_id: str,
        commit_id: str,
        messages: "list[NewSessionMessage]",
    ) -> None:
        """Publish a complete commit's session messages idempotently.

        Each message is tagged with ``commit_id`` metadata; messages already
        present for this commit_id (a retry or recovery re-run) are skipped, so
        the conversation history ends up with exactly one copy. A prior append
        for the same commit_id with differing content is a hard error rather
        than a silent overwrite. Shared by the post-commit complete() step and
        crash recovery so the two cannot diverge."""
        tagged = [
            replace(
                message, metadata={**dict(message.metadata), "commit_id": commit_id}
            )
            for message in messages
        ]
        existing = await self._sessions.list_messages(session_id)
        committed = [
            message
            for message in existing
            if message.metadata.get("commit_id") == commit_id
        ]
        if len(committed) > len(tagged) or any(
            persisted.role is not requested.role
            or persisted.content != requested.content
            or persisted.run_id != requested.run_id
            for persisted, requested in zip(committed, tagged)
        ):
            raise RuntimeError(f"conflicting session commit {commit_id}")
        missing = tagged[len(committed) :]
        if missing:
            await self._sessions.append_messages(session_id, missing)

    async def recover_incomplete_commits(self) -> None:
        """Finish or fail every in-flight commit left by a crash.

        For each incomplete journal, the run's current status is authoritative:
        - If it matches the journal's target status, the commit reached its
          commit point (the transition landed, so every pre-transition write is
          already persisted). Only the best-effort critical events are
          re-appended (the crash likely cut them off).
        - Otherwise the commit did NOT reach its commit point. The run is marked
          FAILED (fail-closed) -- orphan approval/checkpoint/session writes may
          remain on disk but are not surfaced as a successful run. We never
          re-drive non-idempotent writes (checkpoint/session append) because that
          would create duplicates."""
        for record in self._journal.list_incomplete():
            await self._recover_one(record)

    async def _recover_one(self, record: Any) -> None:
        run = await self._runs.get(record.run_id)
        if run is None:
            # The run is gone (expired/cleaned). Nothing to make consistent.
            self._discard(record)
            return
        if run.status.value == record.target_run_status:
            # Commit point reached. Re-publish post-commit writes (messages for
            # COMPLETE) and the critical events. A failure here is NOT fatal to
            # recovery: retain the journal so the next retry/startup re-attempts
            # it, rather than discarding it and permanently losing the event.
            try:
                await self._reappend_critical_events(record)
            except Exception:  # noqa: BLE001
                if self._metrics is not None:
                    self._metrics.counter("critical_event_persist_failure_total")
                _LOGGER.exception(
                    "critical event recovery failed for commit %s; retaining journal",
                    record.commit_id,
                )
                return
            self._discard(record)
            return
        # The commit did not reach its commit point. Mark FAILED (fail-closed).
        # Best-effort: a concurrent terminal transition (e.g. the run was already
        # FAILED/CANCELLED by another path) is tolerated.
        try:
            await mark_failed(
                self._runs,
                record.run_id,
                expected_version=run.version,
                error=RunErrorInfo(
                    error_type="IncompleteCommitRecovery",
                    message=(
                        f"run recovered with incomplete {record.kind.value} "
                        f"commit at state {record.state.value}"
                    ),
                ),
            )
        except Exception:  # noqa: BLE001
            _LOGGER.warning(
                "recovery could not mark run %s FAILED for incomplete commit %s",
                record.run_id,
                record.id,
            )
        self._discard(record)

    def _discard(self, record: Any) -> None:
        """Remove the journal once recovery has handled it (the commit is either
        consistent or the run is FAILED -- either way the journal is resolved)."""
        try:
            self._journal.commit(record)
        except Exception:  # noqa: BLE001
            _LOGGER.warning("could not discard journal %s", record.id)

    async def _reappend_critical_events(self, record: Any) -> None:
        ctx = _ctx_from_dict(record.command.get("event_context") or {})
        if ctx is None:
            return
        # Reuse the SAME commit-scoped append-once as the normal path: if the
        # crash cut off the event write mid-commit, this fills it in; if the
        # event was already written for this commit_id, it is a no-op. Either
        # way recovery cannot duplicate a critical event.
        commit_id = record.commit_id
        if record.kind is TransactionKind.PAUSE:
            await self._append_critical_event_once(
                context=ctx,
                commit_id=commit_id,
                payload=ApprovalRequested(
                    approval_id=record.approval_id or "",
                    tool_name=record.command.get("tool_name", ""),
                    reason=record.command.get("reason") or "",
                ),
            )
            await self._append_critical_event_once(
                context=ctx,
                commit_id=commit_id,
                payload=RunPausedEvent(
                    run_id=record.run_id,
                    reason=f"approval required: {record.approval_id}",
                ),
            )
        else:
            # COMPLETE recovery (post-commit -- the run reached SUCCEEDED):
            # republish the session messages from the journal, then the
            # RunCompleted event. A crash between the commit point and the
            # message write is closed here; both are idempotent via commit_id.
            session_id = record.command.get("session_id")
            messages = [
                _message_from_dict(m) for m in record.command.get("messages") or ()
            ]
            if session_id and messages:
                await self._append_messages_once(
                    session_id=session_id,
                    commit_id=commit_id,
                    messages=messages,
                )
            await self._append_critical_event_once(
                context=ctx,
                commit_id=commit_id,
                payload=RunCompleted(run_id=record.run_id),
            )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _new_message_to_dict(message: NewSessionMessage) -> "dict[str, Any]":
    """Serialize a session message for journal storage (recovery republish)."""
    return {
        "role": message.role.value,
        "content": message.content,
        "run_id": message.run_id,
        "metadata": dict(message.metadata),
    }


def _message_from_dict(data: "Mapping[str, Any]") -> NewSessionMessage:
    """Rehydrate a journaled session message."""
    return NewSessionMessage(
        role=MessageRole(data["role"]),
        content=data["content"],
        run_id=data.get("run_id"),
        metadata=dict(data.get("metadata") or {}),
    )


def _ctx_to_dict(ctx: Any) -> "dict[str, Any]":
    if ctx is None:
        return {}
    return {
        "stream_id": getattr(ctx, "stream_id", ""),
        "run_id": getattr(ctx, "run_id", ""),
        "root_run_id": getattr(ctx, "root_run_id", ""),
        "parent_run_id": getattr(ctx, "parent_run_id", None),
        "session_id": getattr(ctx, "session_id", ""),
        "runnable_id": getattr(ctx, "runnable_id", ""),
    }


def _ctx_from_dict(d: "dict[str, Any]") -> "EventContext | None":
    if not d or not d.get("run_id"):
        return None
    return EventContext(
        stream_id=d.get("stream_id") or d["run_id"],
        run_id=d["run_id"],
        root_run_id=d.get("root_run_id") or d["run_id"],
        parent_run_id=d.get("parent_run_id"),
        session_id=d.get("session_id", ""),
        runnable_id=d.get("runnable_id", ""),
    )
