#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InMemoryRunCommitCoordinator: an from-scratch RunCommitCoordinator built
ONLY from the public ``linktools.ai`` Protocols and domain models, proving the
external-adapter chain does not need to reuse any in-repo reference backend
(``FilesystemRunCommitCoordinator`` included) to drive a run's pause/complete
commit.

The in-memory adapter has no crash-recovery requirement (there is no process
restart to survive in a test run), so this coordinator skips the disk-backed
transaction journal the Filesystem reference uses for crash recovery and
instead tracks in-flight/completed commit_ids in a plain dict for idempotency.
Both coordinators satisfy the same ``RunCommitCoordinator`` Protocol
(pause/complete); this one just does not implement
``recover_incomplete_commits`` (out of Protocol scope, and out of scope for an
in-memory adapter with no disk state to recover)."""

import asyncio
from dataclasses import replace
from typing import Any

from linktools.ai.events.context import EventContext, append_event
from linktools.ai.events.payloads import (
    ApprovalRequested,
    RunCompleted,
    RunPaused as RunPausedEvent,
)
from linktools.ai.events.registry import default_codec
from linktools.ai.run.commit import (
    CompleteRunCommand,
    CompletedRunCommit,
    PauseRunCommand,
    PausedRunCommit,
)
from linktools.ai.run.lifecycle import mark_completed
from linktools.ai.run.models import NewRunCheckpoint, RunStatus


class InMemoryRunCommitCoordinator:
    """Sequential, idempotent-by-commit_id commit for the in-memory external
    Storage. Mirrors the write ORDER the Filesystem reference uses (approval,
    checkpoint, run transition, events for pause; checkpoint, run transition,
    session messages, event for complete) so both coordinators uphold the same
    "the run transition is the commit point" invariant, but composes it purely
    from public Store Protocols -- no reference-backend import."""

    def __init__(
        self,
        *,
        approval_store: Any,
        checkpoint_store: Any,
        run_store: Any,
        session_store: Any,
        event_store: Any,
    ) -> None:
        self._approvals = approval_store
        self._checkpoints = checkpoint_store
        self._runs = run_store
        self._sessions = session_store
        self._events = event_store
        self._lock = asyncio.Lock()

    @classmethod
    def from_storage(cls, storage: Any) -> "InMemoryRunCommitCoordinator":
        """Build a coordinator wired to an in-memory external Storage's
        stores -- the adapter's own composition-root helper, mirroring the
        shape of ``FilesystemRunCommitCoordinator.from_storage`` without
        depending on it."""
        return cls(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
        )

    async def pause(self, command: PauseRunCommand) -> PausedRunCommit:
        async with self._lock:
            approval_id = command.approval_request.get("approval_id", "")
            commit_id = command.commit_id or f"pause:{command.run_id}:{approval_id}"
            existing_run = await self._runs.get(command.run_id)
            if (
                existing_run is not None
                and existing_run.status is RunStatus.WAITING_APPROVAL
            ):
                latest = await self._checkpoints.latest(command.run_id)
                stored_commit_id = (
                    latest.metadata.get("commit_id") if latest is not None else None
                )
                if stored_commit_id == commit_id:
                    return PausedRunCommit(
                        approval_id=latest.metadata.get("approval_id", approval_id),
                        checkpoint_id=latest.id,
                    )

            if command.approval_request.get("tool_call_id") is not None:
                approval = await self._approvals.create_or_get_pending(
                    tenant_id=command.approval_request.get("tenant_id") or "",
                    run_id=command.run_id,
                    tool_call_id=command.approval_request["tool_call_id"],
                    tool_name=command.approval_request.get("tool_name", ""),
                    reason=command.approval_request.get("reason"),
                    arguments=command.approval_request.get("arguments", {}),
                    approval_id=approval_id,
                    binding={
                        k: command.approval_request[k]
                        for k in (
                            "descriptor_fingerprint",
                            "handler_revision",
                            "provider_revision",
                            "policy_revision",
                            "capability_revision",
                            "result_processor_revision",
                            "arguments_hash",
                        )
                        if k in command.approval_request
                    },
                )
                approval_id = approval.id

            checkpoint = NewRunCheckpoint(
                run_id=command.run_id,
                format="pydantic-ai-v1",
                schema_version=1,
                payload=command.checkpoint_payload,
                metadata={"approval_id": approval_id, "commit_id": commit_id},
            )
            persisted = await self._checkpoints.append(checkpoint)

            await self._runs.transition(
                command.run_id,
                RunStatus.WAITING_APPROVAL,
                expected_version=command.expected_version,
            )

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
                await self._append_event_tagged(
                    context=command.event_context, commit_id=commit_id, payload=payload
                )

            return PausedRunCommit(approval_id=approval_id, checkpoint_id=persisted.id)

    async def complete(self, command: CompleteRunCommand) -> CompletedRunCommit:
        async with self._lock:
            commit_id = (
                command.commit_id
                or f"complete:{command.run_id}:{command.expected_version}"
            )
            existing = await self._runs.get(command.run_id)
            if existing is not None and existing.status is RunStatus.SUCCEEDED:
                latest = await self._checkpoints.latest(command.run_id)
                stored_commit_id = (
                    latest.metadata.get("commit_id") if latest is not None else None
                )
                if stored_commit_id == commit_id:
                    return CompletedRunCommit(result=existing.result)

            checkpoint = NewRunCheckpoint(
                run_id=command.run_id,
                format="pydantic-ai-v1",
                schema_version=1,
                payload=command.checkpoint_payload,
                metadata={"commit_id": commit_id},
            )
            await self._checkpoints.append(checkpoint)

            await mark_completed(
                self._runs,
                command.run_id,
                expected_version=command.expected_version,
                result=command.result,
            )

            if command.messages:
                tagged = tuple(
                    replace(
                        message,
                        metadata={**dict(message.metadata), "commit_id": commit_id},
                    )
                    for message in command.messages
                )
                await self._sessions.append_messages(command.session_id, tagged)

            await self._append_event_tagged(
                context=command.event_context,
                commit_id=commit_id,
                payload=RunCompleted(run_id=command.run_id),
            )

            return CompletedRunCommit(result=command.result)

    async def _append_event_tagged(
        self, *, context: EventContext, commit_id: str, payload: Any
    ) -> None:
        event_type = default_codec.registry.event_type_for(payload)
        page = await self._events.list(context.stream_id, after_sequence=0, limit=10000)
        already = any(
            event.run_id == context.run_id
            and default_codec.registry.event_type_for(event.payload) == event_type
            and event.metadata.get("commit_id") == commit_id
            for event in page.items
        )
        if already:
            return
        await append_event(self._events, context, payload, metadata={"commit_id": commit_id})


__all__: "list[str]" = ["InMemoryRunCommitCoordinator"]
