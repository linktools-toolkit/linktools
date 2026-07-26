#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyRunCommitCoordinator: atomic cross-store commit for SQL backends.

pause() and complete() each open one SQLAlchemy storage transaction
UnitOfWork -- every store involved (approvals, checkpoints, runs, sessions,
events) binds to the SAME AsyncSession and the SAME transaction, so the whole
commit either lands together or rolls back together. A failure in any
mandatory step propagates: the ``async with`` exits with an exception, the
transaction rolls back, and no half-committed state survives (no orphan
approval, no orphan checkpoint, the run stays in its prior status).

commit_id-keyed idempotent replay (P5 SQL commit log): every commit records
(commit_id, operation, run_id, request_hash, result_json) in the run_commit_log
table WITHIN the same UoW as the business writes. A retried call with the SAME
commit_id + request_hash returns the recorded result; SAME commit_id +
DIFFERENT request_hash raises RunCommitConflictError. The log lives in the run
domain (storage.sqlalchemy owns the transactional UoW primitive; this module
owns the per-commit replay table)."""

import base64
from typing import TYPE_CHECKING, Any, Mapping

from ....errors import RunConflictError
from ....events.context import append_event
from ....events.payloads import ApprovalRequested
from ...commit import (
    AcknowledgeCancelRunCommand,
    CancelledRunCommit,
    CancellingRunCommit,
    CompleteRunCommand,
    CompletedRunCommit,
    FailRunCommand,
    FailedRunCommit,
    PauseRunCommand,
    PausedRunCommit,
    RequestCancelRunCommand,
    ResumeRunCommand,
    ResumedRunCommit,
    StartRunCommand,
    StartedRunCommit,
    RunCommitPolicy,
    ExecutionFence,
)
from ...models import NewRunCheckpoint, RunRecord, RunStatus
from .commit_log import (
    RunCommitConflictError,
    SqlAlchemyRunCommitLog,
)
from ..codec import RunCommitCodec

if TYPE_CHECKING:
    from ....runtime.persistence.facade import Storage



class SqlAlchemyRunCommitCoordinator:
    """Atomic commit for SqlAlchemy-backed Storage. Every operation opens one
    transaction across every store involved so the commit is all-or-nothing,
    AND records the commit in run_commit_log so a retried call is an
    idempotent replay (same id + same payload) or a real conflict (same id +
    different payload), never a silent re-execution."""

    def __init__(self, storage: "Storage", *, policy: RunCommitPolicy, codec: RunCommitCodec) -> None:
        self._storage = storage
        self._commit_log = SqlAlchemyRunCommitLog()
        self._codec = codec
        self._policy = policy

    async def recover_incomplete_commits(self) -> None:
        """No-op: every SQL commit is one atomic ``Storage.transaction()``
        UnitOfWork, so a crash mid-commit rolls the whole thing back and leaves
        no in-flight state to recover. (Crash-recovery is the Filesystem
        coordinator's concern -- sequential writes through a journal.)"""
        return None

    def _check_execution_token(self, existing: "RunRecord | None", token: str) -> None:
        """Terminal-commit fencing: a non-empty stored ``execution_token``
        that differs from the command's token means a later claim owns this
        run now -- the caller's commit must be rejected rather than
        clobbering the current owner's progress. An empty token on either
        side (an older/custom RunStore that does not support
        claim_execution) skips the check, matching AgentEngine's own
        optional ``claim_execution`` call."""
        self._policy.validate(
            supplied=ExecutionFence(token) if token else None,
            stored_token=existing.execution_token if existing is not None else None,
        )

    async def _check_replay(
        self,
        session,
        *,
        commit_id: str,
        operation: str,
        request_payload: object,
    ) -> "Mapping[str, Any] | None":
        """Within the UoW's session, look up commit_id. Return the recorded
        result dict if this is an idempotent replay (same id + same
        request_hash); raise RunCommitConflictError on a hash mismatch;
        return None on a fresh commit so the caller proceeds with the
        business writes and then records the result itself."""
        request_hash = self._codec.request_hash(operation, request_payload)
        existing = await self._commit_log.find(session, commit_id)
        if existing is None:
            return None
        if existing.request_hash != request_hash:
            raise RunCommitConflictError(
                f"commit {commit_id!r} replayed with a different request"
            )
        result = dict(existing.result)
        if existing.result_payload is not None:
            result["result_payload"] = base64.b64encode(existing.result_payload).decode("ascii")
        return result

    async def pause(self, command: PauseRunCommand) -> PausedRunCommit:
        """Persist approval + checkpoint + transition + events in one txn.

        Any step failing rolls back the whole transaction and propagates, so
        the run is never left WAITING_APPROVAL without its approval/checkpoint
        (and never left half-paused at all)."""
        approval_id = command.approval_request.approval_id
        commit_id = command.commit_id.value
        async with self._storage.transaction() as tx:
            self._check_execution_token(
                await tx.runs.get(command.run_id),
                command.execution_fence.token if command.execution_fence else "",
            )
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="pause",
                request_payload=command,
            )
            if replay is not None:
                return PausedRunCommit(
                    approval_id=replay.get("approval_id", approval_id),
                    checkpoint_id=replay.get("checkpoint_id", ""),
                )

            if command.approval_request.tool_call_id is not None:
                approval = await tx.approvals.create_or_get_pending(
                    run_id=command.run_id,
                    tenant_id=command.approval_request.tenant_id,
                    tool_call_id=command.approval_request.tool_call_id,
                    tool_name=command.approval_request.tool_name,
                    reason=command.approval_request.reason,
                    arguments=command.approval_request.arguments,
                    approval_id=approval_id,
                    binding={k: command.approval_request.binding[k] for k in (
                        "descriptor_fingerprint", "handler_revision", "provider_revision",
                        "policy_revision", "capability_revision", "result_processor_revision",
                        "arguments_hash"
                    ) if k in command.approval_request.binding},
                )
                approval_id = approval.id

            checkpoint = NewRunCheckpoint(
                run_id=command.run_id,
                format="pydantic-ai-v1",
                schema_version=1,
                payload=command.checkpoint_payload,
                metadata={"approval_id": approval_id},
            )
            persisted = await tx.checkpoints.append(checkpoint)

            # WAITING_APPROVAL transition inside the same txn -- a failure
            # rolls back the approval + checkpoint writes above too.
            await tx.runs.transition(
                command.run_id,
                RunStatus.WAITING_APPROVAL,
                expected_version=command.expected_version,
            )

            # Critical lifecycle events ride the same transaction so they are
            # consistent with the state transition.
            await append_event(
                tx.events,
                command.event_context,
                ApprovalRequested(
                    approval_id=approval_id,
                    tool_name=command.approval_request.tool_name,
                    reason=command.approval_request.reason,
                ),
            )
            await append_event(tx.events, command.event_context, command.paused_event)

            # Record the commit in the SAME UoW so a future replay is
            # idempotent. Business writes + commit-log row commit together.
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="pause",
                run_id=command.run_id,
                request_hash=self._codec.request_hash("pause", command),
                result={
                    "approval_id": approval_id,
                    "checkpoint_id": persisted.id,
                },
            )

        return PausedRunCommit(approval_id=approval_id, checkpoint_id=persisted.id)

    async def complete(self, command: CompleteRunCommand) -> CompletedRunCommit:
        """Persist session messages + checkpoint + SUCCEEDED transition +
        RunCompleted event in one txn. Any step failing rolls back everything."""
        commit_id = command.commit_id.value
        checkpoint = NewRunCheckpoint(
            run_id=command.run_id,
            format="pydantic-ai-v1",
            schema_version=1,
            payload=command.checkpoint_payload,
        )
        async with self._storage.transaction() as tx:
            self._check_execution_token(
                await tx.runs.get(command.run_id),
                command.execution_fence.token if command.execution_fence else "",
            )
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="complete",
                request_payload=command,
            )
            if replay is not None:
                encoded = replay.get("result_payload")
                if not isinstance(encoded, str):
                    raise RunCommitConflictError("complete replay has no persisted result payload")
                decoded = self._codec.decode_result(
                    "complete", base64.b64decode(encoded.encode("ascii"))
                )
                from ...models import RunResult
                return CompletedRunCommit(result=RunResult(**decoded))

            await tx.sessions.append_messages(command.session_id, command.messages)
            await tx.checkpoints.append(checkpoint)
            # SUCCEEDED transition persists command.result -- inside the txn so
            # a failure here rolls back the session/checkpoint writes above.
            await tx.runs.transition(
                command.run_id,
                RunStatus.SUCCEEDED,
                expected_version=command.expected_version,
                result=command.result,
            )
            await append_event(tx.events, command.event_context, command.completed_event)
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="complete",
                run_id=command.run_id,
                request_hash=self._codec.request_hash("complete", command),
                result={"run_id": command.run_id},
                result_payload=self._codec.encode_result("complete", command.result),
            )

        return CompletedRunCommit(result=command.result)

    async def start(self, command: StartRunCommand) -> StartedRunCommit:
        """Create the RunRecord (already RUNNING -- callers no longer create
        PENDING then transition separately) + append RunStarted in one txn.

        Idempotent by commit_id: a retried start with the SAME id + record
        payload returns the recorded record instead of double-creating; a
        different payload under the same id is a RunCommitConflictError."""
        commit_id = command.commit_id.value
        async with self._storage.transaction() as tx:
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="start",
                request_payload=command,
            )
            if replay is not None:
                # The original commit returned the created record; the
                # caller's command carries the canonical record, so re-use it.
                return StartedRunCommit(record=command.record)
            created = await tx.runs.create(command.record)
            await append_event(tx.events, command.event_context, command.started_event)
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="start",
                run_id=command.record.id,
                request_hash=self._codec.request_hash("start", command),
                result={"run_id": command.record.id},
            )
        return StartedRunCommit(record=created)

    async def resume(self, command: ResumeRunCommand) -> ResumedRunCommit:
        """Transition WAITING_APPROVAL -> RUNNING + append RunResumed in one
        txn. Idempotent by commit_id."""
        commit_id = command.commit_id.value
        async with self._storage.transaction() as tx:
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="resume",
                request_payload=command,
            )
            if replay is not None:
                return ResumedRunCommit(run_id=command.run_id)
            await tx.runs.transition(
                command.run_id,
                RunStatus.RUNNING,
                expected_version=command.expected_version,
            )
            await append_event(tx.events, command.event_context, command.resumed_event)
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="resume",
                run_id=command.run_id,
                request_hash=self._codec.request_hash("resume", command),
                result={"run_id": command.run_id},
            )
        return ResumedRunCommit(run_id=command.run_id)

    async def fail(self, command: FailRunCommand) -> FailedRunCommit:
        """Transition -> FAILED (with error) + append RunFailed in one txn.
        Idempotent by commit_id."""
        commit_id = command.commit_id.value
        async with self._storage.transaction() as tx:
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="fail",
                request_payload=command,
            )
            if replay is not None:
                return FailedRunCommit(run_id=command.run_id)
            self._check_execution_token(
                await tx.runs.get(command.run_id), (command.execution_fence.token if command.execution_fence else '')
            )
            await tx.runs.transition(
                command.run_id,
                RunStatus.FAILED,
                expected_version=command.expected_version,
                error=command.error,
            )
            await append_event(tx.events, command.event_context, command.failed_event)
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="fail",
                run_id=command.run_id,
                request_hash=self._codec.request_hash("fail", command),
                result={"run_id": command.run_id},
            )
        return FailedRunCommit(run_id=command.run_id)

    async def request_cancel(
        self, command: RequestCancelRunCommand
    ) -> CancellingRunCommit:
        """Transition -> CANCELLING (audit fields only -- no state-critical
        event; RunCancelled is appended later by ``acknowledge_cancel`` once
        the execution has actually stopped). Idempotent by commit_id."""
        from datetime import datetime, timezone

        commit_id = command.commit_id.value
        async with self._storage.transaction() as tx:
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="request_cancel",
                request_payload=command,
            )
            if replay is not None:
                return CancellingRunCommit(run_id=command.run_id)
            await tx.runs.transition(
                command.run_id,
                RunStatus.CANCELLING,
                expected_version=command.expected_version,
                cancel_requested_at=datetime.now(timezone.utc),
                cancel_requested_by=command.requested_by,
                cancel_reason=command.reason,
            )
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="request_cancel",
                run_id=command.run_id,
                request_hash=self._codec.request_hash("request_cancel", command),
                result={"run_id": command.run_id},
            )
        return CancellingRunCommit(run_id=command.run_id)

    async def acknowledge_cancel(
        self, command: AcknowledgeCancelRunCommand
    ) -> CancelledRunCommit:
        """Transition CANCELLING -> CANCELLED + append RunCancelled in one
        txn. Idempotent by commit_id."""
        commit_id = command.commit_id.value
        async with self._storage.transaction() as tx:
            replay = await self._check_replay(
                tx.session,
                commit_id=commit_id,
                operation="acknowledge_cancel",
                request_payload=command,
            )
            if replay is not None:
                return CancelledRunCommit(run_id=command.run_id)
            self._check_execution_token(
                await tx.runs.get(command.run_id), (command.execution_fence.token if command.execution_fence else '')
            )
            await tx.runs.transition(
                command.run_id,
                RunStatus.CANCELLED,
                expected_version=command.expected_version,
            )
            await append_event(
                tx.events, command.event_context, command.cancelled_event
            )
            await self._commit_log.record(
                tx.session,
                commit_id=commit_id,
                operation="acknowledge_cancel",
                run_id=command.run_id,
                request_hash=self._codec.request_hash("acknowledge_cancel", command),
                result={"run_id": command.run_id},
            )
        return CancelledRunCommit(run_id=command.run_id)
