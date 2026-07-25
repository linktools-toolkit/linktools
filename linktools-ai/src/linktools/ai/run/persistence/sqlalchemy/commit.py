#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyRunCommitCoordinator: atomic cross-store commit for SQL backends.

pause() and complete() each open one ``SqlAlchemyStorage.transaction()``
UnitOfWork -- every store involved (approvals, checkpoints, runs, sessions,
events) binds to the SAME AsyncSession and the SAME transaction, so the whole
commit either lands together or rolls back together. A failure in any
mandatory step propagates: the ``async with`` exits with an exception, the
transaction rolls back, and no half-committed state survives (no orphan
approval, no orphan checkpoint, the run stays in its prior status)."""

from typing import TYPE_CHECKING

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
)
from ...models import NewRunCheckpoint, RunRecord, RunStatus

if TYPE_CHECKING:
    from .facade import SqlAlchemyStorage


class SqlAlchemyRunCommitCoordinator:
    """Atomic commit for SqlAlchemy-backed Storage. Every operation opens one
    transaction across every store involved so the commit is all-or-nothing.

    Unlike :class:`~..filesystem.commit.FilesystemRunCommitCoordinator`, this
    coordinator does not yet implement commit_id-based idempotent replay --
    a retried call re-executes rather than detecting an already-committed
    commit_id. Closing that gap is separate, larger follow-up work."""

    def __init__(self, storage: "SqlAlchemyStorage") -> None:
        self._storage = storage

    def _check_execution_token(self, existing: "RunRecord | None", token: str) -> None:
        """Terminal-commit fencing: a non-empty stored ``execution_token``
        that differs from the command's token means a later claim owns this
        run now -- the caller's commit must be rejected rather than
        clobbering the current owner's progress. An empty token on either
        side (an older/custom RunStore that does not support
        claim_execution) skips the check, matching AgentEngine's own
        optional ``claim_execution`` call."""
        if not token or existing is None or not existing.execution_token:
            return
        if existing.execution_token != token:
            raise RunConflictError(
                f"execution token fencing failed for run {existing.id}: "
                "commit token does not match the current claim"
            )

    async def pause(self, command: PauseRunCommand) -> PausedRunCommit:
        """Persist approval + checkpoint + transition + events in one txn.

        Any step failing rolls back the whole transaction and propagates, so
        the run is never left WAITING_APPROVAL without its approval/checkpoint
        (and never left half-paused at all)."""
        approval_id = command.approval_request.get("approval_id", "")
        async with self._storage.transaction() as tx:
            if command.approval_request.get("tool_call_id") is not None:
                approval = await tx.approvals.create_or_get_pending(
                    run_id=command.run_id,
                    tenant_id=command.approval_request.get("tenant_id", ""),
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
            self._check_execution_token(
                await tx.runs.get(command.run_id), command.execution_token
            )
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
                    tool_name=command.approval_request.get("tool_name", ""),
                    reason=command.approval_request.get("reason") or "",
                ),
            )
            await append_event(tx.events, command.event_context, command.paused_event)

        return PausedRunCommit(approval_id=approval_id, checkpoint_id=persisted.id)

    async def complete(self, command: CompleteRunCommand) -> CompletedRunCommit:
        """Persist session messages + checkpoint + SUCCEEDED transition +
        RunCompleted event in one txn. Any step failing rolls back everything."""
        checkpoint = NewRunCheckpoint(
            run_id=command.run_id,
            format="pydantic-ai-v1",
            schema_version=1,
            payload=command.checkpoint_payload,
        )
        async with self._storage.transaction() as tx:
            await tx.sessions.append_messages(command.session_id, command.messages)
            await tx.checkpoints.append(checkpoint)
            # SUCCEEDED transition persists command.result -- inside the txn so
            # a failure here rolls back the session/checkpoint writes above.
            self._check_execution_token(
                await tx.runs.get(command.run_id), command.execution_token
            )
            await tx.runs.transition(
                command.run_id,
                RunStatus.SUCCEEDED,
                expected_version=command.expected_version,
                result=command.result,
            )
            await append_event(tx.events, command.event_context, command.completed_event)

        return CompletedRunCommit(result=command.result)

    async def start(self, command: StartRunCommand) -> StartedRunCommit:
        """Create the RunRecord (already RUNNING -- callers no longer create
        PENDING then transition separately) + append RunStarted in one txn."""
        async with self._storage.transaction() as tx:
            created = await tx.runs.create(command.record)
            await append_event(tx.events, command.event_context, command.started_event)
        return StartedRunCommit(record=created)

    async def resume(self, command: ResumeRunCommand) -> ResumedRunCommit:
        """Transition WAITING_APPROVAL -> RUNNING + append RunResumed in one
        txn."""
        async with self._storage.transaction() as tx:
            await tx.runs.transition(
                command.run_id,
                RunStatus.RUNNING,
                expected_version=command.expected_version,
            )
            await append_event(tx.events, command.event_context, command.resumed_event)
        return ResumedRunCommit(run_id=command.run_id)

    async def fail(self, command: FailRunCommand) -> FailedRunCommit:
        """Transition -> FAILED (with error) + append RunFailed in one txn."""
        async with self._storage.transaction() as tx:
            self._check_execution_token(
                await tx.runs.get(command.run_id), command.execution_token
            )
            await tx.runs.transition(
                command.run_id,
                RunStatus.FAILED,
                expected_version=command.expected_version,
                error=command.error,
            )
            await append_event(tx.events, command.event_context, command.failed_event)
        return FailedRunCommit(run_id=command.run_id)

    async def request_cancel(
        self, command: RequestCancelRunCommand
    ) -> CancellingRunCommit:
        """Transition -> CANCELLING (audit fields only -- no state-critical
        event; RunCancelled is appended later by ``acknowledge_cancel`` once
        the execution has actually stopped)."""
        from datetime import datetime, timezone

        async with self._storage.transaction() as tx:
            await tx.runs.transition(
                command.run_id,
                RunStatus.CANCELLING,
                expected_version=command.expected_version,
                cancel_requested_at=datetime.now(timezone.utc),
                cancel_requested_by=command.requested_by,
                cancel_reason=command.reason,
            )
        return CancellingRunCommit(run_id=command.run_id)

    async def acknowledge_cancel(
        self, command: AcknowledgeCancelRunCommand
    ) -> CancelledRunCommit:
        """Transition CANCELLING -> CANCELLED + append RunCancelled in one
        txn."""
        async with self._storage.transaction() as tx:
            self._check_execution_token(
                await tx.runs.get(command.run_id), command.execution_token
            )
            await tx.runs.transition(
                command.run_id,
                RunStatus.CANCELLED,
                expected_version=command.expected_version,
            )
            await append_event(
                tx.events, command.event_context, command.cancelled_event
            )
        return CancelledRunCommit(run_id=command.run_id)
