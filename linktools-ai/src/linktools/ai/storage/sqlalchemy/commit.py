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

from ...events.context import append_event
from ...events.payloads import (
    ApprovalRequested,
    RunCompleted,
    RunPaused as RunPausedEvent,
)
from ...run.commit import (
    CompleteRunCommand,
    CompletedRunCommit,
    PauseRunCommand,
    PausedRunCommit,
)
from ...run.models import NewRunCheckpoint, RunStatus

if TYPE_CHECKING:
    from .facade import SqlAlchemyStorage


class SqlAlchemyRunCommitCoordinator:
    """Atomic commit for SqlAlchemy-backed Storage. pause() and complete()
    share one transaction across every store so the commit is all-or-nothing."""

    def __init__(self, storage: "SqlAlchemyStorage") -> None:
        self._storage = storage

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
            await append_event(
                tx.events,
                command.event_context,
                RunPausedEvent(
                    run_id=command.run_id,
                    reason=f"approval required: {approval_id}",
                ),
            )

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
            await tx.runs.transition(
                command.run_id,
                RunStatus.SUCCEEDED,
                expected_version=command.expected_version,
                result=command.result,
            )
            await append_event(
                tx.events,
                command.event_context,
                RunCompleted(run_id=command.run_id),
            )

        return CompletedRunCommit(result=command.result)
