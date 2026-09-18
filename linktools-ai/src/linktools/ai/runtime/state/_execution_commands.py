#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution state checkpoint commands."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from ...errors import AIError, ErrorCode
from ._contracts import (
    ExecutionEventAppend,
    ExecutionHistorySealRecord,
    ExecutionHistoryState,
    ExecutionRepository,
    ExecutionRunSealHead,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
)
from ._durability import (
    CommitObservation,
    DurableCommitState,
    run_durable_commit,
)
from ._step_contracts import ContinuableSnapshot, RunRecord, StepEvent
from ._step_archive import (
    PreparedExecutionProjection,
    PreparedStepSnapshotBatch,
    StateStepArchive,
)
from ._store import StateStore, StateTransaction


class ExecutionStateCommands:
    """Commit execution audit, step projection, and terminal state together."""

    def __init__(
        self,
        state_store: StateStore,
        executions: ExecutionRepository,
        steps: StateStepArchive | None,
        *,
        background_tasks: "set[asyncio.Task[object]]",
    ) -> None:
        self._state_store = state_store
        self._executions = executions
        self._steps = steps
        self._background_tasks = background_tasks

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        step_run: RunRecord | None,
        step_events: Sequence[StepEvent] = (),
        snapshots: Sequence[ContinuableSnapshot] = (),
        audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionTerminalCommitResult:
        prepared_snapshots = ()
        if self._steps is not None and step_run is not None and snapshots:
            prepared_snapshots = await self._steps.prepare_snapshots(
                step_run,
                snapshots,
            )
        history_seal = _execution_history_seal(
            commit,
            audit_events=audit_events,
            projections=(),
            current_run=step_run,
            current_events=step_events,
            current_batch=(
                prepared_snapshots
                if isinstance(prepared_snapshots, PreparedStepSnapshotBatch)
                else None
            ),
        )

        async def mutate(
            transaction: StateTransaction,
        ) -> ExecutionTerminalCommitResult:
            head, head_record = (
                await self._executions.require_open_history_head_in_transaction(
                    transaction,
                    commit.execution.execution_id,
                )
            )
            effective_commit = commit
            if effective_commit.idempotency is None:
                idempotency = (
                    await self._executions.terminal_idempotency_in_transaction(
                        transaction,
                        effective_commit,
                    )
                )
                effective_commit = replace(effective_commit, idempotency=idempotency)
            if (
                self._steps is None
                and step_run is not None
                and (step_events or snapshots)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self._steps is not None and step_run is not None:
                await self._steps.sync_projection_in_transaction(
                    transaction,
                    step_run,
                    events=step_events,
                    snapshots=(
                        prepared_snapshots.snapshots
                        if isinstance(prepared_snapshots, PreparedStepSnapshotBatch)
                        else ()
                    ),
                    execution_id=commit.execution.execution_id,
                    history_head_guard=(head, head_record),
                )
            await self._executions.put_history_seal_in_transaction(
                transaction,
                history_seal,
            )
            await self._executions.replace_history_head_in_transaction(
                transaction,
                head_record,
                replace(
                    head,
                    state=ExecutionHistoryState.SEALED,
                    revision=head.revision + 1,
                    seal_digest=history_seal.seal_digest,
                ),
            )
            return await self._executions.commit_terminal_in_transaction(
                transaction,
                effective_commit,
                pending_events=audit_events,
            )

        async def readback() -> CommitObservation[ExecutionTerminalCommitResult]:
            try:
                execution = await self._executions.get(
                    commit.execution.execution_id,
                    tenant_id=commit.execution.tenant_id,
                )
                head = await self._executions.get_history_head(
                    commit.execution.execution_id,
                    tenant_id=commit.execution.tenant_id,
                )
                seal = await self._executions.get_history_seal(
                    commit.execution.execution_id,
                    tenant_id=commit.execution.tenant_id,
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(
                    DurableCommitState.UNRESOLVED,
                    error=error,
                )
            if execution is None or head is None or seal is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            expected_revision = commit.expected_revision + len(audit_events) + 1
            expected_event_sequence = (
                commit.expected_event_sequence + len(audit_events) + 1
            )
            if (
                head.state is ExecutionHistoryState.SEALED
                and seal == history_seal
                and execution.status is commit.execution.status
                and execution.result == commit.result
                and execution.error_code == commit.execution.error_code
                and execution.revision == expected_revision
                and execution.event_sequence == expected_event_sequence
            ):
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=ExecutionTerminalCommitResult(
                        execution,
                        commit.result,
                    ),
                )
            if head.state is ExecutionHistoryState.SEALED:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            return CommitObservation(DurableCommitState.NOT_COMMITTED)

        outcome = await run_durable_commit(
            lambda: self._state_store.mutate(mutate),
            readback,
            background_tasks=self._background_tasks,
        )
        if outcome.state is DurableCommitState.COMMITTED:
            if outcome.value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if outcome.cancelled:
                raise asyncio.CancelledError
            return outcome.value
        if outcome.state is DurableCommitState.NOT_COMMITTED:
            if outcome.error is not None:
                raise outcome.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "terminal checkpoint left partial durable state",
            ) from outcome.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error


def _execution_history_seal(
    commit: ExecutionTerminalCommit,
    *,
    audit_events: Sequence[ExecutionEventAppend],
    projections: Sequence[PreparedExecutionProjection],
    current_run: RunRecord | None,
    current_events: Sequence[StepEvent],
    current_batch: PreparedStepSnapshotBatch | None,
) -> ExecutionHistorySealRecord:
    heads = [
        ExecutionRunSealHead(
            projection.run.run_id,
            projection.target_event_offset,
            projection.target_snapshot_offset,
            projection.target_transcript_message_count,
            projection.projection_digest,
            projection.target_interaction_offset,
        )
        for projection in projections
    ]
    if not projections and current_run is not None:
        heads.append(
            ExecutionRunSealHead(
                current_run.run_id,
                len(current_events),
                0 if current_batch is None else len(current_batch.snapshots),
                0
                if current_batch is None
                else current_batch.target_transcript_message_count,
                "empty"
                if current_batch is None or not current_batch.snapshots
                else current_batch.snapshots[-1].projection.digest,
                0,
            )
        )
    ordered_heads = tuple(sorted(heads, key=lambda head: head.run_id))
    execution_event_high_water = (
        commit.expected_event_sequence + len(audit_events) + 1
    )
    return ExecutionHistorySealRecord(
        execution_id=commit.execution.execution_id,
        tenant_id=commit.execution.tenant_id,
        run_heads=ordered_heads,
        execution_event_high_water=execution_event_high_water,
    )


__all__ = ["ExecutionStateCommands"]
