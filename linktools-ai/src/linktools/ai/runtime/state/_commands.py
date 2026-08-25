#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named state commands for multi-record Runtime checkpoints."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from linktools.core import environ
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
)

from ...core import (
    ExecutionEventType,
    ExecutionStatus,
    IdempotencyStatus,
    SessionStatus,
    ToolOperationStatus,
    canonical_json_bytes,
    step_run_id,
)
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._tool import ToolOperationRecord
from ._contracts import (
    AgentAttemptClaim,
    ConversationCursor,
    ConversationHistoryRecord,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionHistorySealRecord,
    ExecutionHistoryState,
    ExecutionRecord,
    ExecutionRunSealHead,
    ExecutionStartClaim,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    HistoryQuality,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    SessionRecord,
    ToolOperationAdmission,
)
from ._durability import (
    CommitObservation,
    DurableCommitState,
    run_durable_commit,
)
from ._repositories import (
    ConversationHistoryRepositoryImpl,
    EventRepositoryImpl,
    ExecutionRepositoryImpl,
    OperationLedgerRepository,
    RecoveryCheckpointRepositoryImpl,
    SessionRepositoryImpl,
    ToolRepositoryImpl,
)
from ._steps import (
    PreparedExecutionProjection,
    PreparedStepSnapshot,
    PreparedStepSnapshotBatch,
    StateStepArchive,
)
from ._store import StateGroupTransaction, StateStore, StateTransaction

_logger = environ.get_logger("ai.runtime.state.commands")


class RuntimeStateCommands:
    """Named semantic commands that may enlist several Runtime domains."""

    def __init__(
        self,
        execution: ExecutionRepositoryImpl,
        *,
        namespace: str,
        events: EventRepositoryImpl,
        operations: OperationLedgerRepository | None = None,
        conversation: SessionRepositoryImpl | None = None,
        recovery: RecoveryCheckpointRepositoryImpl | None = None,
        conversation_history: ConversationHistoryRepositoryImpl | None = None,
        tools: ToolRepositoryImpl | None = None,
        conversation_steps: StateStepArchive | None = None,
        execution_steps: StateStepArchive | None = None,
        recovery_steps: StateStepArchive | None = None,
        background_tasks: "set[asyncio.Task[object]]",
    ) -> None:
        self._execution = execution
        self._namespace = namespace
        self._events = events
        self._operations = operations
        self._conversation = conversation
        self._recovery = recovery
        self._conversation_history = conversation_history
        self._tools = tools
        self._conversation_steps = conversation_steps
        self._execution_steps = execution_steps
        self._recovery_steps = recovery_steps
        self._background_tasks = background_tasks

    async def _commit_or_raise(
        self,
        operation: Callable[[], Awaitable[object]],
        readback: Callable[[], Awaitable[CommitObservation[object]]],
    ) -> None:
        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            return
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "durable command left partial state",
            ) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def _promote_history_in_transaction(
        self,
        transaction: StateTransaction,
        session: SessionRecord,
        prepared: PreparedStepSnapshot,
    ) -> ConversationHistoryRecord:
        if self._conversation_history is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        history_id = session.history_id or prepared.owner_id
        history = await self._conversation_history.get_in_transaction(
            transaction,
            history_id,
            tenant_id=session.tenant_id,
        )
        if history is None:
            history = ConversationHistoryRecord(
                history_id=history_id,
                session_id=session.session_id,
                tenant_id=session.tenant_id,
                parent_history_id=None,
                prefix_index_head_id=None,
                inherited_message_count=0,
                inherited_history_item_count=0,
            )
            await self._conversation_history.create_in_transaction(
                transaction,
                history,
            )
        return history



    async def commit_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
        audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord:
        expected_events = tuple(audit_events) + (
            ExecutionEventAppend(
                ExecutionEventType.CANCEL_REQUESTED,
                {"operation_id": commit.operation_id},
            ),
        )
        target_revision = commit.expected_revision + len(expected_events)
        target_sequence = commit.expected_event_sequence + len(expected_events)

        async def operation() -> ExecutionRecord:
            return await self._execution.request_cancel(commit, pending_events=audit_events)

        async def readback() -> CommitObservation[ExecutionRecord]:
            try:
                execution = await self._execution.get(
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                )
                if execution is None:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                page = await self._events.list(
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                    after_sequence=commit.expected_event_sequence,
                    limit=len(expected_events) + 1,
                )
                items = page.items
                if any(
                    event.sequence != commit.expected_event_sequence + index
                    for index, event in enumerate(items, 1)
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                revision_delta = execution.revision - commit.expected_revision
                sequence_delta = execution.event_sequence - commit.expected_event_sequence
                if (
                    revision_delta < 0
                    or sequence_delta < 0
                    or revision_delta < sequence_delta
                    or items and items[-1].sequence > execution.event_sequence
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                prefix = items[:len(expected_events)]
                prefix_matches = (
                    len(prefix) == len(expected_events)
                    and all(
                        actual.event_type is expected.event_type
                        and actual.payload == expected.payload
                        for actual, expected in zip(prefix, expected_events)
                    )
                )
                if prefix_matches:
                    if execution.revision < target_revision or execution.event_sequence < target_sequence:
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                        )
                    if execution.status in {
                        ExecutionStatus.CANCELLING,
                        ExecutionStatus.FINALIZING,
                        ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                    }:
                        return CommitObservation(DurableCommitState.COMMITTED, value=execution)
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                if revision_delta == 0 and sequence_delta == 0:
                    if execution.status is expected_status and not items:
                        return CommitObservation(DurableCommitState.NOT_COMMITTED)
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                if sequence_delta > 0 and not items:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.STORAGE_CONFLICT),
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        outcome = await run_durable_commit(
            operation,
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
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error

    async def commit_start_checkpoint(
        self,
        claim: ExecutionStartClaim,
        *,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
    ) -> ExecutionRecord:
        _logger.debug(
            "start checkpoint requested: execution=%s session=%s recovery=%s",
            claim.execution_id,
            session_id,
            recovery_checkpoint is not None,
        )
        stores = [self._execution.state_store]
        if recovery_checkpoint is not None:
            self._require_recovery()
            stores.append(self._recovery.state_store)
        if session_id is not None:
            self._require_conversation()
            stores.append(self._conversation.state_store)
        if _same_group(stores):
            async def callback(group: StateGroupTransaction) -> ExecutionRecord:
                transaction = group.transaction(self._execution.state_store)
                if recovery_checkpoint is not None:
                    await self._recovery.admit_in_transaction(
                        group.transaction(self._recovery.state_store),
                        recovery_checkpoint,
                    )
                if session_id is not None:
                    session_transaction = group.transaction(self._conversation.state_store)
                    session = await self._conversation.get_in_transaction(
                        session_transaction,
                        session_id,
                        tenant_id=claim.tenant_id,
                    )
                    if session.active_execution_id not in {None, claim.execution_id}:
                        owner = await self._execution.get_in_transaction(
                            transaction,
                            session.active_execution_id,
                            tenant_id=claim.tenant_id,
                        )
                        if (
                            owner is None
                            or owner.tenant_id != claim.tenant_id
                            or owner.session_id != session_id
                            or owner.parent_execution_id is not None
                        ):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if owner.status not in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            if session.status is not SessionStatus.OPEN:
                                raise AIError(ErrorCode.SESSION_CONFLICT)
                            raise AIError(ErrorCode.SESSION_BUSY)
                        await self._conversation.release_execution_in_transaction(
                            session_transaction,
                            session_id,
                            tenant_id=claim.tenant_id,
                            execution_id=owner.execution_id,
                        )
                    await self._conversation.admit_execution_in_transaction(
                        session_transaction,
                        session_id,
                        tenant_id=claim.tenant_id,
                        execution_id=claim.execution_id,
                        expected=expected_cursor,
                    )
                return await self._execution.claim_start_in_transaction(transaction, claim)

            try:
                return await stores[0].storage_group.mutate(stores, callback)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise
                observed = await self._reconcile_start_unknown(
                    claim,
                    recovery_checkpoint=recovery_checkpoint,
                    session_id=session_id,
                    expected_cursor=expected_cursor,
                )
                if observed is not None:
                    return observed
                return await stores[0].storage_group.mutate(stores, callback)
        if recovery_checkpoint is not None:
            for attempt in range(2):
                try:
                    await self._recovery.state_store.mutate(
                        lambda transaction: self._recovery.admit_in_transaction(
                            transaction,
                            recovery_checkpoint,
                        )
                    )
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    actual = await self._recovery.get(
                        recovery_checkpoint.execution_id,
                        tenant_id=recovery_checkpoint.tenant_id,
                    )
                    if actual == recovery_checkpoint:
                        break
        if session_id is not None:
            for attempt in range(2):
                try:
                    await self._admit_split_start(
                        session_id,
                        tenant_id=claim.tenant_id,
                        execution_id=claim.execution_id,
                        expected=expected_cursor,
                    )
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    session = await self._conversation.get(
                        session_id,
                        tenant_id=claim.tenant_id,
                    )
                    if (
                        session is not None
                        and session.active_execution_id == claim.execution_id
                        and session.continuation == expected_cursor
                    ):
                        break
        return await self._execution.claim_start(claim)

    async def commit_agent_attempt_checkpoint(
        self,
        claim: AgentAttemptClaim,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        """Atomically publish the execution sequence and active recovery run."""
        self._require_recovery()
        stores = [self._execution.state_store, self._recovery.state_store]
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        next_sequence = claim.expected_agent_run_sequence + 1
        next_run_id = step_run_id(
            namespace=self._namespace,
            tenant_id=claim.tenant_id,
            execution_id=claim.execution_id,
            segment_sequence=next_sequence,
        )

        async def callback(
            group: StateGroupTransaction,
        ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            execution_transaction = group.transaction(self._execution.state_store)
            recovery_transaction = group.transaction(self._recovery.state_store)
            current_execution = await self._execution.get_in_transaction(
                execution_transaction,
                claim.execution_id,
                tenant_id=claim.tenant_id,
            )
            if current_execution is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current_recovery = await self._recovery.get_in_transaction(
                recovery_transaction,
                claim.execution_id,
                tenant_id=claim.tenant_id,
            )
            if current_recovery is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if (
                current_execution.revision != claim.expected_execution_revision
                or current_execution.agent_run_sequence
                != claim.expected_agent_run_sequence
                or current_recovery.revision != claim.expected_recovery_revision
                or current_recovery.state is not claim.expected_recovery_state
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current_recovery.state is not RecoveryCheckpointState.ADMITTED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            updated_execution = await self._execution.claim_next_agent_run_in_transaction(
                execution_transaction,
                claim.execution_id,
                tenant_id=claim.tenant_id,
                expected_revision=claim.expected_execution_revision,
                expected_agent_run_sequence=claim.expected_agent_run_sequence,
            )
            updated_recovery = replace(
                current_recovery,
                step_run_id=next_run_id,
                agent_run_sequence=next_sequence,
                state=RecoveryCheckpointState.ACTIVE,
                revision=current_recovery.revision + 1,
                updated_at=updated_execution.updated_at,
            )
            updated_recovery = await self._recovery.compare_and_swap_in_transaction(
                recovery_transaction,
                claim.execution_id,
                tenant_id=claim.tenant_id,
                expected_revision=claim.expected_recovery_revision,
                next_record=updated_recovery,
            )
            return updated_execution, updated_recovery

        _logger.debug(
            "agent attempt checkpoint requested: execution=%s sequence=%s",
            claim.execution_id,
            next_sequence,
        )
        try:
            return await stores[0].storage_group.mutate(stores, callback)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            execution = await self._execution.get(
                claim.execution_id,
                tenant_id=claim.tenant_id,
            )
            recovery = await self._recovery.get(
                claim.execution_id,
                tenant_id=claim.tenant_id,
            )
            execution_target = (
                execution is not None
                and execution.revision == claim.expected_execution_revision + 1
                and execution.agent_run_sequence == next_sequence
            )
            recovery_target = (
                recovery is not None
                and recovery.revision == claim.expected_recovery_revision + 1
                and recovery.state is RecoveryCheckpointState.ACTIVE
                and recovery.step_run_id == next_run_id
                and recovery.agent_run_sequence == next_sequence
            )
            execution_predecessor = (
                execution is not None
                and execution.revision == claim.expected_execution_revision
                and execution.agent_run_sequence == claim.expected_agent_run_sequence
            )
            recovery_predecessor = (
                recovery is not None
                and recovery.revision == claim.expected_recovery_revision
                and recovery.state is claim.expected_recovery_state
                and recovery.step_run_id is None
                and recovery.agent_run_sequence == claim.expected_agent_run_sequence
            )
            if execution_target and recovery_target:
                _logger.warning(
                    "agent attempt commit outcome reconciled: execution=%s sequence=%s",
                    claim.execution_id,
                    next_sequence,
                )
                return execution, recovery
            if not (execution_predecessor and recovery_predecessor):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            return await stores[0].storage_group.mutate(stores, callback)

    async def commit_start_attempt_checkpoint(
        self,
        claim: ExecutionStartClaim,
        *,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
    ) -> ExecutionRecord:
        """Commit the initial execution start and optional recovery admission."""
        return await self.commit_start_checkpoint(
            claim,
            recovery_checkpoint=recovery_checkpoint,
            session_id=session_id,
            expected_cursor=expected_cursor,
        )

    async def commit_tool_terminal(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> ToolOperationRecord:
        if self._tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if result_payload is None and error_code is None:
            raise ValueError("tool terminal command requires a result or error")
        expected_status = (
            ToolOperationStatus.COMPLETED
            if result_payload is not None
            else ToolOperationStatus.FAILED
        )
        terminal_error_code = error_code or ErrorCode.EXECUTION_FAILED.value
        _logger.debug(
            "tool terminal checkpoint requested: operation=%s status=%s",
            tool_operation_id,
            expected_status.value,
        )
        stores = [self._tools.state_store]

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            transaction = group.transaction(self._tools.state_store)
            if result_payload is not None:
                return await self._tools.complete_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    result_payload=result_payload,
                )
            return await self._tools.fail_in_transaction(
                transaction,
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                error_code=error_code or ErrorCode.EXECUTION_FAILED.value,
                error_payload=error_payload,
            )

        async def readback() -> CommitObservation[ToolOperationRecord]:
            try:
                observed = await self._tools.get_operation(
                    tool_operation_id,
                    tenant_id=tenant_id,
                )
                if observed is None:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                payload_matches = (
                    observed.result_payload == result_payload
                    if result_payload is not None
                    else observed.error_code == terminal_error_code
                    and observed.error_payload == error_payload
                )
                if observed.status is expected_status:
                    if not payload_matches:
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.TOOL_RESULT_CONFLICT),
                        )
                    return CommitObservation(
                        DurableCommitState.COMMITTED,
                        value=observed,
                    )
                if (
                    observed.status is ToolOperationStatus.CLAIMED
                    and observed.owner == owner
                    and observed.fence == fence
                ):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(DurableCommitState.UNRESOLVED)
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

        result = await run_durable_commit(
            lambda: stores[0].storage_group.mutate(stores, callback),
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "tool terminal commit left partial durable state",
            ) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def commit_tool_admission(
        self,
        request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        if self._tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        _logger.debug(
            "tool admission checkpoint requested: operation=%s",
            request.tool_operation_id,
        )
        stores = [self._tools.state_store]

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            operation = await self._tools.admit_in_transaction(
                group.transaction(self._tools.state_store),
                request,
            )
            return operation

        try:
            return await stores[0].storage_group.mutate(stores, callback)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            observed = await self._tools.get_operation(
                request.tool_operation_id,
                tenant_id=request.tenant_id,
            )
            if observed is None:
                return await stores[0].storage_group.mutate(stores, callback)
            if not _tool_admission_matches(observed, request):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
            if observed.status is ToolOperationStatus.CANCELLED:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
            if observed.status in {
                ToolOperationStatus.CLAIMED,
                ToolOperationStatus.COMPLETED,
                ToolOperationStatus.FAILED,
            }:
                return observed
            return await stores[0].storage_group.mutate(stores, callback)

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
        next_cursor: ConversationCursor | None = None,
        conversation_run: RunRecord | None = None,
        conversation_snapshot: ContinuableSnapshot | None = None,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
        recovery_run: RunRecord | None = None,
        recovery_snapshot: ContinuableSnapshot | None = None,
        execution_run: RunRecord | None = None,
        execution_events: Sequence[StepEvent] = (),
        execution_snapshots: Sequence[ContinuableSnapshot] = (),
        execution_projections: Sequence[PreparedExecutionProjection] = (),
        audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionTerminalCommitResult:
        if execution_projections and (
            self._execution_steps is None or execution_run is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is None and (conversation_run is not None or conversation_snapshot is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is None and (recovery_run is not None or recovery_snapshot is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "terminal checkpoint requested: execution=%s session=%s recovery=%s",
            commit.execution.execution_id,
            session_id,
            recovery_checkpoint is not None,
        )
        prepared_conversation = ()
        if (
            self._conversation_steps is not None
            and conversation_run is not None
            and conversation_snapshot is not None
        ):
            prepared_conversation = await self._conversation_steps.prepare_snapshots(
                conversation_run,
                (conversation_snapshot,),
            )
        prepared_recovery = ()
        if (
            self._recovery_steps is not None
            and recovery_run is not None
            and recovery_snapshot is not None
        ):
            prepared_recovery = await self._recovery_steps.prepare_snapshots(
                recovery_run,
                (recovery_snapshot,),
            )
        prepared_execution: PreparedStepSnapshotBatch | tuple[()] = ()
        if (
            not execution_projections
            and self._execution_steps is not None
            and execution_run is not None
            and execution_snapshots
        ):
            prepared_execution = await self._execution_steps.prepare_snapshots(
                execution_run,
                execution_snapshots,
            )
        seal = _execution_history_seal(
            commit,
            audit_events=audit_events,
            projections=execution_projections,
            current_run=execution_run,
            current_events=execution_events,
            current_batch=(
                prepared_execution
                if isinstance(prepared_execution, PreparedStepSnapshotBatch)
                else None
            ),
        )
        stores = [self._execution.state_store]
        if session_id is not None:
            self._require_conversation()
            stores.append(self._conversation.state_store)
        if recovery_checkpoint is not None:
            self._require_recovery()
            stores.append(self._recovery.state_store)
        if execution_run is not None:
            if self._execution_steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.append(self._execution_steps.state_store)
        if conversation_run is not None or conversation_snapshot is not None:
            if self._conversation_steps is None or conversation_run is None or conversation_snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.append(self._conversation_steps.state_store)
        if recovery_run is not None or recovery_snapshot is not None:
            if self._recovery_steps is None or recovery_run is None or recovery_snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.append(self._recovery_steps.state_store)
        if _same_group(stores):
            async def callback(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
                execution_transaction = group.transaction(self._execution.state_store)
                head, head_record = await self._execution.require_open_history_head_in_transaction(
                    execution_transaction,
                    commit.execution.execution_id,
                )
                effective_commit = await self._effective_terminal_commit(
                    execution_transaction,
                    commit,
                )
                if session_id is not None:
                    conversation_transaction = group.transaction(self._conversation.state_store)
                    if conversation_run is None or conversation_snapshot is None:
                        await self._conversation.release_execution_in_transaction(
                            conversation_transaction,
                            session_id,
                            tenant_id=commit.execution.tenant_id,
                            execution_id=commit.execution.execution_id,
                        )
                    else:
                        if next_cursor is None or self._conversation_steps is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await self._conversation_steps.materialize_snapshot_in_transaction(
                            group.transaction(self._conversation_steps.state_store),
                            conversation_run,
                            prepared_conversation[0],
                        )
                        session = await self._conversation.get_in_transaction(
                            conversation_transaction,
                            session_id,
                            tenant_id=commit.execution.tenant_id,
                        )
                        await self._promote_history_in_transaction(
                            conversation_transaction,
                            session,
                            prepared_conversation[0],
                        )
                        await self._conversation.complete_execution_in_transaction(
                            conversation_transaction,
                            session_id,
                            tenant_id=commit.execution.tenant_id,
                            execution_id=commit.execution.execution_id,
                            expected=expected_cursor,
                            next_cursor=next_cursor,
                            history_quality="complete",
                        )
                if recovery_checkpoint is not None:
                    if recovery_run is not None or recovery_snapshot is not None:
                        if self._recovery_steps is None or recovery_run is None or recovery_snapshot is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await self._recovery_steps.materialize_snapshot_in_transaction(
                            group.transaction(self._recovery_steps.state_store),
                            recovery_run,
                            prepared_recovery[0],
                        )
                    await self._recovery.compare_and_swap_in_transaction(
                        group.transaction(self._recovery.state_store),
                        recovery_checkpoint.execution_id,
                        tenant_id=recovery_checkpoint.tenant_id,
                        expected_revision=recovery_checkpoint.revision - 1,
                        next_record=recovery_checkpoint,
                    )
                if execution_run is not None:
                    execution_transaction_for_steps = group.transaction(
                        self._execution_steps.state_store
                    )
                    if execution_projections:
                        for projection in execution_projections:
                            await self._execution_steps.sync_projection_in_transaction(
                                execution_transaction_for_steps,
                                projection.run,
                                events=projection.events,
                                snapshots=projection.snapshots,
                                execution_id=commit.execution.execution_id,
                                history_head_guard=(head, head_record),
                            )
                    else:
                        await self._execution_steps.sync_projection_in_transaction(
                            execution_transaction_for_steps,
                            execution_run,
                            events=execution_events,
                            snapshots=(
                                prepared_execution.snapshots
                                if isinstance(
                                    prepared_execution,
                                    PreparedStepSnapshotBatch,
                                )
                                else ()
                            ),
                            execution_id=commit.execution.execution_id,
                            history_head_guard=(head, head_record),
                        )
                await self._execution.put_history_seal_in_transaction(
                    execution_transaction,
                    seal,
                )
                await self._execution.replace_history_head_in_transaction(
                    execution_transaction,
                    head_record,
                    replace(
                        head,
                        state=ExecutionHistoryState.SEALED,
                        revision=head.revision + 1,
                        seal_digest=seal.seal_digest,
                    ),
                )
                return await self._execution.commit_terminal_in_transaction(
                    execution_transaction,
                    effective_commit,
                    pending_events=audit_events,
                )

            async def readback() -> CommitObservation[ExecutionTerminalCommitResult]:
                try:
                    visible = await self._terminal_targets_visible(
                        commit,
                        pending_event_count=len(audit_events),
                        session_id=session_id,
                        next_cursor=next_cursor,
                        conversation_run=conversation_run,
                        conversation_snapshot=conversation_snapshot,
                        recovery_checkpoint=recovery_checkpoint,
                        recovery_run=recovery_run,
                        recovery_snapshot=recovery_snapshot,
                        execution_run=execution_run,
                        execution_events=execution_events,
                        execution_snapshots=execution_snapshots,
                        execution_projections=execution_projections,
                        audit_events=audit_events,
                    )
                    if not visible:
                        return CommitObservation(DurableCommitState.NOT_COMMITTED)
                    execution = await self._execution.get(
                        commit.execution.execution_id,
                        tenant_id=commit.execution.tenant_id,
                    )
                    if execution is None:
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                        )
                    return CommitObservation(
                        DurableCommitState.COMMITTED,
                        value=ExecutionTerminalCommitResult(
                            execution,
                            commit.result,
                        ),
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

            result = await run_durable_commit(
                lambda: stores[0].storage_group.mutate(stores, callback),
                readback,
                background_tasks=self._background_tasks,
            )
            if result.state is DurableCommitState.COMMITTED:
                if result.value is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if result.cancelled:
                    raise asyncio.CancelledError
                return result.value
            if result.state is DurableCommitState.NOT_COMMITTED:
                if result.error is not None:
                    raise result.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "terminal checkpoint left partial durable state",
                ) from result.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error
        recovery_pending = False
        if recovery_checkpoint is not None:
            actual_recovery = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=recovery_checkpoint.tenant_id,
            )
            if actual_recovery is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if actual_recovery == recovery_checkpoint:
                if not await self._terminal_targets_visible(
                    commit,
                    pending_event_count=len(audit_events),
                    session_id=session_id,
                    next_cursor=next_cursor,
                    conversation_run=conversation_run,
                    conversation_snapshot=conversation_snapshot,
                    recovery_checkpoint=recovery_checkpoint,
                    recovery_run=recovery_run,
                    recovery_snapshot=recovery_snapshot,
                    execution_run=execution_run,
                    execution_events=execution_events,
                    execution_snapshots=execution_snapshots,
                    execution_projections=execution_projections,
                    audit_events=audit_events,
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution = await self._execution.get(
                    commit.execution.execution_id,
                    tenant_id=commit.execution.tenant_id,
                )
                if execution is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return ExecutionTerminalCommitResult(execution, commit.result)
            if not _recovery_completion_predecessor(actual_recovery, recovery_checkpoint):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            recovery_pending = True
        recovery_steps_merged = False
        if session_id is not None and conversation_run is not None and conversation_snapshot is not None:
            if next_cursor is None or self._conversation_steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            conversation_stores = [
                self._conversation.state_store,
                self._conversation_steps.state_store,
            ]

            async def commit_conversation(group: StateGroupTransaction) -> None:
                conversation_transaction = group.transaction(self._conversation.state_store)
                await self._conversation_steps.materialize_snapshot_in_transaction(
                    group.transaction(self._conversation_steps.state_store),
                    conversation_run,
                    prepared_conversation[0],
                )
                session = await self._conversation.get_in_transaction(
                    conversation_transaction,
                    session_id,
                    tenant_id=commit.execution.tenant_id,
                )
                await self._promote_history_in_transaction(
                    conversation_transaction,
                    session,
                    prepared_conversation[0],
                )
                await self._conversation.advance_continuation_in_transaction(
                    group.transaction(self._conversation.state_store),
                    session_id,
                    tenant_id=commit.execution.tenant_id,
                    execution_id=commit.execution.execution_id,
                    expected=expected_cursor,
                    next_cursor=next_cursor,
                    release_execution=False,
                    history_quality="complete",
                )

            if _same_group(conversation_stores):
                for attempt in range(2):
                    try:
                        await self._conversation.state_store.storage_group.mutate(
                            conversation_stores,
                            commit_conversation,
                        )
                        break
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                            raise
                        if await self._conversation_target_visible(
                            session_id,
                            tenant_id=commit.execution.tenant_id,
                            execution_id=commit.execution.execution_id,
                            next_cursor=next_cursor,
                            run=conversation_run,
                            snapshot=conversation_snapshot,
                        ):
                            break
            else:
                await self._materialize_prepared_snapshot_with_reconciliation(
                    self._conversation_steps,
                    conversation_run,
                    prepared_conversation,
                )
                for attempt in range(2):
                    try:
                        await self._conversation.advance_continuation(
                            session_id,
                            tenant_id=commit.execution.tenant_id,
                            execution_id=commit.execution.execution_id,
                            expected=expected_cursor,
                            next_cursor=next_cursor,
                        )
                        break
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                            raise
                        session = await self._conversation.get(
                            session_id,
                            tenant_id=commit.execution.tenant_id,
                        )
                        if session is not None and session.continuation == next_cursor:
                            break

        execution_stores = [self._execution.state_store]
        if execution_run is not None:
            if self._execution_steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution_stores.append(self._execution_steps.state_store)
        recovery_state_merged = (
            recovery_pending
            and recovery_checkpoint is not None
            and self._recovery.state_store.storage_group is self._execution.state_store.storage_group
        )
        if recovery_state_merged:
            if recovery_run is not None or recovery_snapshot is not None:
                if self._recovery_steps is None or recovery_run is None or recovery_snapshot is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                recovery_steps_merged = (
                    self._recovery_steps.state_store.storage_group
                    is self._execution.state_store.storage_group
                )
            execution_stores.append(self._recovery.state_store)
            if recovery_steps_merged:
                execution_stores.append(self._recovery_steps.state_store)
        if recovery_pending and not recovery_steps_merged:
            if self._recovery_steps is None or recovery_run is None or recovery_snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._materialize_prepared_snapshot_with_reconciliation(
                self._recovery_steps,
                recovery_run,
                prepared_recovery,
            )

        execution_steps_in_transaction = False

        async def commit_execution(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
            execution_transaction = group.transaction(self._execution.state_store)
            head, head_record = await self._execution.require_open_history_head_in_transaction(
                execution_transaction,
                commit.execution.execution_id,
            )
            effective_commit = await self._effective_terminal_commit(execution_transaction, commit)
            if execution_run is not None and execution_steps_in_transaction:
                execution_transaction_for_steps = group.transaction(
                    self._execution_steps.state_store
                )
                if execution_projections:
                    for projection in execution_projections:
                        await self._execution_steps.sync_projection_in_transaction(
                            execution_transaction_for_steps,
                            projection.run,
                            events=projection.events,
                            snapshots=projection.snapshots,
                            execution_id=commit.execution.execution_id,
                            history_head_guard=(head, head_record),
                        )
                else:
                    await self._execution_steps.sync_projection_in_transaction(
                        execution_transaction_for_steps,
                        execution_run,
                        events=execution_events,
                        snapshots=(
                            prepared_execution.snapshots
                            if isinstance(
                                prepared_execution,
                                PreparedStepSnapshotBatch,
                            )
                            else ()
                        ),
                        execution_id=commit.execution.execution_id,
                        history_head_guard=(head, head_record),
                    )
            await self._execution.put_history_seal_in_transaction(
                execution_transaction,
                seal,
            )
            await self._execution.replace_history_head_in_transaction(
                execution_transaction,
                head_record,
                replace(
                    head,
                    state=ExecutionHistoryState.SEALED,
                    revision=head.revision + 1,
                    seal_digest=seal.seal_digest,
                ),
            )
            if recovery_state_merged:
                if recovery_steps_merged:
                    await self._recovery_steps.materialize_snapshot_in_transaction(
                        group.transaction(self._recovery_steps.state_store),
                        recovery_run,
                        prepared_recovery[0],
                    )
                await self._recovery.compare_and_swap_in_transaction(
                    group.transaction(self._recovery.state_store),
                    recovery_checkpoint.execution_id,
                    tenant_id=recovery_checkpoint.tenant_id,
                    expected_revision=recovery_checkpoint.revision - 1,
                    next_record=recovery_checkpoint,
                )
            return await self._execution.commit_terminal_in_transaction(
                execution_transaction,
                effective_commit,
                pending_events=audit_events,
            )

        async def execution_target_visible() -> bool:
            return await self._terminal_targets_visible(
                commit,
                pending_event_count=len(audit_events),
                session_id=session_id,
                next_cursor=next_cursor,
                conversation_run=conversation_run,
                conversation_snapshot=conversation_snapshot,
                recovery_checkpoint=None,
                recovery_run=None,
                recovery_snapshot=None,
                execution_run=execution_run,
                execution_events=execution_events,
                execution_snapshots=execution_snapshots,
                execution_projections=execution_projections,
                audit_events=audit_events,
                require_session_release=False,
                require_recovery=False,
            )

        async def commit_execution_with_reconciliation(
            stores: Sequence[StateStore],
        ) -> ExecutionTerminalCommitResult:
            async def readback() -> CommitObservation[ExecutionTerminalCommitResult]:
                try:
                    if not await execution_target_visible():
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED
                        )
                    execution = await self._execution.get(
                        commit.execution.execution_id,
                        tenant_id=commit.execution.tenant_id,
                    )
                    if execution is None:
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                        )
                    return CommitObservation(
                        DurableCommitState.COMMITTED,
                        value=ExecutionTerminalCommitResult(
                            execution,
                            commit.result,
                        ),
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

            outcome = await run_durable_commit(
                lambda: self._execution.state_store.storage_group.mutate(
                    stores,
                    commit_execution,
                ),
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

        if await execution_target_visible():
            execution = await self._execution.get(
                commit.execution.execution_id,
                tenant_id=commit.execution.tenant_id,
            )
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = ExecutionTerminalCommitResult(execution, commit.result)
        elif _same_group(execution_stores):
            execution_steps_in_transaction = execution_run is not None
            result = await commit_execution_with_reconciliation(execution_stores)
        else:
            if execution_projections:
                for projection in execution_projections:
                    await self._sync_prepared_projection_with_reconciliation(
                        self._execution_steps,
                        projection,
                        execution_id=commit.execution.execution_id,
                    )
            elif execution_run is not None:
                await self._sync_projection_with_reconciliation(
                    execution_run,
                    execution_id=commit.execution.execution_id,
                    events=execution_events,
                    snapshots=execution_snapshots,
                )
            terminal_stores = [self._execution.state_store]
            if recovery_state_merged:
                terminal_stores.append(self._recovery.state_store)
                if recovery_steps_merged:
                    terminal_stores.append(self._recovery_steps.state_store)
            result = await commit_execution_with_reconciliation(terminal_stores)
        if recovery_state_merged and recovery_checkpoint is not None:
            actual_recovery = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=recovery_checkpoint.tenant_id,
            )
            if actual_recovery == recovery_checkpoint:
                recovery_pending = False
            elif actual_recovery is not None and not _recovery_completion_predecessor(
                actual_recovery,
                recovery_checkpoint,
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            for attempt in range(2):
                try:
                    await self._conversation.release_execution(
                        session_id,
                        tenant_id=commit.execution.tenant_id,
                        execution_id=commit.execution.execution_id,
                    )
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    session = await self._conversation.get(
                        session_id,
                        tenant_id=commit.execution.tenant_id,
                    )
                    if session is not None and session.active_execution_id is None:
                        break
        if recovery_pending and recovery_checkpoint is not None:
            await self._complete_recovery_checkpoint(recovery_checkpoint)
        return result

    async def _materialize_recovery_snapshot(
        self,
        run: RunRecord | None,
        snapshot: ContinuableSnapshot | None,
    ) -> None:
        if run is None and snapshot is None:
            return
        if self._recovery_steps is None or run is None or snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if await self._step_snapshot_visible(
            self._recovery_steps,
            run,
            snapshot,
        ):
            return
        prepared = await self._recovery_steps.prepare_snapshots(
            run,
            (snapshot,),
        )
        stores = [self._recovery.state_store, self._recovery_steps.state_store]

        async def materialize(group: StateGroupTransaction) -> None:
            await self._recovery_steps.materialize_snapshot_in_transaction(
                group.transaction(self._recovery_steps.state_store),
                run,
                prepared[0],
            )

        if _same_group(stores):
            async def readback() -> CommitObservation[None]:
                try:
                    visible = await self._step_snapshot_visible(
                        self._recovery_steps,
                        run,
                        snapshot,
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
                return CommitObservation(
                    DurableCommitState.COMMITTED
                    if visible
                    else DurableCommitState.NOT_COMMITTED
                )

            await self._commit_or_raise(
                lambda: stores[0].storage_group.mutate(stores, materialize),
                readback,
            )
        else:
            await self._materialize_snapshot_with_reconciliation(
                self._recovery_steps,
                run,
                snapshot,
            )

    async def _materialize_snapshot_with_reconciliation(
        self,
        archive: StateStepArchive,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        if await self._step_snapshot_visible(
            archive,
            run,
            snapshot,
        ):
            return
        async def operation() -> None:
            await archive.materialize_snapshot(
                run,
                snapshot,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                visible = await self._step_snapshot_visible(
                    archive,
                    run,
                    snapshot,
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        await self._commit_or_raise(operation, readback)

    async def _materialize_prepared_snapshot_with_reconciliation(
        self,
        archive: StateStepArchive,
        run: RunRecord,
        batch: PreparedStepSnapshotBatch,
        *,
        execution_id: str | None = None,
    ) -> None:
        if len(batch) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = batch[0]
        if await self._prepared_snapshot_visible(archive, run, prepared):
            return

        async def materialize(transaction: StateTransaction) -> None:
            await archive.materialize_snapshot_in_transaction(
                transaction,
                run,
                prepared,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                visible = await self._prepared_snapshot_visible(archive, run, prepared)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        await self._commit_or_raise(materialize, readback)

    async def _sync_prepared_projection_with_reconciliation(
        self,
        archive: StateStepArchive | None,
        projection: PreparedExecutionProjection,
        *,
        execution_id: str,
    ) -> None:
        if archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def sync(transaction: StateTransaction) -> None:
            await archive.sync_projection_in_transaction(
                transaction,
                projection.run,
                events=projection.events,
                snapshots=projection.snapshots,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                visible = await archive.verify_execution_projection_head(projection)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        await self._commit_or_raise(sync, readback)

    async def _prepared_snapshot_visible(
        self,
        archive: StateStepArchive,
        run: RunRecord,
        prepared: PreparedStepSnapshot,
    ) -> bool:
        if await archive.get_run(run_id=run.run_id) != run:
            return False
        snapshot = await archive.latest_snapshot(
            run_id=run.run_id,
            include_interrupted=True,
        )
        if snapshot is None:
            return False
        projection = await archive.transcript_repository.load_projection(
            prepared.owner_id
        )
        return (
            snapshot.step_index == prepared.stored.step_index
            and snapshot.timestamp == prepared.stored.timestamp
            and snapshot.state == prepared.stored.state
            and projection is not None
            and projection.digest == prepared.projection.digest
        )

    async def _step_snapshot_visible(
        self,
        archive: StateStepArchive,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
    ) -> bool:
        return (
            await archive.get_run(run_id=run.run_id) == run
            and await archive.verify_snapshot_projection(
                run_id=run.run_id,
                snapshot=snapshot,
            )
        )

    async def _conversation_target_visible(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        next_cursor: ConversationCursor,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
    ) -> bool:
        session = await self._conversation.get(session_id, tenant_id=tenant_id)
        return (
            session is not None
            and session.active_execution_id == execution_id
            and session.continuation == next_cursor
            and self._conversation_steps is not None
            and await self._step_snapshot_visible(
                self._conversation_steps,
                run,
                snapshot,
            )
        )

    async def _sync_projection_with_reconciliation(
        self,
        run: RunRecord,
        *,
        execution_id: str,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
    ) -> None:
        if self._execution_steps is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def operation() -> None:
            await self._execution_steps.sync_projection(
                run,
                events=events,
                snapshots=snapshots,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                stored_run = await self._execution_steps.get_run(run_id=run.run_id)
                stored_events = await self._execution_steps.list_events(
                    run_id=run.run_id
                )
                stored_snapshot = await self._execution_steps.latest_snapshot(
                    run_id=run.run_id,
                    include_interrupted=True,
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
            if stored_run is not None and stored_run != run:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            visible = (
                stored_run == run
                and (
                    not events
                    or tuple(stored_events[-len(events) :]) == tuple(events)
                )
                and (not snapshots or stored_snapshot == snapshots[-1])
            )
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            return
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "execution projection left partial durable state",
            ) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def _complete_recovery_checkpoint(self, target: RecoveryCheckpoint) -> None:
        current = await self._recovery.get(target.execution_id, tenant_id=target.tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current == target:
            return
        if not _recovery_completion_predecessor(current, target):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for _ in range(2):
            try:
                await self._recovery.compare_and_swap(
                    target.execution_id,
                    tenant_id=target.tenant_id,
                    expected_revision=current.revision,
                    next_record=target,
                )
                return
            except AIError as error:
                if error.code not in {
                    ErrorCode.STORAGE_COMMIT_UNKNOWN,
                    ErrorCode.STORAGE_CONFLICT,
                }:
                    raise
                observed = await self._recovery.get(target.execution_id, tenant_id=target.tenant_id)
                if observed == target:
                    return
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or observed is None:
                    raise
                if not _recovery_completion_predecessor(observed, target):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                current = observed
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    def _require_conversation(self) -> None:
        if self._conversation is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _require_recovery(self) -> None:
        if self._recovery is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def _effective_terminal_commit(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
    ) -> ExecutionTerminalCommit:
        if commit.idempotency is not None:
            return commit
        return replace(
            commit,
            idempotency=await self._execution.terminal_idempotency_in_transaction(
                transaction,
                commit,
            ),
        )

    async def _reconcile_start_unknown(
        self,
        claim: ExecutionStartClaim,
        *,
        recovery_checkpoint: RecoveryCheckpoint | None,
        session_id: str | None,
        expected_cursor: ConversationCursor | None,
    ) -> ExecutionRecord | None:
        execution = await self._execution.get(
            claim.execution_id,
            tenant_id=claim.tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.status is ExecutionStatus.PENDING_START
            and execution.revision == claim.expected_revision
            and execution.event_sequence == claim.expected_event_sequence
        ):
            return None
        if (
            execution.status is not ExecutionStatus.STARTED
            or execution.revision != claim.expected_revision + 1
            or execution.event_sequence != claim.expected_event_sequence + 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._execution.get_start_idempotency(claim)
        if idempotency is None or idempotency.status is not IdempotencyStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        started_events = await self._events.list(
            claim.execution_id,
            tenant_id=claim.tenant_id,
            after_sequence=claim.expected_event_sequence,
            limit=1,
        )
        if (
            len(started_events.items) != 1
            or started_events.items[0].sequence != claim.expected_event_sequence + 1
            or started_events.items[0].event_type is not ExecutionEventType.EXECUTION_STARTED
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is not None:
            actual = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=recovery_checkpoint.tenant_id,
            )
            if (
                actual is None
                or actual.input != recovery_checkpoint.input
                or actual.agent_run_sequence != 0
                or actual.step_run_id is not None
                or actual.state is not RecoveryCheckpointState.ADMITTED
                or actual.handoff_phase is not RecoveryHandoffPhase.NONE
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            session = await self._conversation.get(session_id, tenant_id=claim.tenant_id)
            if (
                session is None
                or session.status is not SessionStatus.OPEN
                or session.active_execution_id != claim.execution_id
                or session.continuation != expected_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return execution

    async def _terminal_targets_visible(
        self,
        commit: ExecutionTerminalCommit,
        *,
        pending_event_count: int,
        session_id: str | None,
        next_cursor: ConversationCursor | None,
        conversation_run: RunRecord | None,
        conversation_snapshot: ContinuableSnapshot | None,
        recovery_checkpoint: RecoveryCheckpoint | None,
        recovery_run: RunRecord | None,
        recovery_snapshot: ContinuableSnapshot | None,
        execution_run: RunRecord | None,
        execution_events: Sequence[StepEvent],
        execution_snapshots: Sequence[ContinuableSnapshot],
        execution_projections: Sequence[PreparedExecutionProjection],
        audit_events: Sequence[ExecutionEventAppend],
        require_session_release: bool = True,
        require_recovery: bool = True,
    ) -> bool:
        execution = await self._execution.get(
            commit.execution.execution_id,
            tenant_id=commit.execution.tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.revision == commit.expected_revision
            and execution.event_sequence == commit.expected_event_sequence
        ):
            return False
        expected_revision = commit.expected_revision + pending_event_count + 1
        expected_sequence = commit.expected_event_sequence + pending_event_count + 1
        if (
            execution.status is not commit.execution.status
            or execution.revision != expected_revision
            or execution.event_sequence != expected_sequence
            or execution.result != commit.result
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_seal = await self._execution.get_history_seal(
            commit.execution.execution_id,
            tenant_id=commit.execution.tenant_id,
        )
        if history_seal is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_head = await self._execution.get_history_head(
            commit.execution.execution_id,
            tenant_id=commit.execution.tenant_id,
        )
        if (
            history_head is None
            or history_head.state is not ExecutionHistoryState.SEALED
            or history_head.seal_digest != history_seal.seal_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_projections or execution_run is None:
            expected_seal = _execution_history_seal(
                commit,
                audit_events=audit_events,
                projections=execution_projections,
                current_run=execution_run,
                current_events=execution_events,
                current_batch=None,
            )
            if history_seal != expected_seal:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_projections:
            if self._execution_steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for projection in execution_projections:
                if not await self._execution_steps.verify_execution_projection_head(
                    projection
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._execution.get_terminal_idempotency(
            commit.execution.execution_id,
            tenant_id=commit.execution.tenant_id,
        )
        if idempotency is not None:
            expected_status = (
                IdempotencyStatus.COMPLETED
                if commit.execution.status is ExecutionStatus.SUCCEEDED
                else IdempotencyStatus.CANCELLED
                if commit.execution.status is ExecutionStatus.CANCELLED
                else IdempotencyStatus.FAILED
            )
            expected_digest = None if commit.result.output is None else commit.result.output.digest
            if (
                idempotency.status is not expected_status
                or idempotency.result_digest != expected_digest
                or idempotency.error_code != commit.execution.error_code
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if commit.operation is not None:
            if self._operations is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation = await self._operations.get(
                commit.operation.operation_id,
                tenant_id=commit.execution.tenant_id,
            )
            if (
                operation is None
                or operation.status is not commit.operation.next_status
                or operation.result_ref != commit.operation.result_ref
                or operation.result_digest != commit.operation.result_digest
                or operation.error_code != commit.operation.error_code
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = await self._events.list(
            commit.execution.execution_id,
            tenant_id=commit.execution.tenant_id,
            after_sequence=commit.expected_event_sequence,
            limit=pending_event_count + 1,
        )
        expected_events = tuple(audit_events) + (
            ExecutionEventAppend(commit.terminal_event_type, commit.terminal_event_payload),
        )
        if len(events.items) != len(expected_events) or any(
            actual.sequence != commit.expected_event_sequence + index + 1
            or actual.event_type is not expected.event_type
            or actual.payload != expected.payload
            for index, (actual, expected) in enumerate(
                zip(events.items, expected_events, strict=True)
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            session = await self._conversation.get(
                session_id,
                tenant_id=commit.execution.tenant_id,
            )
            if session is None or require_session_release and session.active_execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if conversation_run is not None and (
                next_cursor is None or session.continuation != next_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is not None and require_recovery:
            actual = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=recovery_checkpoint.tenant_id,
            )
            if actual != recovery_checkpoint:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for archive, run, snapshot in (
            (
                self._conversation_steps,
                conversation_run,
                conversation_snapshot,
            ),
            (
                self._recovery_steps,
                recovery_run,
                recovery_snapshot,
            ),
        ):
            if run is None and snapshot is None:
                continue
            if archive is None or run is None or snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_run = await archive.get_run(run_id=run.run_id)
            if not await archive.verify_snapshot_projection(
                run_id=run.run_id,
                snapshot=snapshot,
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if stored_run != run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_run is not None:
            if self._execution_steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_run = await self._execution_steps.get_run(run_id=execution_run.run_id)
            stored_events = await self._execution_steps.list_events(run_id=execution_run.run_id)
            if stored_run != execution_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution_events and tuple(stored_events[-len(execution_events) :]) != tuple(execution_events):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_snapshot = await self._execution_steps.latest_snapshot(
                run_id=execution_run.run_id,
                include_interrupted=True,
            )
            if execution_snapshots and stored_snapshot != execution_snapshots[-1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution_snapshots and not await self._execution_steps.verify_snapshot_projection(
                run_id=execution_run.run_id,
                snapshot=execution_snapshots[-1],
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return True

    async def _admit_split_start(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
    ) -> None:
        session = await self._conversation.get(session_id, tenant_id=tenant_id)
        if session is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        owner_id = session.active_execution_id
        if owner_id not in {None, execution_id}:
            owner = await self._execution.get(owner_id, tenant_id=tenant_id)
            if (
                owner is None
                or owner.tenant_id != tenant_id
                or owner.session_id != session_id
                or owner.parent_execution_id is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if owner.status not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                if session.status is not SessionStatus.OPEN:
                    raise AIError(ErrorCode.SESSION_CONFLICT)
                raise AIError(ErrorCode.SESSION_BUSY)
            await self._conversation.release_execution(
                session_id,
                tenant_id=tenant_id,
                execution_id=owner.execution_id,
            )
        await self._conversation.admit_execution(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
        )


def _same_group(stores: Sequence[StateStore]) -> bool:
    return bool(stores) and all(store.storage_group is stores[0].storage_group for store in stores[1:])


def _tool_admission_matches(
    operation: ToolOperationRecord,
    request: ToolOperationAdmission,
) -> bool:
    return (
        operation.tenant_id == request.tenant_id
        and operation.tool_operation_id == request.tool_operation_id
        and operation.step_run_id in {request.step_run_id, request.recovery_step_run_id}
        and operation.tool_call_id == request.tool_call_id
        and operation.idempotency_key_digest == request.idempotency_key_digest
        and operation.tool_name == request.tool_name
        and operation.arguments_digest == request.arguments_digest
        and operation.binding_fingerprint == request.binding_fingerprint
        and operation.replay_safe is request.replay_safe
    )


def _recovery_completion_predecessor(
    current: RecoveryCheckpoint,
    target: RecoveryCheckpoint,
) -> bool:
    return (
        current.state is RecoveryCheckpointState.HANDOFF
        and current.handoff_phase is RecoveryHandoffPhase.PREPARED
        and target.state is RecoveryCheckpointState.COMPLETED
        and target.handoff_phase is RecoveryHandoffPhase.COMPLETED
        and replace(
            current,
            state=target.state,
            handoff_phase=target.handoff_phase,
            terminal_handoff=target.terminal_handoff,
            handoff_contract_digest=target.handoff_contract_digest,
            pending_operation_id=target.pending_operation_id,
            revision=target.revision,
            updated_at=target.updated_at,
        )
        == target
    )


class ExecutionStateCommands:
    """Commit execution audit, step projection, and terminal state together."""

    def __init__(
        self,
        state_store: StateStore,
        executions: ExecutionRepositoryImpl,
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
            head, head_record = await self._executions.require_open_history_head_in_transaction(
                transaction,
                commit.execution.execution_id,
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
            expected_revision = (
                commit.expected_revision + len(audit_events) + 1
            )
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


class ConversationStateCommands:
    """Commit the durable conversation snapshot and continuation together."""

    def __init__(
        self,
        state_store: StateStore,
        sessions: SessionRepositoryImpl,
        steps: StateStepArchive | None,
        histories: ConversationHistoryRepositoryImpl | None = None,
    ) -> None:
        self._state_store = state_store
        self._sessions = sessions
        self._steps = steps
        self._histories = histories

    async def commit_snapshot_and_advance(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        step_run: RunRecord,
        snapshot: ContinuableSnapshot,
    ) -> SessionRecord:
        if self._steps is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = await self._steps.prepare_snapshots(
            step_run,
            (snapshot,),
        )

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            if self._steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = await self._sessions.get_in_transaction(
                transaction,
                session_id,
                tenant_id=tenant_id,
            )
            if current.continuation == next_cursor:
                return current
            await self._steps.sync_projection_in_transaction(
                transaction,
                step_run,
                events=(),
                snapshots=prepared.snapshots,
            )
            if self._histories is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            history_id = current.history_id or prepared[0].owner_id
            history = await self._histories.get_in_transaction(
                transaction,
                history_id,
                tenant_id=tenant_id,
            )
            if history is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            quality = (
                "conservative"
                if prepared[0].history_quality is HistoryQuality.CONSERVATIVE
                else "complete"
            )
            return await self._sessions.advance_continuation_in_transaction(
                transaction,
                session_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                expected=expected,
                next_cursor=next_cursor,
                history_quality=quality,
            )

        result = await self._state_store.mutate(mutate)
        return result


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
            )
        )
    ordered_heads = tuple(sorted(heads, key=lambda head: head.run_id))
    execution_event_high_water = (
        commit.expected_event_sequence + len(audit_events) + 1
    )
    digest_input = {
        "execution_id": commit.execution.execution_id,
        "tenant_id": commit.execution.tenant_id,
        "seal_version": 1,
        "run_heads": [
            {
                "run_id": head.run_id,
                "event_count": head.event_count,
                "snapshot_count": head.snapshot_count,
                "transcript_message_count": head.transcript_message_count,
                "projection_digest": head.projection_digest,
            }
            for head in ordered_heads
        ],
        "execution_event_high_water": execution_event_high_water,
    }
    seal_digest = hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()
    return ExecutionHistorySealRecord(
        commit.execution.execution_id,
        commit.execution.tenant_id,
        1,
        ordered_heads,
        execution_event_high_water,
        seal_digest,
    )


__all__ = [
    "ConversationStateCommands",
    "ExecutionStateCommands",
    "RuntimeStateCommands",
]
