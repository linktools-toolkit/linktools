#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named state commands for multi-record Runtime checkpoints."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import cast

from linktools.core import environ
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
)

from ...core import (
    ExecutionEventType,
    ApprovalStatus,
    ExecutionStatus,
    IdempotencyStatus,
    OperationKind,
    OperationStatus,
    ResourceKind,
    SessionStatus,
    ToolOperationStatus,
    canonical_json_bytes,
    canonical_sha256,
    step_run_id,
)
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._tool import ToolOperationRecord
from ._contracts import (
    AgentAttemptClaim,
    ApprovalRecord,
    ApprovalRepository,
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
    PendingApprovalContinuation,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    SessionRecord,
    ToolApprovalAdmission,
    ToolOperationAdmission,
)
from ._durability import (
    CommitObservation,
    DurableCommitResult,
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
    _tool_admission_matches,
)
from ._plan import RuntimeDomain
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
        approvals: ApprovalRepository | None = None,
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
        self._approvals = approvals
        self._conversation = conversation
        self._recovery = recovery
        self._conversation_history = conversation_history
        self._tools = tools
        self._conversation_steps = conversation_steps
        self._execution_steps = execution_steps
        self._recovery_steps = recovery_steps
        self._background_tasks = background_tasks

    def _require_approvals(self) -> ApprovalRepository:
        if self._approvals is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._approvals

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
        background_tasks: "set[asyncio.Task[object]] | None" = None,
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

        owner_tasks = (
            self._background_tasks
            if background_tasks is None
            else background_tasks
        )
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
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

    async def commit_approval_wait_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_execution_revision: int,
        expected_event_sequence: int,
        expected_recovery_revision: int,
        expected_agent_run_sequence: int,
        expected_previous_pending_approval: PendingApprovalContinuation | None,
        continuation: PendingApprovalContinuation,
        admissions: Sequence[ToolApprovalAdmission],
        audit_events: Sequence[ExecutionEventAppend] = (),
        recovery_run: RunRecord,
        recovery_snapshot: ContinuableSnapshot,
        occurred_at: datetime,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        approvals = self._require_approvals()
        self._require_recovery()
        if self._recovery_steps is None or self._recovery_steps.runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        ordered_admissions = tuple(admissions)
        ordered_audit = tuple(audit_events)
        if not ordered_admissions:
            raise ValueError("approval wait requires admissions")
        approval_ids = tuple(item.record.approval_id for item in ordered_admissions)
        if len(set(approval_ids)) != len(approval_ids):
            raise ValueError("approval ids must be unique")
        if not isinstance(occurred_at, datetime) or occurred_at.tzinfo is None:
            raise ValueError("approval wait timestamp must be timezone-aware")
        if (
            continuation.source_step_run_id != recovery_run.run_id
            or continuation.source_step_run_id != recovery_snapshot.run_id
            or recovery_snapshot.state != "interrupted"
            or not recovery_snapshot.messages
            or continuation.batch_id
            != _approval_batch_id(execution_id, continuation.source_step_run_id, approval_ids)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            expected_previous_pending_approval is not None
            and expected_previous_pending_approval.source_step_run_id == continuation.source_step_run_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for admission in ordered_admissions:
            record = admission.record
            operation = admission.operation
            if (
                record.status is not ApprovalStatus.PENDING
                or record.idempotency_key_digest is not None
                or record.decision is not None
                or record.decided_by is not None
                or record.decision_digest is not None
                or record.decided_at is not None
                or record.tenant_id != tenant_id
                or record.execution_id != execution_id
                or operation.tenant_id != tenant_id
                or operation.execution_id != execution_id
                or operation.resource_kind is not ResourceKind.APPROVAL
                or operation.resource_id != record.approval_id
                or operation.operation_kind is not OperationKind.APPROVAL
                or operation.status is not OperationStatus.SUCCEEDED
                or operation.result_ref != record.approval_id
                or operation.result_digest is None
                or operation.error_code is not None
                or not operation.compactable
            ):
                raise ValueError("approval admission identity is invalid")

        prepared = await self._recovery_steps.prepare_snapshots(
            recovery_run,
            (recovery_snapshot,),
        )
        if (
            len(prepared) != 1
            or prepared[0].owner_id != recovery_run.run_id
            or prepared[0].stored.state != "interrupted"
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = _dedupe_stores(
            (
                self._execution.state_store,
                self._recovery.state_store,
                approvals.state_store,
                self._recovery_steps.state_store,
            )
        )
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        approval_events = tuple(
            ExecutionEventAppend(
                ExecutionEventType.APPROVAL_REQUESTED,
                {
                    "approval_id": admission.record.approval_id,
                    "tool_name": admission.tool_name,
                    "args_digest": admission.args_digest,
                    "batch_id": continuation.batch_id,
                },
            )
            for admission in ordered_admissions
        )
        audit_count = len(ordered_audit)
        approval_count = len(ordered_admissions)
        target_execution_revision = expected_execution_revision + audit_count + approval_count
        target_event_sequence = expected_event_sequence + audit_count + approval_count
        target_recovery_revision = expected_recovery_revision + 1

        async def operation() -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            async def mutate(group: StateGroupTransaction) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
                execution_transaction = group.transaction(self._execution.state_store)
                recovery_transaction = group.transaction(self._recovery.state_store)
                approval_transaction = group.transaction(approvals.state_store)
                step_transaction = group.transaction(self._recovery_steps.state_store)
                current_execution = await self._execution.get_in_transaction(
                    execution_transaction,
                    execution_id,
                    tenant_id=tenant_id,
                )
                current_recovery = await self._recovery.get_in_transaction(
                    recovery_transaction,
                    execution_id,
                    tenant_id=tenant_id,
                )
                if not _approval_wait_predecessor_matches(
                    current_execution,
                    current_recovery,
                    expected_execution_revision=expected_execution_revision,
                    expected_event_sequence=expected_event_sequence,
                    expected_recovery_revision=expected_recovery_revision,
                    expected_agent_run_sequence=expected_agent_run_sequence,
                    expected_previous_pending_approval=expected_previous_pending_approval,
                    source_step_run_id=continuation.source_step_run_id,
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if (
                    expected_previous_pending_approval is not None
                    and current_recovery is not None
                    and current_recovery.step_run_id
                    == expected_previous_pending_approval.source_step_run_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._recovery_steps.materialize_snapshot_in_transaction(
                    step_transaction,
                    recovery_run,
                    prepared[0],
                )
                for admission in ordered_admissions:
                    _record, replayed = await approvals.create_with_operation_in_transaction(
                        approval_transaction,
                        admission.record,
                        operation=admission.operation,
                    )
                    if replayed:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                updated_execution = await self._execution.enter_approval_wait_in_transaction(
                    execution_transaction,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_execution_revision,
                    expected_event_sequence=expected_event_sequence,
                    expected_agent_run_sequence=expected_agent_run_sequence,
                    audit_events=ordered_audit,
                    approval_events=approval_events,
                    occurred_at=occurred_at,
                )
                if current_recovery is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                updated_recovery = await self._recovery.compare_and_swap_in_transaction(
                    recovery_transaction,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        current_recovery,
                        state=RecoveryCheckpointState.WAITING,
                        pending_approval=continuation,
                        revision=target_recovery_revision,
                        updated_at=updated_execution.updated_at,
                    ),
                )
                return updated_execution, updated_recovery

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[tuple[ExecutionRecord, RecoveryCheckpoint]]:
            try:
                execution = await self._execution.get(execution_id, tenant_id=tenant_id)
                recovery = await self._recovery.get(execution_id, tenant_id=tenant_id)
                stored_run = await self._recovery_steps.get_run(run_id=continuation.source_step_run_id)
                stored_snapshot = await self._recovery_steps.latest_snapshot(
                    run_id=continuation.source_step_run_id,
                    include_interrupted=True,
                )
                page = await self._events.list(
                    execution_id,
                    tenant_id=tenant_id,
                    after_sequence=expected_event_sequence,
                    limit=audit_count + approval_count + 1,
                )
                events = page.items
                approval_records = tuple(
                    [
                        await approvals.get(admission.record.approval_id, tenant_id=tenant_id)
                        for admission in ordered_admissions
                    ]
                )
                approval_slice = events[audit_count:audit_count + approval_count]
                marker_count = sum(
                    1
                    for index, event in enumerate(approval_slice)
                    if event.sequence == expected_event_sequence + audit_count + index + 1
                    and event.event_type is ExecutionEventType.APPROVAL_REQUESTED
                    and event.payload == approval_events[index].payload
                )
                if marker_count == 0:
                    if _approval_wait_predecessor_matches(
                        execution,
                        recovery,
                        expected_execution_revision=expected_execution_revision,
                        expected_event_sequence=expected_event_sequence,
                        expected_recovery_revision=expected_recovery_revision,
                        expected_agent_run_sequence=expected_agent_run_sequence,
                        expected_previous_pending_approval=expected_previous_pending_approval,
                        source_step_run_id=continuation.source_step_run_id,
                    ):
                        if any(record is not None for record in approval_records):
                            return CommitObservation(
                                DurableCommitState.NOT_COMMITTED,
                                error=AIError(ErrorCode.APPROVAL_CONFLICT),
                            )
                        return CommitObservation(DurableCommitState.NOT_COMMITTED)
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.STORAGE_CONFLICT),
                    )
                if marker_count != approval_count:
                    return _partial_integrity()
                expected_all_events = (*ordered_audit, *approval_events)
                if (
                    len(events) < len(expected_all_events)
                    or any(
                        event.sequence != expected_event_sequence + index
                        or event.event_type is not expected.event_type
                        or event.payload != expected.payload
                        for index, (event, expected) in enumerate(zip(events, expected_all_events), 1)
                    )
                    or stored_run != recovery_run
                    or stored_snapshot != recovery_snapshot
                    or any(
                        record is None
                        or not _approval_create_identity_matches(record, admission.record)
                        or record.status not in {
                            ApprovalStatus.PENDING,
                            ApprovalStatus.APPROVED,
                            ApprovalStatus.DENIED,
                            ApprovalStatus.CANCELLED,
                            ApprovalStatus.EXPIRED,
                        }
                        for record, admission in zip(approval_records, ordered_admissions)
                    )
                    or execution is None
                    or recovery is None
                    or execution.revision < target_execution_revision
                    or execution.event_sequence < target_event_sequence
                    or execution.agent_run_sequence < expected_agent_run_sequence
                    or recovery.revision < target_recovery_revision
                    or recovery.agent_run_sequence < expected_agent_run_sequence
                    or execution.agent_run_sequence != recovery.agent_run_sequence
                ):
                    return _partial_integrity()
                if (
                    execution.revision == target_execution_revision
                    and execution.event_sequence == target_event_sequence
                    and (
                        execution.status is not ExecutionStatus.WAITING_APPROVAL
                        or execution.agent_run_sequence != expected_agent_run_sequence
                    )
                ):
                    return _partial_integrity()
                if recovery.revision == target_recovery_revision and not _approval_wait_target_recovery_matches(
                    recovery,
                    continuation=continuation,
                    expected_agent_run_sequence=expected_agent_run_sequence,
                ):
                    return _partial_integrity()
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=(execution, recovery),
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return _partial_integrity(error)
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(operation, readback, background_tasks=owner_tasks)
        return _require_durable_pair(outcome)

    async def commit_approval_policy_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_recovery_revision: int,
        expected_pending_approval: PendingApprovalContinuation,
        batch_approval_ids: Sequence[str],
        denied_approval_ids: Sequence[str],
        decided_at: datetime,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> tuple[ApprovalRecord, ...]:
        approvals = self._require_approvals()
        self._require_recovery()
        batch = tuple(batch_approval_ids)
        denied = tuple(denied_approval_ids)
        if (
            not batch
            or len(set(batch)) != len(batch)
            or not denied
            or len(set(denied)) != len(denied)
            or tuple(value for value in batch if value in set(denied)) != denied
            or expected_pending_approval.batch_id
            != _approval_batch_id(execution_id, expected_pending_approval.source_step_run_id, batch)
            or not isinstance(decided_at, datetime)
            or decided_at.tzinfo is None
        ):
            raise ValueError("approval policy checkpoint input is invalid")
        stores = _dedupe_stores((self._recovery.state_store, approvals.state_store))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        target_recovery_revision = expected_recovery_revision + 1

        async def operation() -> tuple[ApprovalRecord, ...]:
            async def mutate(group: StateGroupTransaction) -> tuple[ApprovalRecord, ...]:
                recovery_transaction = group.transaction(self._recovery.state_store)
                approval_transaction = group.transaction(approvals.state_store)
                recovery = await self._recovery.get_in_transaction(
                    recovery_transaction,
                    execution_id,
                    tenant_id=tenant_id,
                )
                if not _approval_waiting_recovery_predecessor(
                    recovery,
                    expected_revision=expected_recovery_revision,
                    expected_pending_approval=expected_pending_approval,
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                records: list[ApprovalRecord] = []
                for approval_id in batch:
                    record = await approvals.get_approval_in_transaction(
                        approval_transaction,
                        approval_id,
                        tenant_id=tenant_id,
                    )
                    if record is None or record.execution_id != execution_id or record.tenant_id != tenant_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    records.append(record)
                denied_set = set(denied)
                if any(record.status is not ApprovalStatus.PENDING for record in records if record.approval_id in denied_set):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                updated = await approvals.cancel_pending_in_transaction(
                    approval_transaction,
                    denied,
                    execution_id=execution_id,
                    tenant_id=tenant_id,
                    decided_at=decided_at,
                )
                if recovery is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._recovery.compare_and_swap_in_transaction(
                    recovery_transaction,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        recovery,
                        revision=target_recovery_revision,
                        updated_at=decided_at,
                    ),
                )
                return updated

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[tuple[ApprovalRecord, ...]]:
            try:
                records = tuple([await approvals.get(value, tenant_id=tenant_id) for value in batch])
                recovery = await self._recovery.get(execution_id, tenant_id=tenant_id)
                if any(record is None or record.tenant_id != tenant_id or record.execution_id != execution_id for record in records):
                    return _partial_integrity()
                by_id = {record.approval_id: record for record in records if record is not None}
                denied_records = tuple(by_id[value] for value in denied)
                cancelled_count = sum(record.status is ApprovalStatus.CANCELLED for record in denied_records)
                if any(record.status in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED, ApprovalStatus.EXPIRED} for record in denied_records):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.STORAGE_CONFLICT))
                if 0 < cancelled_count < len(denied):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.STORAGE_CONFLICT))
                if recovery is None or recovery.revision < expected_recovery_revision:
                    return _partial_integrity()
                if cancelled_count == len(denied):
                    if recovery.revision < target_recovery_revision:
                        return _partial_integrity()
                    if recovery.revision == target_recovery_revision and not _approval_waiting_recovery_predecessor(
                        recovery,
                        expected_revision=target_recovery_revision,
                        expected_pending_approval=expected_pending_approval,
                    ):
                        return _partial_integrity()
                    return CommitObservation(DurableCommitState.COMMITTED, value=tuple(cast(ApprovalRecord, record) for record in records))
                if _approval_waiting_recovery_predecessor(
                    recovery,
                    expected_revision=expected_recovery_revision,
                    expected_pending_approval=expected_pending_approval,
                ):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.STORAGE_CONFLICT))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return _partial_integrity(error)
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(operation, readback, background_tasks=owner_tasks)
        return _require_durable_tuple(outcome)

    async def claim_approval_resume_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_execution_revision: int,
        expected_event_sequence: int,
        expected_recovery_revision: int,
        expected_agent_run_sequence: int,
        expected_pending_approval: PendingApprovalContinuation,
        approval_ids: Sequence[str],
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        approvals = self._require_approvals()
        self._require_recovery()
        ordered = tuple(approval_ids)
        if (
            not ordered
            or len(set(ordered)) != len(ordered)
            or expected_pending_approval.batch_id
            != _approval_batch_id(execution_id, expected_pending_approval.source_step_run_id, ordered)
        ):
            raise ValueError("approval resume checkpoint input is invalid")
        stores = _dedupe_stores((self._execution.state_store, self._recovery.state_store, approvals.state_store))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        next_sequence = expected_agent_run_sequence + 1
        target_execution_revision = expected_execution_revision + 1
        target_recovery_revision = expected_recovery_revision + 1
        next_run_id = step_run_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            segment_sequence=next_sequence,
        )

        async def operation() -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            async def mutate(group: StateGroupTransaction) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(self._recovery.state_store)
                approval_tx = group.transaction(approvals.state_store)
                execution = await self._execution.get_in_transaction(execution_tx, execution_id, tenant_id=tenant_id)
                recovery = await self._recovery.get_in_transaction(recovery_tx, execution_id, tenant_id=tenant_id)
                if (
                    execution is None
                    or execution.status is not ExecutionStatus.WAITING_APPROVAL
                    or execution.revision != expected_execution_revision
                    or execution.event_sequence != expected_event_sequence
                    or execution.agent_run_sequence != expected_agent_run_sequence
                    or not _approval_waiting_recovery_predecessor(
                        recovery,
                        expected_revision=expected_recovery_revision,
                        expected_pending_approval=expected_pending_approval,
                        expected_agent_run_sequence=expected_agent_run_sequence,
                    )
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                for approval_id in ordered:
                    record = await approvals.get_approval_in_transaction(approval_tx, approval_id, tenant_id=tenant_id)
                    if record is None or record.execution_id != execution_id or record.tenant_id != tenant_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if record.status is ApprovalStatus.PENDING:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                updated_execution = await self._execution.claim_approval_resume_in_transaction(
                    execution_tx,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_execution_revision,
                    expected_event_sequence=expected_event_sequence,
                    expected_agent_run_sequence=expected_agent_run_sequence,
                )
                if recovery is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                updated_recovery = await self._recovery.compare_and_swap_in_transaction(
                    recovery_tx,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        recovery,
                        state=RecoveryCheckpointState.ACTIVE,
                        step_run_id=next_run_id,
                        agent_run_sequence=next_sequence,
                        revision=target_recovery_revision,
                        updated_at=updated_execution.updated_at,
                    ),
                )
                return updated_execution, updated_recovery

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[tuple[ExecutionRecord, RecoveryCheckpoint]]:
            try:
                execution = await self._execution.get(execution_id, tenant_id=tenant_id)
                recovery = await self._recovery.get(execution_id, tenant_id=tenant_id)
                records = tuple([await approvals.get(value, tenant_id=tenant_id) for value in ordered])
                if any(record is None or record.execution_id != execution_id or record.tenant_id != tenant_id for record in records):
                    return _partial_integrity()
                if execution is None or recovery is None:
                    return _partial_integrity()
                all_terminal = all(cast(ApprovalRecord, record).status is not ApprovalStatus.PENDING for record in records)
                exact_predecessor = (
                    all_terminal
                    and execution.status is ExecutionStatus.WAITING_APPROVAL
                    and execution.revision == expected_execution_revision
                    and execution.event_sequence == expected_event_sequence
                    and execution.agent_run_sequence == expected_agent_run_sequence
                    and _approval_waiting_recovery_predecessor(
                        recovery,
                        expected_revision=expected_recovery_revision,
                        expected_pending_approval=expected_pending_approval,
                        expected_agent_run_sequence=expected_agent_run_sequence,
                    )
                )
                if exact_predecessor:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if execution.agent_run_sequence == expected_agent_run_sequence and recovery.agent_run_sequence == expected_agent_run_sequence:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.STORAGE_CONFLICT))
                if (
                    execution.agent_run_sequence < next_sequence
                    or recovery.agent_run_sequence < next_sequence
                    or execution.agent_run_sequence != recovery.agent_run_sequence
                    or execution.revision < target_execution_revision
                    or execution.event_sequence < expected_event_sequence
                    or recovery.revision < target_recovery_revision
                ):
                    return _partial_integrity()
                if (
                    execution.revision == target_execution_revision
                    and execution.agent_run_sequence == next_sequence
                    and (
                        execution.status is not ExecutionStatus.STARTED
                        or execution.event_sequence != expected_event_sequence
                    )
                ):
                    return _partial_integrity()
                if (
                    recovery.revision == target_recovery_revision
                    and recovery.agent_run_sequence == next_sequence
                    and (
                        recovery.state is not RecoveryCheckpointState.ACTIVE
                        or recovery.step_run_id != next_run_id
                        or recovery.pending_approval != expected_pending_approval
                    )
                ):
                    return _partial_integrity()
                return CommitObservation(DurableCommitState.COMMITTED, value=(execution, recovery))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return _partial_integrity(error)
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(operation, readback, background_tasks=owner_tasks)
        return _require_durable_pair(outcome)

    async def commit_waiting_approval_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        approval_ids: Sequence[str],
        expected_recovery_revision: int,
        expected_agent_run_sequence: int,
        expected_pending_approval: PendingApprovalContinuation,
        audit_events: Sequence[ExecutionEventAppend] = (),
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> ExecutionRecord:
        approvals = self._require_approvals()
        self._require_recovery()
        ordered = tuple(approval_ids)
        ordered_audit = tuple(audit_events)
        if (
            not ordered
            or len(set(ordered)) != len(ordered)
            or expected_agent_run_sequence < 1
            or expected_pending_approval.batch_id
            != _approval_batch_id(commit.execution_id, expected_pending_approval.source_step_run_id, ordered)
        ):
            raise ValueError("waiting approval cancel checkpoint input is invalid")
        stores = _dedupe_stores((self._execution.state_store, self._recovery.state_store, approvals.state_store))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        expected_events = (*ordered_audit, ExecutionEventAppend(ExecutionEventType.CANCEL_REQUESTED, {"operation_id": commit.operation_id}))
        event_count = len(expected_events)
        target_execution_revision = commit.expected_revision + event_count
        target_event_sequence = commit.expected_event_sequence + event_count
        target_recovery_revision = expected_recovery_revision + 1

        async def operation() -> ExecutionRecord:
            async def mutate(group: StateGroupTransaction) -> ExecutionRecord:
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(self._recovery.state_store)
                approval_tx = group.transaction(approvals.state_store)
                execution = await self._execution.get_in_transaction(execution_tx, commit.execution_id, tenant_id=commit.tenant_id)
                recovery = await self._recovery.get_in_transaction(recovery_tx, commit.execution_id, tenant_id=commit.tenant_id)
                if (
                    execution is None
                    or execution.status is not ExecutionStatus.WAITING_APPROVAL
                    or execution.revision != commit.expected_revision
                    or execution.event_sequence != commit.expected_event_sequence
                    or execution.agent_run_sequence != expected_agent_run_sequence
                    or not _approval_waiting_recovery_predecessor(
                        recovery,
                        expected_revision=expected_recovery_revision,
                        expected_pending_approval=expected_pending_approval,
                        expected_agent_run_sequence=expected_agent_run_sequence,
                    )
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                for approval_id in ordered:
                    record = await approvals.get_approval_in_transaction(approval_tx, approval_id, tenant_id=commit.tenant_id)
                    if record is None or record.execution_id != commit.execution_id or record.tenant_id != commit.tenant_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await approvals.cancel_pending_in_transaction(
                    approval_tx,
                    ordered,
                    execution_id=commit.execution_id,
                    tenant_id=commit.tenant_id,
                    decided_at=commit.requested_at,
                )
                updated_execution = await self._execution.request_cancel_in_transaction(
                    execution_tx,
                    commit,
                    expected_status=ExecutionStatus.WAITING_APPROVAL,
                    pending_events=ordered_audit,
                )
                if recovery is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._recovery.compare_and_swap_in_transaction(
                    recovery_tx,
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        recovery,
                        revision=target_recovery_revision,
                        updated_at=commit.requested_at,
                    ),
                )
                return updated_execution

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            try:
                execution = await self._execution.get(commit.execution_id, tenant_id=commit.tenant_id)
                recovery = await self._recovery.get(commit.execution_id, tenant_id=commit.tenant_id)
                records = tuple([await approvals.get(value, tenant_id=commit.tenant_id) for value in ordered])
                page = await self._events.list(
                    commit.execution_id,
                    tenant_id=commit.tenant_id,
                    after_sequence=commit.expected_event_sequence,
                    limit=event_count + 1,
                )
                events = page.items
                if execution is None or recovery is None or any(
                    record is None or record.execution_id != commit.execution_id or record.tenant_id != commit.tenant_id
                    for record in records
                ):
                    return _partial_integrity()
                revision_delta = execution.revision - commit.expected_revision
                sequence_delta = execution.event_sequence - commit.expected_event_sequence
                if (
                    revision_delta < 0
                    or sequence_delta < 0
                    or revision_delta < sequence_delta
                    or any(event.sequence != commit.expected_event_sequence + index for index, event in enumerate(events, 1))
                    or (events and events[-1].sequence > execution.event_sequence)
                    or (sequence_delta > 0 and not events)
                ):
                    return _partial_integrity()
                prefix = events[:event_count]
                prefix_matches = len(prefix) == event_count and all(
                    actual.event_type is expected.event_type and actual.payload == expected.payload
                    for actual, expected in zip(prefix, expected_events)
                )
                if not prefix_matches:
                    exact_predecessor = (
                        execution.status is ExecutionStatus.WAITING_APPROVAL
                        and execution.revision == commit.expected_revision
                        and execution.event_sequence == commit.expected_event_sequence
                        and execution.agent_run_sequence == expected_agent_run_sequence
                        and _approval_waiting_recovery_predecessor(
                            recovery,
                            expected_revision=expected_recovery_revision,
                            expected_pending_approval=expected_pending_approval,
                            expected_agent_run_sequence=expected_agent_run_sequence,
                        )
                    )
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=None if exact_predecessor else AIError(ErrorCode.STORAGE_CONFLICT),
                    )
                if (
                    any(cast(ApprovalRecord, record).status is ApprovalStatus.PENDING for record in records)
                    or execution.revision < target_execution_revision
                    or execution.event_sequence < target_event_sequence
                    or recovery.revision < target_recovery_revision
                    or execution.agent_run_sequence != expected_agent_run_sequence
                    or recovery.agent_run_sequence != expected_agent_run_sequence
                    or execution.agent_run_sequence != recovery.agent_run_sequence
                ):
                    return _partial_integrity()
                if recovery.revision == target_recovery_revision and not _approval_waiting_recovery_predecessor(
                    recovery,
                    expected_revision=target_recovery_revision,
                    expected_pending_approval=expected_pending_approval,
                    expected_agent_run_sequence=expected_agent_run_sequence,
                ):
                    return _partial_integrity()
                if recovery.revision > target_recovery_revision and (
                    recovery.state not in {RecoveryCheckpointState.HANDOFF, RecoveryCheckpointState.COMPLETED}
                    or recovery.pending_approval is not None
                ):
                    return _partial_integrity()
                if execution.status is ExecutionStatus.CANCELLING:
                    if recovery.state is RecoveryCheckpointState.COMPLETED:
                        return _partial_integrity()
                elif execution.status is ExecutionStatus.CANCELLED:
                    if recovery.revision <= target_recovery_revision or recovery.state not in {RecoveryCheckpointState.HANDOFF, RecoveryCheckpointState.COMPLETED}:
                        return _partial_integrity()
                else:
                    return _partial_integrity()
                return CommitObservation(DurableCommitState.COMMITTED, value=execution)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return _partial_integrity(error)
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(operation, readback, background_tasks=owner_tasks)
        return _require_durable_execution(outcome)

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
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if result_payload is None and error_code is None:
            raise ValueError("tool terminal command requires a result or error")
        expected_status = (
            ToolOperationStatus.COMPLETED
            if result_payload is not None
            else ToolOperationStatus.FAILED
        )
        terminal_error_code = error_code or ErrorCode.EXECUTION_FAILED.value
        stores = [tools.state_store]
        cancelled = False

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            transaction = group.transaction(tools.state_store)
            if result_payload is not None:
                return await tools.complete_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    result_payload=result_payload,
                )
            return await tools.fail_in_transaction(
                transaction,
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                error_code=terminal_error_code,
                error_payload=error_payload,
            )

        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await tools.get_operation(
                tool_operation_id,
                tenant_id=tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.status is expected_status:
                if observed.owner != owner or observed.fence != fence:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                if result_payload is not None:
                    if observed.result_payload != result_payload:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED,
                            error=AIError(ErrorCode.TOOL_RESULT_CONFLICT),
                        )
                elif (
                    observed.error_code != terminal_error_code
                    or observed.error_payload != error_payload
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
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
            if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_EFFECT_UNKNOWN),
                )
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        while True:
            result = await run_durable_commit(
                lambda: stores[0].storage_group.mutate(stores, callback),
                readback,
                background_tasks=self._background_tasks,
            )
            cancelled = cancelled or result.cancelled
            if result.state is DurableCommitState.COMMITTED:
                if result.value is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if cancelled:
                    raise asyncio.CancelledError
                return result.value
            if result.state is DurableCommitState.NOT_COMMITTED:
                if (
                    isinstance(result.error, AIError)
                    and result.error.code is ErrorCode.STORAGE_CONFLICT
                ):
                    await asyncio.sleep(0)
                    continue
                if result.error is not None:
                    raise result.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
            if isinstance(result.error, AIError) and result.error.code in {
                ErrorCode.TOOL_OPERATION_CONFLICT,
                ErrorCode.TOOL_RESULT_CONFLICT,
                ErrorCode.TOOL_EFFECT_UNKNOWN,
            }:
                raise result.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def commit_tool_admission(
        self,
        request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
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

        while True:
            try:
                return await attempt()
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                tools = self._tools
                if tools is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
                observed = await tools.get_operation(
                    request.tool_operation_id,
                    tenant_id=request.tenant_id,
                )
                if observed is None:
                    await asyncio.sleep(0)
                    continue
                if not _tool_admission_matches(observed, request):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
                if observed.status is ToolOperationStatus.CANCELLED:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
                if observed.status in {
                    ToolOperationStatus.COMPLETED,
                    ToolOperationStatus.FAILED,
                }:
                    return observed
                # CLAIMED/PENDING must be classified by a fresh repository
                # transaction so lease expiry and owner takeover semantics are
                # never guessed from an out-of-transaction readback.
                await asyncio.sleep(0)

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
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> ExecutionTerminalCommitResult:
        if execution_projections and (
            self._execution_steps is None or execution_run is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is None and (conversation_run is not None or conversation_snapshot is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is None and (recovery_run is not None or recovery_snapshot is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (recovery_run is None) != (recovery_snapshot is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "terminal checkpoint requested: execution=%s session=%s recovery=%s",
            commit.execution.execution_id,
            session_id,
            recovery_checkpoint is not None,
        )
        owner_tasks = (
            self._background_tasks
            if background_tasks is None
            else background_tasks
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
                background_tasks=owner_tasks,
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
        has_recovery_step_target = recovery_run is not None or recovery_snapshot is not None
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
            if has_recovery_step_target:
                if self._recovery_steps is None or recovery_run is None or recovery_snapshot is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                recovery_steps_merged = (
                    self._recovery_steps.state_store.storage_group
                    is self._execution.state_store.storage_group
                )
            execution_stores.append(self._recovery.state_store)
            if recovery_steps_merged:
                execution_stores.append(self._recovery_steps.state_store)
        if recovery_pending and has_recovery_step_target and not recovery_steps_merged:
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
                background_tasks=owner_tasks,
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



def _dedupe_stores(stores: Sequence[StateStore]) -> tuple[StateStore, ...]:
    result: list[StateStore] = []
    seen: set[int] = set()
    for store in stores:
        identity = id(store)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(store)
    return tuple(result)


def _approval_batch_id(
    execution_id: str,
    source_step_run_id: str,
    approval_ids: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "contract": "tool-approval-batch-v1",
            "execution_id": execution_id,
            "source_step_run_id": source_step_run_id,
            "approval_ids": sorted(approval_ids),
        }
    )


def _approval_create_identity_matches(
    current: ApprovalRecord,
    expected: ApprovalRecord,
) -> bool:
    return (
        current.approval_id == expected.approval_id
        and current.execution_id == expected.execution_id
        and current.tenant_id == expected.tenant_id
        and current.operation_id == expected.operation_id
    )


def _approval_wait_predecessor_matches(
    execution: ExecutionRecord | None,
    recovery: RecoveryCheckpoint | None,
    *,
    expected_execution_revision: int,
    expected_event_sequence: int,
    expected_recovery_revision: int,
    expected_agent_run_sequence: int,
    expected_previous_pending_approval: PendingApprovalContinuation | None,
    source_step_run_id: str,
) -> bool:
    return (
        execution is not None
        and execution.status is ExecutionStatus.STARTED
        and execution.revision == expected_execution_revision
        and execution.event_sequence == expected_event_sequence
        and execution.agent_run_sequence == expected_agent_run_sequence
        and recovery is not None
        and recovery.state is RecoveryCheckpointState.ACTIVE
        and recovery.revision == expected_recovery_revision
        and recovery.agent_run_sequence == expected_agent_run_sequence
        and recovery.step_run_id == source_step_run_id
        and recovery.pending_approval == expected_previous_pending_approval
        and recovery.pending_operation_id is None
        and recovery.handoff_phase is RecoveryHandoffPhase.NONE
        and recovery.terminal_handoff is None
    )


def _approval_wait_target_recovery_matches(
    recovery: RecoveryCheckpoint,
    *,
    continuation: PendingApprovalContinuation,
    expected_agent_run_sequence: int,
) -> bool:
    return (
        recovery.state is RecoveryCheckpointState.WAITING
        and recovery.step_run_id == continuation.source_step_run_id
        and recovery.agent_run_sequence == expected_agent_run_sequence
        and recovery.pending_approval == continuation
        and recovery.pending_operation_id is None
        and recovery.handoff_phase is RecoveryHandoffPhase.NONE
        and recovery.terminal_handoff is None
    )


def _approval_waiting_recovery_predecessor(
    recovery: RecoveryCheckpoint | None,
    *,
    expected_revision: int,
    expected_pending_approval: PendingApprovalContinuation,
    expected_agent_run_sequence: int | None = None,
) -> bool:
    return (
        recovery is not None
        and recovery.state is RecoveryCheckpointState.WAITING
        and recovery.revision == expected_revision
        and (
            expected_agent_run_sequence is None
            or recovery.agent_run_sequence == expected_agent_run_sequence
        )
        and recovery.pending_approval == expected_pending_approval
        and recovery.step_run_id == expected_pending_approval.source_step_run_id
        and recovery.pending_operation_id is None
        and recovery.handoff_phase is RecoveryHandoffPhase.NONE
        and recovery.terminal_handoff is None
    )


def _partial_integrity(
    error: AIError | None = None,
) -> CommitObservation[object]:
    return CommitObservation(
        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
        error=error or AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
    )


def _require_durable_pair(
    outcome: DurableCommitResult[tuple[ExecutionRecord, RecoveryCheckpoint]],
) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
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


def _require_durable_tuple(
    outcome: DurableCommitResult[tuple[ApprovalRecord, ...]],
) -> tuple[ApprovalRecord, ...]:
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


def _require_durable_execution(
    outcome: DurableCommitResult[ExecutionRecord],
) -> ExecutionRecord:
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


def _same_group(stores: Sequence[StateStore]) -> bool:
    return bool(stores) and all(store.storage_group is stores[0].storage_group for store in stores[1:])




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
