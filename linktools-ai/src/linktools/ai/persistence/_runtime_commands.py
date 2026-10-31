#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named state commands for multi-record Runtime checkpoints."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from linktools.core import environ
from ..model import StoredUserInput

from ..core import (
    ApprovalStatus,
    PersistenceDomain,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    OperationLedgerInput,
    SessionStatus,
    ToolOperationStatus,
    canonical_sha256,
    agent_run_id,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ._archive import ContentArchive, PreparedArchive
from .contracts import (
    deferred_resource_id,
    AgentAttemptClaim,
    ApprovalRecord,
    ApprovalRepository,
    ConversationCursor,
    ConversationHistoryRecord,
    ConversationHistoryRepository,
    EventRepository,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionHistoryState,
    ExecutionRecord,
    ResultRecord,
    ExecutionRepository,
    ExecutionStartClaim,
    ExecutionStartReservation,
    RuntimePayloadRef,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalCallRecord,
    ExternalCallRepository,
    PendingToolContinuation,
    OperationLedgerRepository,
    RecoveryCheckpoint,
    RecoveryCheckpointRepository,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    SessionRecord,
    SessionRepository,
    ToolOperationAdmission,
    ToolOperationRecord,
    decode_tool_operation_failure,
)
from .durability import (
    CommitObservation,
    DurableCommitResult,
    DurableCommitState,
    run_durable_commit,
)
from ._execution_commands import _execution_history_seal
from ._execution_repositories import ExecutionRepositoryImpl
from ._model_interaction_store import ModelInteractionStateStepArchive
from ._conversation_repositories import (
    ConversationHistoryRepositoryImpl,
    SessionRepositoryImpl,
)
from ._recovery_repositories import (
    RecoveryCheckpointRepositoryImpl,
    tool_admission_matches as _tool_admission_matches,
)
from ._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
    StepEvent,
)
from ._step_archive import (
    PreparedExecutionProjection,
    PreparedAgentRunCheckpoint,
    PreparedAgentRunCheckpointBatch,
    StateStepArchive,
)
from ._repository_common import operation_identity_matches, validate_recovery_operation
from .state import GroupRecordReader, StateGroupTransaction, StateStore, StateTransaction

_logger = environ.get_logger("ai.runtime.state.commands")


def _same_execution_except_hold_metadata(
    current: ExecutionRecord, expected: ExecutionRecord,
) -> bool:
    """Holds share the revision counter without changing execution semantics."""
    return (
        current.revision >= expected.revision
        and replace(
            current,
            revision=expected.revision,
            updated_at=expected.updated_at,
            dependency_hold_ids=expected.dependency_hold_ids,
        ) == expected
    )


def _copy_archive_union(values: Sequence[PreparedArchive]) -> PreparedArchive:
    """Combine values without consuming a caller's preparation receipts."""
    unique = {id(value): value for value in values}
    return PreparedArchive.combine(tuple(
        PreparedArchive(owner=value.owner, records=dict(value.records), nodes=dict(value.nodes))
        for value in unique.values()
    ))


def _timeline_turn_message_range(
    history: ConversationHistoryRecord,
    prepared: PreparedAgentRunCheckpointBatch,
    expected_cursor: ConversationCursor | None,
) -> tuple[int, int]:
    if not prepared.checkpoints:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    local_start = min(
        (chunk.first_message_index for chunk in prepared.checkpoints[0].chunks),
        default=None,
    )
    end = history.inherited_message_count + prepared.target_transcript_message_count
    if local_start is not None:
        start = history.inherited_message_count + local_start
    elif expected_cursor is None:
        if history.inherited_message_count != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        start = 0
    elif expected_cursor.message_count is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    else:
        start = expected_cursor.message_count
    if end <= start:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return start, end


if TYPE_CHECKING:
    from .capabilities import RuntimeCapabilityWrites, PreparedCapabilityWrites


class StateCommands:
    """Named semantic commands that may enlist several Runtime domains."""

    def __init__(
        self,
        execution: ExecutionRepository,
        *,
        namespace: str,
        events: EventRepository,
        operations: OperationLedgerRepository | None = None,
        approvals: ApprovalRepository | None = None,
        external_calls: ExternalCallRepository | None = None,
        conversation: SessionRepository | None = None,
        recovery: RecoveryCheckpointRepository | None = None,
        conversation_history: ConversationHistoryRepository | None = None,
        tools: object | None = None,
        conversation_run_store: StateStepArchive | None = None,
        execution_run_store: StateStepArchive | None = None,
        recovery_run_store: StateStepArchive | None = None,
        background_tasks: "set[asyncio.Task[object]]",
        observe_cancellation: bool = True,
        capability_writes: "RuntimeCapabilityWrites | None" = None,
    ) -> None:
        self._execution = execution
        self._namespace = namespace
        self._tenant_id = execution.tenant_id
        self._events = events
        self._operations = operations
        self._approvals = approvals
        self._external_calls = external_calls
        self._conversation = conversation
        self._recovery = recovery
        self._conversation_history = conversation_history
        self._tools = tools
        self._conversation_run_store = conversation_run_store
        self._execution_run_store = execution_run_store
        self._recovery_run_store = recovery_run_store
        self._background_tasks = background_tasks
        self._observe_cancellation = observe_cancellation
        self._capability_writes = capability_writes

    async def _load_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._execution.state_store.read(
            lambda _transaction: self._execution.get(execution_id, tenant_id=tenant_id)
        )

    async def _load_session(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._conversation
        if repository is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )

    def _require_approvals(self) -> ApprovalRepository:
        if self._approvals is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._approvals

    def _require_external_calls(self) -> ExternalCallRepository:
        if self._external_calls is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._external_calls

    async def commit_deferred_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_execution_revision: int,
        expected_event_seq: int,
        expected_recovery_revision: int,
        expected_agent_run_seq: int,
        continuation: PendingToolContinuation,
        audit_events: Sequence[ExecutionEventAppend] = (),
        approval_records: Sequence[ApprovalRecord] = (),
        external_records: Sequence[ExternalCallRecord] = (),
        occurred_at: datetime,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
        producer_generation: int | None = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        approvals = self._require_approvals()
        external_calls = self._require_external_calls()
        if occurred_at.tzinfo is None:
            raise ValueError("deferred checkpoint timestamp must be timezone-aware")
        if continuation.source_agent_run_id == "":
            raise ValueError("deferred checkpoint source step is required")
        approval_values = tuple(approval_records)
        external_values = tuple(external_records)
        request_events = tuple(
            ExecutionEventAppend(
                ExecutionEventType.APPROVAL_REQUESTED,
                {
                    "tool_call_id": item.tool_call_id,
                    "tool_name": item.tool_name,
                    "arguments_digest": item.arguments_payload.digest,
                },
            )
            for item in continuation.approvals
        ) + tuple(
            ExecutionEventAppend(
                ExecutionEventType.EXTERNAL_REQUESTED,
                {
                    "tool_call_id": item.tool_call_id,
                    "tool_name": item.tool_name,
                    "arguments_digest": item.arguments_payload.digest,
                },
            )
            for item in continuation.calls
        )
        if not request_events:
            raise ValueError("deferred checkpoint requires requests")
        if len(approval_values) != len(continuation.approvals):
            raise ValueError("approval records do not match continuation")
        if len(external_values) != len(continuation.calls):
            raise ValueError("external records do not match continuation")
        if any(
            record.execution_id != execution_id
            or self._tenant_id != tenant_id
            or record.status is not ApprovalStatus.PENDING
            for record in approval_values
        ):
            raise ValueError("approval record identity is invalid")
        if any(
            record.execution_id != execution_id
            or self._tenant_id != tenant_id
            or record.status is not ExternalCallStatus.PENDING
            for record in external_values
        ):
            raise ValueError("external record identity is invalid")
        if tuple(record.approval_id for record in approval_values) != tuple(
            deferred_resource_id(
                "approval-v1",
                tenant_id,
                execution_id,
                continuation.source_agent_run_id,
                item.tool_call_id,
            )
            for item in continuation.approvals
        ):
            raise ValueError("approval record ids do not match continuation")
        if tuple(record.call_id for record in external_values) != tuple(
            deferred_resource_id(
                "external-call-v1",
                tenant_id,
                execution_id,
                continuation.source_agent_run_id,
                item.tool_call_id,
            )
            for item in continuation.calls
        ):
            raise ValueError("external record ids do not match continuation")
        stores = _dedupe_stores(
            (
                self._execution.state_store,
                self._recovery.state_store,
                approvals.state_store,
                external_calls.state_store,
            )
        )
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def operation() -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            async def mutate(
                group: StateGroupTransaction,
            ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(self._recovery.state_store)
                if producer_generation is not None:
                    await self._execution.require_open_history_head_in_transaction(
                        execution_tx, execution_id,
                        expected_producer_generation=producer_generation, allow_cancelling=False,
                    )
                current_execution = await self._execution.get_in_transaction(
                    execution_tx,
                    execution_id,
                    tenant_id=tenant_id,
                )
                current_checkpoint = await self._recovery.get_in_transaction(
                    recovery_tx,
                    execution_id,
                    tenant_id=tenant_id,
                )
                if (
                    current_execution is None
                    or current_checkpoint is None
                    or current_execution.status is not ExecutionStatus.STARTED
                    or current_execution.revision != expected_execution_revision
                    or current_execution.event_seq != expected_event_seq
                    or current_execution.agent_run_seq
                    != expected_agent_run_seq
                    or current_checkpoint.revision != expected_recovery_revision
                    or current_checkpoint.state is not RecoveryCheckpointState.ACTIVE
                    or current_checkpoint.agent_run_id
                    != continuation.source_agent_run_id
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                for record in approval_values:
                    await approvals.create_in_transaction(recovery_tx, record)
                for record in external_values:
                    await external_calls.create_in_transaction(recovery_tx, record)
                updated_execution = (
                    await self._execution.enter_deferred_wait_in_transaction(
                        execution_tx,
                        execution_id,
                        tenant_id=tenant_id,
                        expected_revision=expected_execution_revision,
                        expected_event_seq=expected_event_seq,
                        expected_agent_run_seq=expected_agent_run_seq,
                        audit_events=audit_events,
                        deferred_events=request_events,
                        occurred_at=occurred_at,
                    )
                )
                updated_checkpoint = await self._recovery.compare_and_swap_in_transaction(
                    recovery_tx,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        current_checkpoint,
                        state=RecoveryCheckpointState.WAITING,
                        pending_tools=continuation,
                        revision=expected_recovery_revision + 1,
                        updated_at=occurred_at,
                    ),
                )
                return updated_execution, updated_checkpoint

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[tuple[ExecutionRecord, RecoveryCheckpoint]]:
            execution = await self._load_execution(execution_id, tenant_id=tenant_id)
            checkpoint = await self._recovery.get(execution_id, tenant_id=tenant_id)
            if execution is None or checkpoint is None:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            if producer_generation is not None:
                head = await self._execution.get_history_head(execution_id, tenant_id=tenant_id)
                if head is not None and head.producer_generation > producer_generation:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.STORAGE_CONFLICT),
                    )
                if execution.status is ExecutionStatus.CANCELLING:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.EXECUTION_CANCELLED),
                    )
            for expected in approval_values:
                actual = await approvals.get(
                    expected.approval_id,
                    tenant_id=tenant_id,
                )
                if actual != expected:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
            for expected in external_values:
                actual = await external_calls.get(
                    expected.call_id,
                    tenant_id=tenant_id,
                )
                if actual != expected:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
            if (
                execution.status is ExecutionStatus.WAITING_DEFERRED
                and execution.agent_run_seq == expected_agent_run_seq
                and checkpoint.state is RecoveryCheckpointState.WAITING
                and checkpoint.pending_tools == continuation
            ):
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=(execution, checkpoint),
                )
            if (
                execution.status is ExecutionStatus.STARTED
                and execution.revision == expected_execution_revision
                and execution.event_seq == expected_event_seq
                and checkpoint.revision == expected_recovery_revision
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        return _require_durable_pair(outcome)

    async def commit_deferred_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_recovery_revision: int,
        expected_pending_tools: PendingToolContinuation,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> ExecutionRecord:
        approvals = self._require_approvals()
        external_calls = self._require_external_calls()
        stores = _dedupe_stores(
            (
                self._execution.state_store,
                self._recovery.state_store,
                approvals.state_store,
                external_calls.state_store,
            )
        )
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        approval_ids = tuple(
            deferred_resource_id(
                "approval-v1",
                self._tenant_id,
                commit.execution_id,
                expected_pending_tools.source_agent_run_id,
                item.tool_call_id,
            )
            for item in expected_pending_tools.approvals
        )
        call_ids = tuple(
            deferred_resource_id(
                "external-call-v1",
                self._tenant_id,
                commit.execution_id,
                expected_pending_tools.source_agent_run_id,
                item.tool_call_id,
            )
            for item in expected_pending_tools.calls
        )

        async def operation() -> ExecutionRecord:
            async def mutate(group: StateGroupTransaction) -> ExecutionRecord:
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(self._recovery.state_store)
                execution = await self._execution.get_in_transaction(
                    execution_tx,
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                )
                checkpoint = await self._recovery.get_in_transaction(
                    recovery_tx,
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                )
                if (
                    execution is None
                    or checkpoint is None
                    or execution.status is not ExecutionStatus.WAITING_DEFERRED
                    or execution.revision != commit.expected_revision
                    or execution.event_seq != commit.expected_event_seq
                    or checkpoint.revision != expected_recovery_revision
                    or checkpoint.state is not RecoveryCheckpointState.WAITING
                    or checkpoint.pending_tools != expected_pending_tools
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await approvals.cancel_pending_in_transaction(
                    recovery_tx,
                    approval_ids,
                    execution_id=commit.execution_id,
                    tenant_id=self._tenant_id,
                    decided_at=commit.requested_at,
                )
                await external_calls.cancel_pending_in_transaction(
                    recovery_tx,
                    call_ids,
                    execution_id=commit.execution_id,
                    tenant_id=self._tenant_id,
                    cancelled_at=commit.requested_at,
                )
                updated = await self._execution.request_cancel_in_transaction(
                    execution_tx,
                    commit,
                    expected_status=ExecutionStatus.WAITING_DEFERRED,
                )
                await self._recovery.compare_and_swap_in_transaction(
                    recovery_tx,
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        checkpoint,
                        state=RecoveryCheckpointState.ACTIVE,
                        pending_tools=None,
                        revision=expected_recovery_revision + 1,
                        updated_at=commit.requested_at,
                    ),
                )
                return updated

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            execution = await self._load_execution(
                commit.execution_id,
                tenant_id=self._tenant_id,
            )
            checkpoint = await self._recovery.get(
                commit.execution_id,
                tenant_id=self._tenant_id,
            )
            if execution is None or checkpoint is None:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            for approval_id in approval_ids:
                record = await approvals.get(
                    approval_id,
                    tenant_id=self._tenant_id,
                )
                if (
                    record is None
                    or record.execution_id != commit.execution_id
                    or record.status is ApprovalStatus.PENDING
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
            for call_id in call_ids:
                record = await external_calls.get(
                    call_id,
                    tenant_id=self._tenant_id,
                )
                if (
                    record is None
                    or record.execution_id != commit.execution_id
                    or record.status is ExternalCallStatus.PENDING
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
            if (
                execution.status
                in {
                    ExecutionStatus.CANCELLING,
                    ExecutionStatus.FINALIZING,
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }
                and execution.event_seq
                >= commit.expected_event_seq + 1
                and checkpoint.state
                in {
                    RecoveryCheckpointState.ACTIVE,
                    RecoveryCheckpointState.COMPLETED,
                }
                and checkpoint.pending_tools is None
                and checkpoint.handoff_phase is RecoveryHandoffPhase.NONE
                and checkpoint.terminal_handoff is None
                and checkpoint.revision >= expected_recovery_revision + 1
            ):
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=execution,
                )
            if (
                execution.status is ExecutionStatus.WAITING_DEFERRED
                and execution.revision == commit.expected_revision
                and execution.event_seq == commit.expected_event_seq
                and checkpoint.revision == expected_recovery_revision
                and checkpoint.pending_tools == expected_pending_tools
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        return _require_durable_record(outcome)

    async def claim_deferred_resume_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_execution_revision: int,
        expected_event_seq: int,
        expected_recovery_revision: int,
        expected_agent_run_seq: int,
        expected_pending_tools: PendingToolContinuation,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        stores = _dedupe_stores(
            (self._execution.state_store, self._recovery.state_store)
        )
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        next_sequence = expected_agent_run_seq + 1
        producer_claim_id = uuid4().hex
        producer_generation = expected_execution_revision + 1
        next_agent_run_id = agent_run_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            agent_run_seq=next_sequence,
        )

        async def operation() -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            async def mutate(
                group: StateGroupTransaction,
            ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(self._recovery.state_store)
                execution = await self._execution.get_in_transaction(
                    execution_tx,
                    execution_id,
                    tenant_id=tenant_id,
                )
                checkpoint = await self._recovery.get_in_transaction(
                    recovery_tx,
                    execution_id,
                    tenant_id=tenant_id,
                )
                if (
                    execution is None
                    or checkpoint is None
                    or execution.status is not ExecutionStatus.WAITING_DEFERRED
                    or execution.revision != expected_execution_revision
                    or execution.event_seq != expected_event_seq
                    or execution.agent_run_seq != expected_agent_run_seq
                    or checkpoint.revision != expected_recovery_revision
                    or checkpoint.state is not RecoveryCheckpointState.WAITING
                    or checkpoint.pending_tools != expected_pending_tools
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                updated_execution = (
                    await self._execution.claim_deferred_resume_in_transaction(
                        execution_tx,
                        execution_id,
                        tenant_id=tenant_id,
                        expected_revision=expected_execution_revision,
                        expected_event_seq=expected_event_seq,
                        expected_agent_run_seq=expected_agent_run_seq,
                    )
                )
                await self._execution.admit_history_producer_in_transaction(
                    execution_tx, updated_execution,
                    producer_claim_id=producer_claim_id,
                )
                updated_checkpoint = await self._recovery.compare_and_swap_in_transaction(
                    recovery_tx,
                    execution_id,
                    tenant_id=tenant_id,
                    expected_revision=expected_recovery_revision,
                    next_record=replace(
                        checkpoint,
                        state=RecoveryCheckpointState.ACTIVE,
                        agent_run_id=next_agent_run_id,
                        pending_tools=None,
                        revision=expected_recovery_revision + 1,
                        updated_at=updated_execution.updated_at,
                    ),
                )
                return updated_execution, updated_checkpoint

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[tuple[ExecutionRecord, RecoveryCheckpoint]]:
            execution = await self._load_execution(execution_id, tenant_id=tenant_id)
            checkpoint = await self._recovery.get(execution_id, tenant_id=tenant_id)
            if execution is None or checkpoint is None:
                return _partial_integrity()
            head = await self._execution.get_history_head(execution_id, tenant_id=tenant_id)
            own_claim = (
                head is not None
                and head.producer_generation == producer_generation
                and head.producer_claim_id == producer_claim_id
            )
            if (
                own_claim
                and execution.status is ExecutionStatus.STARTED
                and execution.agent_run_seq == next_sequence
                and checkpoint.state is RecoveryCheckpointState.ACTIVE
                and checkpoint.agent_run_id == next_agent_run_id
                and checkpoint.pending_tools is None
            ):
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=(execution, checkpoint),
                )
            if (
                execution.status is ExecutionStatus.WAITING_DEFERRED
                and execution.revision == expected_execution_revision
                and checkpoint.revision == expected_recovery_revision
                and checkpoint.pending_tools == expected_pending_tools
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if not own_claim:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.STORAGE_CONFLICT, retryable=False),
                )
            return _partial_integrity()

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        return _require_durable_pair(outcome)

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
        prepared: PreparedAgentRunCheckpoint,
    ) -> ConversationHistoryRecord:
        if self._conversation_history is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        history_id = session.history_id or prepared.owner_id
        history = await self._conversation_history.get_in_transaction(
            transaction,
            history_id,
            tenant_id=self._tenant_id,
        )
        if history is None:
            history = ConversationHistoryRecord(
                history_id=history_id,
                session_id=session.session_id,
                parent_history_id=None,
                prefix_index_head_id=None,
                inherited_message_count=0,
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
        producer_generation: int | None = None,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> ExecutionRecord:
        expected_events = tuple(audit_events) + (
            ExecutionEventAppend(
                ExecutionEventType.CANCEL_REQUESTED,
                {"operation_id": commit.operation_id},
            ),
        )
        target_revision = commit.expected_revision + len(expected_events)
        target_sequence = commit.expected_event_seq + len(expected_events)

        async def operation() -> ExecutionRecord:
            if producer_generation is None:
                return await self._execution.request_cancel(commit, pending_events=audit_events)

            async def mutate(transaction: StateTransaction) -> ExecutionRecord:
                await self._execution.require_open_history_head_in_transaction(
                    transaction, commit.execution_id,
                    expected_producer_generation=producer_generation,
                )
                return await self._execution.request_cancel_in_transaction(
                    transaction, commit, expected_status=expected_status,
                    pending_events=audit_events,
                )

            return await self._execution.state_store.mutate(mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            try:
                execution = await self._load_execution(
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                )
                if execution is None:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                page = await self._events.list(
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                    after_event_seq=commit.expected_event_seq,
                    limit=len(expected_events) + 1,
                )
                items = page.items
                if any(
                    event.event_seq != commit.expected_event_seq + index
                    for index, event in enumerate(items, 1)
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                revision_delta = execution.revision - commit.expected_revision
                sequence_delta = execution.event_seq - commit.expected_event_seq
                if (
                    revision_delta < 0
                    or sequence_delta < 0
                    or revision_delta < sequence_delta
                    or items and items[-1].event_seq > execution.event_seq
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                prefix = items[:len(expected_events)]
                prefix_matches = (
                    len(prefix) == len(expected_events)
                    and all(
                        actual.event_type == expected.event_type
                        and actual.payload == expected.payload
                        for actual, expected in zip(prefix, expected_events)
                    )
                )
                if prefix_matches:
                    if (
                        execution.revision < target_revision
                        or execution.event_seq < target_sequence
                    ):
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
                        tenant_id=self._tenant_id,
                    )
                    if session.active_execution_id not in {None, claim.execution_id}:
                        owner = await self._execution.get_in_transaction(
                            transaction,
                            session.active_execution_id,
                            tenant_id=self._tenant_id,
                        )
                        if (
                            owner is None
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
                            tenant_id=self._tenant_id,
                            execution_id=owner.execution_id,
                            history_unavailable=owner.invalidates_session_history,
                            require_turn_commit=owner.conversation,
                        )
                    await self._conversation.admit_execution_in_transaction(
                        session_transaction,
                        session_id,
                        tenant_id=self._tenant_id,
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
                        tenant_id=self._tenant_id,
                    )
                    if actual == recovery_checkpoint:
                        break
        if session_id is not None:
            for attempt in range(2):
                try:
                    await self._admit_split_start(
                        session_id,
                        tenant_id=self._tenant_id,
                        execution_id=claim.execution_id,
                        expected=expected_cursor,
                    )
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    session = await self._load_session(
                        session_id,
                        tenant_id=self._tenant_id,
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
        *,
        recovery_operation: OperationLedgerInput | None = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        return await self._commit_agent_attempt_checkpoint(claim, recovery_operation=recovery_operation)

    async def commit_prepared_agent_attempt_checkpoint(
        self, claim: AgentAttemptClaim, input_context: RuntimePayloadRef, *,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        return await self._commit_agent_attempt_checkpoint(claim, prepared_input_context=input_context, context_write=context_write)

    async def _commit_agent_attempt_checkpoint(
        self, claim: AgentAttemptClaim, *,
        recovery_operation: OperationLedgerInput | None = None,
        prepared_input_context: RuntimePayloadRef | None = None,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
        producer_claim_id: str | None = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        """Atomically publish the execution sequence and active recovery run."""
        self._require_recovery()
        stores = [self._execution.state_store, self._recovery.state_store]
        operations = self._operations
        if recovery_operation is not None:
            if operations is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores.append(operations.state_store)
        stores = list(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        next_sequence = claim.expected_agent_run_seq + 1
        producer_claim_id = producer_claim_id or (
            recovery_operation.result_ref if recovery_operation is not None else uuid4().hex
        )
        producer_generation = claim.expected_execution_revision + 1
        next_agent_run_id = agent_run_id(
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            execution_id=claim.execution_id,
            agent_run_seq=next_sequence,
        )

        async def callback(
            group: StateGroupTransaction,
        ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            execution_transaction = group.transaction(self._execution.state_store)
            recovery_transaction = group.transaction(self._recovery.state_store)
            existing_operation = None
            if recovery_operation is not None:
                if operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                existing_operation = await operations.get_in_transaction(
                    group.transaction(operations.state_store),
                    recovery_operation.operation_id, tenant_id=self._tenant_id,
                )
                if existing_operation is not None:
                    if (
                        existing_operation.execution_id != claim.execution_id
                        or not operation_identity_matches(existing_operation, recovery_operation)
                    ):
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                else:
                    validate_recovery_operation(
                        recovery_operation, execution_id=claim.execution_id,
                        tenant_id=self._tenant_id, producer=True,
                    )
            current_execution = await self._execution.get_in_transaction(
                execution_transaction,
                claim.execution_id,
                tenant_id=self._tenant_id,
            )
            if current_execution is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current_recovery = await self._recovery.get_in_transaction(
                recovery_transaction,
                claim.execution_id,
                tenant_id=self._tenant_id,
            )
            if current_recovery is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if existing_operation is not None:
                return current_execution, current_recovery
            if (
                current_execution.revision != claim.expected_execution_revision
                or current_execution.agent_run_seq
                != claim.expected_agent_run_seq
                or current_recovery.revision != claim.expected_recovery_revision
                or current_recovery.state is not claim.expected_recovery_state
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current_recovery.state is not RecoveryCheckpointState.ADMITTED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if context_write is not None:
                content_archive, archive_records = context_write
                await content_archive.store.mutate(lambda tx: content_archive.publish(tx, archive_records))
            if prepared_input_context is not None:
                if type(self._execution) is not ExecutionRepositoryImpl:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                updated_execution = await self._execution.claim_prepared_agent_run_in_transaction(
                    execution_transaction, claim.execution_id, tenant_id=self._tenant_id,
                    expected_revision=claim.expected_execution_revision,
                    expected_agent_run_seq=claim.expected_agent_run_seq,
                    input_context=prepared_input_context, producer_claim_id=producer_claim_id,
                )
            else:
                updated_execution = await self._execution.claim_next_agent_run_in_transaction(
                    execution_transaction,
                    claim.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=claim.expected_execution_revision,
                    expected_agent_run_seq=claim.expected_agent_run_seq,
                )
                await self._execution.admit_history_producer_in_transaction(
                    execution_transaction, updated_execution,
                    producer_claim_id=producer_claim_id,
                )
            updated_recovery = replace(
                current_recovery,
                agent_run_id=next_agent_run_id,
                state=RecoveryCheckpointState.ACTIVE,
                revision=current_recovery.revision + 1,
                updated_at=updated_execution.updated_at,
            )
            updated_recovery = await self._recovery.compare_and_swap_in_transaction(
                recovery_transaction,
                claim.execution_id,
                tenant_id=self._tenant_id,
                expected_revision=claim.expected_recovery_revision,
                next_record=updated_recovery,
            )
            if recovery_operation is not None:
                if operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                await operations.append_in_transaction(
                    group.transaction(operations.state_store), recovery_operation,
                )
            return updated_execution, updated_recovery

        if recovery_operation is not None:
            async def readback() -> CommitObservation[tuple[ExecutionRecord, RecoveryCheckpoint]]:
                if operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                existing = await operations.get(
                    recovery_operation.operation_id, tenant_id=self._tenant_id,
                )
                if existing is None:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if (
                    existing.execution_id != claim.execution_id
                    or not operation_identity_matches(existing, recovery_operation)
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
                    )
                current_execution = await self._load_execution(
                    claim.execution_id, tenant_id=self._tenant_id,
                )
                current_recovery = await self._recovery.get(
                    claim.execution_id, tenant_id=self._tenant_id,
                )
                if current_execution is None or current_recovery is None:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=(current_execution, current_recovery),
                )

            outcome = await run_durable_commit(
                lambda: stores[0].storage_group.mutate(stores, callback),
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
            execution = await self._load_execution(
                claim.execution_id,
                tenant_id=self._tenant_id,
            )
            recovery = await self._recovery.get(
                claim.execution_id,
                tenant_id=self._tenant_id,
            )
            head = await self._execution.get_history_head(claim.execution_id, tenant_id=self._tenant_id)
            own_claim = (
                head is not None
                and head.producer_generation == producer_generation
                and head.producer_claim_id == producer_claim_id
            )
            execution_target = (
                execution is not None
                and execution.revision >= claim.expected_execution_revision + (2 if prepared_input_context is not None else 1)
                and (prepared_input_context is None or execution.input_context == prepared_input_context)
                and execution.agent_run_seq == next_sequence
            )
            recovery_target = (
                recovery is not None
                and recovery.revision >= claim.expected_recovery_revision + 1
                and recovery.state is RecoveryCheckpointState.ACTIVE
                and recovery.agent_run_id == next_agent_run_id
            )
            execution_predecessor = (
                execution is not None
                and execution.revision == claim.expected_execution_revision
                and execution.agent_run_seq == claim.expected_agent_run_seq
            )
            recovery_predecessor = (
                recovery is not None
                and recovery.revision == claim.expected_recovery_revision
                and recovery.state is claim.expected_recovery_state
                and recovery.agent_run_id is None
            )
            if own_claim and execution_target and recovery_target:
                _logger.warning(
                    "agent attempt commit outcome reconciled: execution=%s sequence=%s",
                    claim.execution_id,
                    next_sequence,
                )
                return execution, recovery
            if not (execution_predecessor and recovery_predecessor):
                if not own_claim:
                    raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False) from error
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
    ) -> tuple[ToolOperationRecord, bool]:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if result_payload is not None:
            if error_code is not None or error_payload is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif error_code is None or error_payload is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            decode_tool_operation_failure(error_code, error_payload)
        expected_status = (
            ToolOperationStatus.COMPLETED
            if result_payload is not None
            else ToolOperationStatus.FAILED
        )
        terminal_error_code: str | None = None
        terminal_error_payload: StoredPayload | None = None
        if result_payload is None:
            terminal_error_code = cast(str, error_code)
            terminal_error_payload = cast(StoredPayload, error_payload)
        stores = [tools.state_store]
        if self._observe_cancellation:
            stores.append(self._execution.state_store)
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        cancelled = False
        cancellation_requested = False

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            nonlocal cancellation_requested
            transaction = group.transaction(tools.state_store)
            if result_payload is not None:
                record = await tools.complete_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    result_payload=result_payload,
                )
            else:
                record = await tools.fail_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    error_code=terminal_error_code,
                    error_payload=terminal_error_payload,
                )
            if self._observe_cancellation:
                execution = await self._execution.get_in_transaction(
                    group.transaction(self._execution.state_store), record.execution_id, tenant_id=tenant_id,
                )
                if execution is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                cancellation_requested = execution.status in {
                    ExecutionStatus.CANCELLING, ExecutionStatus.CANCELLED,
                }
            return record

        async def readback() -> CommitObservation[ToolOperationRecord]:
            nonlocal cancellation_requested
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
                    or observed.error_payload != terminal_error_payload
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                if self._observe_cancellation:
                    execution = await self._load_execution(observed.execution_id, tenant_id=tenant_id)
                    if execution is None:
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                        )
                    cancellation_requested = execution.status in {
                        ExecutionStatus.CANCELLING, ExecutionStatus.CANCELLED,
                    }
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
                return result.value, cancellation_requested
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

    async def commit_tool_deferred(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        stores = [tools.state_store]

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            return await tools.defer_in_transaction(
                group.transaction(tools.state_store),
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
            )

        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await tools.get_operation(
                tool_operation_id,
                tenant_id=tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if (
                observed.status is ToolOperationStatus.PENDING
                and observed.owner is None
                and observed.lease_expires_at is None
                and observed.fence == fence
            ):
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
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        result = await run_durable_commit(
            lambda: stores[0].storage_group.mutate(stores, callback),
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED and result.value is not None:
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.error is not None:
            raise result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

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
            if request.producer_generation is not None:
                stores.append(self._execution.state_store)
                if not _same_group(stores):
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

            async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
                if request.producer_generation is not None:
                    execution_transaction = group.transaction(self._execution.state_store)
                    await self._execution.require_open_history_head_in_transaction(
                        execution_transaction,
                        request.execution_id,
                        expected_producer_generation=request.producer_generation,
                        allow_cancelling=False,
                    )
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
                    tenant_id=self._tools.tenant_id,
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
                result = await attempt()
                if result.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                return result
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                if request.producer_generation is not None:
                    head = await self._execution.get_history_head(
                        request.execution_id, tenant_id=self._tenant_id,
                    )
                    execution = await self._load_execution(request.execution_id, tenant_id=self._tenant_id)
                    if (
                        head is None
                        or head.state is not ExecutionHistoryState.OPEN
                        or head.producer_generation != request.producer_generation
                        or execution is None
                        or execution.status not in {
                            ExecutionStatus.STARTED,
                            ExecutionStatus.CANCELLING,
                            ExecutionStatus.FINALIZING,
                        }
                    ):
                        raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False) from error
                tools = self._tools
                if tools is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
                observed = await tools.get_operation(
                    request.tool_operation_id,
                    tenant_id=tools.tenant_id,
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

    async def commit_recovery_handoff(
        self,
        checkpoint: RecoveryCheckpoint,
        *,
        expected_revision: int,
        producer_generation: int,
    ) -> RecoveryCheckpoint:
        """Publish a worker's terminal handoff while its producer fence is held."""
        self._require_recovery()
        if (
            checkpoint.state is not RecoveryCheckpointState.HANDOFF
            or checkpoint.handoff_phase is not RecoveryHandoffPhase.PREPARED
            or checkpoint.terminal_handoff is None
            or checkpoint.revision != expected_revision + 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = _dedupe_stores((self._execution.state_store, self._recovery.state_store))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def operation() -> RecoveryCheckpoint:
            async def mutate(group: StateGroupTransaction) -> RecoveryCheckpoint:
                await self._execution.require_open_history_head_in_transaction(
                    group.transaction(self._execution.state_store),
                    checkpoint.execution_id,
                    expected_producer_generation=producer_generation,
                )
                return await self._recovery.compare_and_swap_in_transaction(
                    group.transaction(self._recovery.state_store),
                    checkpoint.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=expected_revision,
                    next_record=checkpoint,
                )
            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[object]:
            observed = await self._recovery.get(checkpoint.execution_id, tenant_id=self._tenant_id)
            if observed == checkpoint:
                return CommitObservation(DurableCommitState.COMMITTED)
            return CommitObservation(DurableCommitState.NOT_COMMITTED)

        await self._commit_or_raise(operation, readback)
        return checkpoint

    async def read_pending_turn(
        self, reservation: ExecutionStartReservation, *, expected_session: SessionRecord | None,
    ) -> ExecutionTerminalCommitResult | None:
        """Confirm a terminal turn from source without retrying its write."""
        if (not reservation.execution.conversation
                or reservation.execution.session_id != (None if expected_session is None else expected_session.session_id)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, self._recovery.state_store]
        if expected_session is not None:
            stores.append(self._conversation.state_store)
        elif self._recovery_run_store is not None:
            stores.extend(self._recovery_run_store.transaction_stores)
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def read(group: StateGroupTransaction) -> ExecutionTerminalCommitResult | None:
            return await self._read_pending_turn_in_group(group, reservation, expected_session)

        return await stores[0].storage_group.read_many(stores, read)

    async def _read_pending_turn_in_group(
        self, group: StateGroupTransaction, reservation: ExecutionStartReservation,
        expected_session: SessionRecord | None,
    ) -> ExecutionTerminalCommitResult | None:
        result = await self._execution.pending_turn_result_in_transaction(
            group.transaction(self._execution.state_store), reservation,
        )
        if result is None:
            return None
        execution = result.execution
        if (not execution.conversation or not execution.retention_closed
                or execution.session_id != reservation.execution.session_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        recovery = await self._recovery.get_in_transaction(
            group.transaction(self._recovery.state_store), execution.execution_id, tenant_id=self._tenant_id,
        )
        expected_run_id = (None if execution.agent_run_seq == 0 else agent_run_id(
            namespace=self._namespace, tenant_id=self._tenant_id,
            execution_id=execution.execution_id, agent_run_seq=execution.agent_run_seq,
        ))
        if (recovery is None or recovery.state is not RecoveryCheckpointState.COMPLETED
                or recovery.agent_run_id != expected_run_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is None and execution.status is ExecutionStatus.SUCCEEDED:
            if self._recovery_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            run = await self._recovery_run_store.get_agent_run(agent_run_id=recovery.agent_run_id)
            checkpoint = await self._recovery_run_store.latest_checkpoint(agent_run_id=recovery.agent_run_id)
            if (run is None or checkpoint is None or checkpoint.pending_request_index is not None
                    or checkpoint.agent_run_id != run.agent_run_id
                    or run.parent_agent_run_id is not None
                    or run.agent_id != execution.agent_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (await self._recovery_run_store.checkpoint_count(agent_run_id=run.agent_run_id) != 1
                    or not await self._recovery_run_store.verify_checkpoint_projection(
                        agent_run_id=run.agent_run_id, checkpoint=checkpoint,
                    )
                    or await self._recovery_run_store.transcript_message_count_for_run(run) != len(checkpoint.messages)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is not None:
            transaction = group.transaction(self._conversation.state_store)
            session = await self._conversation.get_in_transaction(
                transaction, expected_session.session_id, tenant_id=self._tenant_id,
            )
            committed = await self._conversation.pending_turn_commit_in_transaction(
                transaction, session_id=expected_session.session_id, execution_id=execution.execution_id,
            )
            successful = execution.status is ExecutionStatus.SUCCEEDED
            if (session is None or committed is None or session.history_id != expected_session.history_id
                    or committed.status is not execution.status
                    or committed.stored_user_input.digest != execution.stored_user_input.digest
                    or committed.error_code != execution.error_code
                    or committed.safe_error_details != execution.safe_error_details
                    or (committed.output is not None) != successful
                    or (result.result.output is not None) != successful):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if successful:
                if (execution.conversation_agent_run_id is None
                        or committed.end_message_index <= committed.start_message_index
                        or committed.output.digest != result.result.output.digest):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif committed.start_message_index != 0 or committed.end_message_index != 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result

    async def read_pending_turn_promotion(
        self, reservation: ExecutionStartReservation, *, expected_session: SessionRecord | None,
        captured_input_context: RuntimePayloadRef | None, producer_claim_id: str,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint] | None:
        """Confirm only the caller's first durable promotion receipt."""
        if not isinstance(producer_claim_id, str) or not producer_claim_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        initial, identity = reservation.execution, reservation.idempotency
        if initial.session_id != (None if expected_session is None else expected_session.session_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, self._recovery.state_store]
        if expected_session is not None:
            stores.append(self._conversation.state_store)
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        claim = ExecutionStartClaim(initial.execution_id, 0, 0, identity.scope,
                                    identity.idempotency_key_digest, identity.request_digest, initial.created_at)

        async def read(group: StateGroupTransaction) -> tuple[ExecutionRecord, RecoveryCheckpoint] | None:
            idempotency = await self._execution.get_start_idempotency(claim)
            if idempotency is None:
                return None
            if (idempotency.request_digest != identity.request_digest
                    or idempotency.resource_kind != identity.resource_kind):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if (idempotency.resource_id != initial.execution_id
                    or idempotency.status is not IdempotencyStatus.STARTED):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            execution = await self._execution.get_in_transaction(
                group.transaction(self._execution.state_store), initial.execution_id, tenant_id=self._tenant_id,
            )
            recovery = await self._recovery.get_in_transaction(
                group.transaction(self._recovery.state_store), initial.execution_id, tenant_id=self._tenant_id,
            )
            head = await self._execution.get_history_head(initial.execution_id, tenant_id=self._tenant_id)
            prepared = captured_input_context is not None
            if (execution is None or recovery is None or head is None
                    or execution.status is not ExecutionStatus.STARTED
                    or execution.session_id != initial.session_id or execution.event_seq != 1
                    or execution.revision != (3 if prepared else 1)
                    or execution.agent_run_seq != (1 if prepared else 0)
                    or execution.input_context != (captured_input_context if prepared else initial.input_context)
                    or recovery.revision != (1 if prepared else 0)
                    or recovery.state is not (RecoveryCheckpointState.ACTIVE if prepared else RecoveryCheckpointState.ADMITTED)
                    or recovery.agent_run_id != (agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                                              execution_id=initial.execution_id, agent_run_seq=1)
                                                 if prepared else None)
                    or head.state is not ExecutionHistoryState.OPEN
                    or (prepared and (head.producer_generation != 2 or head.producer_claim_id != producer_claim_id))
                    or (not prepared and (head.producer_claim_id is not None or head.producer_generation != 0))):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if expected_session is not None:
                session = await self._conversation.get_in_transaction(
                    group.transaction(self._conversation.state_store), expected_session.session_id, tenant_id=self._tenant_id,
                )
                if (session.active_execution_id != initial.execution_id
                        or session.continuation != expected_session.continuation):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            return execution, recovery

        return await stores[0].storage_group.read_many(stores, read)

    async def promote_pending_turn(
        self, reservation: ExecutionStartReservation, *,
        expected_session: SessionRecord | None, captured_input_context: RuntimePayloadRef | None,
        producer_claim_id: str | None = None,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
    ) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
        """Confirm durable producer ownership before a local turn performs effects."""
        self._require_recovery()
        producer_claim_id = uuid4().hex if producer_claim_id is None else producer_claim_id
        if not isinstance(producer_claim_id, str) or not producer_claim_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is not None:
            self._require_conversation()
        initial = reservation.execution
        if (type(self._execution) is not ExecutionRepositoryImpl
                or (expected_session is not None and type(self._conversation) is not SessionRepositoryImpl)
                or type(self._recovery) is not RecoveryCheckpointRepositoryImpl):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (initial.status is not ExecutionStatus.PENDING_START or initial.revision != 0
                or initial.event_seq != 0 or initial.agent_run_seq != 0
                or initial.binding_kind != "agent" or initial.parent_execution_id is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is None:
            if (initial.session_id is not None or initial.conversation_agent_run_id is not None
                    or initial.previous_execution_id is not None or initial.fork_base_execution_id is not None
                    or initial.lineage_kind is not ExecutionLineageKind.RUN
                    or initial.root_execution_id != initial.execution_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif (initial.session_id != expected_session.session_id
              or expected_session.active_execution_id is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, self._recovery.state_store]
        if expected_session is not None:
            stores.append(self._conversation.state_store)
        stores = tuple(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        identity = reservation.idempotency
        claim = ExecutionStartClaim(initial.execution_id, 0, 0, identity.scope,
                                    identity.idempotency_key_digest, identity.request_digest,
                                    initial.created_at)
        admitted = RecoveryCheckpoint(initial.execution_id, None, RecoveryCheckpointState.ADMITTED,
                                      0, initial.created_at, initial.created_at)
        resolved: tuple[ExecutionRecord, RecoveryCheckpoint] | None = None
        confirmed_head = None
        rejected = False

        async def mutate(group: StateGroupTransaction) -> tuple[ExecutionRecord, RecoveryCheckpoint]:
            nonlocal resolved, confirmed_head, rejected
            if expected_session is not None:
                session = await self._conversation.get_in_transaction(
                    group.transaction(self._conversation.state_store),
                    expected_session.session_id, tenant_id=self._tenant_id,
                )
                if session != expected_session:
                    rejected = True
                    raise AIError(ErrorCode.SESSION_BUSY)
            try:
                reserved = await self._execution.reserve_start(reservation)
            except AIError as error:
                if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                    rejected = True
                raise
            if not reserved.created:
                rejected = True
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            started = await self.commit_start_attempt_checkpoint(
                claim, recovery_checkpoint=admitted,
                session_id=None if expected_session is None else expected_session.session_id,
                expected_cursor=None if expected_session is None else expected_session.continuation,
            )
            if captured_input_context is None:
                resolved = (started, admitted)
            else:
                resolved = await self._commit_agent_attempt_checkpoint(
                    AgentAttemptClaim(initial.execution_id, started.revision, 0, 0,
                                      RecoveryCheckpointState.ADMITTED),
                    prepared_input_context=captured_input_context, producer_claim_id=producer_claim_id,
                    context_write=context_write,
                )
            confirmed_head = await self._execution.get_history_head(
                initial.execution_id, tenant_id=self._tenant_id,
            )
            if (confirmed_head is None
                    or (captured_input_context is not None
                        and (confirmed_head.producer_claim_id is None
                             or confirmed_head.producer_generation != 2))):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return resolved

        async def operation() -> object:
            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[object]:
            nonlocal resolved
            if rejected:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            try:
                observed = await self.read_pending_turn_promotion(
                    reservation, expected_session=expected_session,
                    captured_input_context=captured_input_context, producer_claim_id=producer_claim_id,
                )
            except AIError as error:
                if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
                if error.code is ErrorCode.STORAGE_CONFLICT:
                    own_execution = await self._execution.get(initial.execution_id, tenant_id=self._tenant_id)
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED if own_execution is None else DurableCommitState.UNRESOLVED,
                        error=error,
                    )
                raise
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            resolved = observed
            return CommitObservation(DurableCommitState.COMMITTED, value=observed)

        await self._commit_or_raise(operation, readback)
        assert resolved is not None
        return resolved

    async def retain_conversation_turn_payloads(
        self, execution: ExecutionRecord, result: ResultRecord | None,
    ) -> tuple[StoredUserInput, StoredPayload | None]:
        archive = self._conversation_run_store
        if archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history = archive.transcript_repository
        stored_input = replace(
            execution.stored_user_input,
            payload=await history.retain_payload(
                execution.stored_user_input.payload, source_domain=PersistenceDomain.EXECUTION,
            ),
        )
        output = None if result is None else result.output
        if output is not None:
            output = await history.retain_payload(output, source_domain=PersistenceDomain.EXECUTION)
        return stored_input, output

    async def commit_pending_turn(
        self, reservation: ExecutionStartReservation, commit: ExecutionTerminalCommit, *,
        expected_session: SessionRecord | None = None, next_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        recovery_run: AgentRunRecord | None = None,
        recovery_agent_run_checkpoint: AgentRunCheckpoint | None = None,
        audit_events: Sequence[ExecutionEventAppend] = (),
        capability_run_id: str | None = None,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
    ) -> ExecutionTerminalCommitResult:
        """Materialize one effect-free local turn at its final atomic boundary."""
        audit_events = tuple(audit_events)
        archive = self._conversation_run_store
        if (type(self._execution) is not ExecutionRepositoryImpl
                or type(self._recovery) is not RecoveryCheckpointRepositoryImpl):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        execution = commit.execution
        successful = execution.status is ExecutionStatus.SUCCEEDED
        for candidate in (reservation.execution, execution):
            if (not candidate.conversation or candidate.binding_kind != "agent"
                    or candidate.parent_execution_id is not None
                    or candidate.parent_invocation_id is not None or candidate.requires_task_invocation_capture
                    or candidate.budget_scope_id is not None or candidate.restart_checkpoints
                    or candidate.durable_trace or candidate.dependency_hold_ids):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                or reservation.budget is not None
                or execution.agent_run_seq not in ({1} if successful else {0, 1})
                or reservation.execution.status is not ExecutionStatus.PENDING_START
                or reservation.execution.agent_run_seq != 0 or reservation.execution.event_seq != 0
                or commit.expected_event_seq != 1 or commit.expected_revision < (2 if successful else 1)
                or commit.operation is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, self._recovery.state_store]
        if expected_session is None:
            if any(value is not None for value in (next_cursor, conversation_agent_run, conversation_checkpoint)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for candidate in (reservation.execution, execution):
                if (candidate.session_id is not None or candidate.conversation_agent_run_id is not None
                        or candidate.previous_execution_id is not None or candidate.fork_base_execution_id is not None
                        or candidate.lineage_kind is not ExecutionLineageKind.RUN
                        or candidate.root_execution_id != candidate.execution_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if successful:
                if (recovery_run is None or recovery_agent_run_checkpoint is None
                        or recovery_agent_run_checkpoint.state != "complete"
                        or recovery_agent_run_checkpoint.pending_request_index is not None
                        or recovery_run.parent_agent_run_id is not None
                        or recovery_run.agent_id != execution.agent_id
                        or recovery_run.agent_run_id != agent_run_id(
                            namespace=self._namespace, tenant_id=self._tenant_id,
                            execution_id=execution.execution_id, agent_run_seq=1,
                        )
                        or any(getattr(recovery_run, name) != getattr(recovery_agent_run_checkpoint, name)
                               for name in ("agent_run_id", "agent_conversation_id", "agent_id", "parent_agent_run_id"))):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if self._recovery_run_store is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                stores.extend(self._recovery_run_store.transaction_stores)
            elif recovery_run is not None or recovery_agent_run_checkpoint is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            if type(self._conversation) is not SessionRepositoryImpl:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if execution.session_id != expected_session.session_id or expected_session.active_execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if successful:
                if recovery_run is not None or recovery_agent_run_checkpoint is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (next_cursor is None or conversation_agent_run is None or conversation_checkpoint is None):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (type(archive) not in {StateStepArchive, ModelInteractionStateStepArchive}
                        or self._conversation_history is None):
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                if (conversation_checkpoint.state != "complete" or conversation_checkpoint.pending_request_index is not None
                        or conversation_agent_run.parent_agent_run_id is not None
                        or any(getattr(conversation_agent_run, name) != getattr(conversation_checkpoint, name)
                               for name in ("agent_conversation_id", "agent_id", "parent_agent_run_id"))
                        or execution.conversation_agent_run_id != conversation_agent_run.agent_run_id
                        or execution.session_id != expected_session.session_id
                        or expected_session.active_execution_id is not None
                        or next_cursor.history_id != expected_session.history_id
                        or next_cursor.agent_run_id != conversation_agent_run.agent_run_id
                        or conversation_checkpoint.agent_run_id != conversation_agent_run.agent_run_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stores.extend((self._conversation.state_store, archive.state_store))
            else:
                if any(value is not None for value in (
                    next_cursor, conversation_agent_run, conversation_checkpoint,
                    recovery_run, recovery_agent_run_checkpoint,
                )) or execution.conversation_agent_run_id is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stores.append(self._conversation.state_store)
        stores = tuple(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        repository = self._execution

        existing = await self.read_pending_turn(reservation, expected_session=expected_session)
        if existing is not None:
            if self._capability_writes is not None:
                prefix = await self._capability_writes.prepare_terminal(
                    execution.execution_id, capability_run_id, stores[0].storage_group,
                )
                try:
                    if not await prefix.readback():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await prefix.settle()
                finally:
                    await prefix.abort()
            return existing
        turn_payloads = None if expected_session is None else await self.retain_conversation_turn_payloads(execution, commit.result)
        prepared = None
        prepared_recovery = None
        history = None
        final_cursor = None
        start = end = 0
        if expected_session is not None and successful:
            assert archive is not None and conversation_agent_run is not None and conversation_checkpoint is not None
            assert next_cursor is not None and self._conversation_history is not None
            _checkpoint, prepared, history = await self._prepare_conversation_checkpoint(
                archive, conversation_agent_run, conversation_checkpoint, next_cursor.history_id,
            )
            if prepared[0].owner_id != next_cursor.history_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if history is None or history.session_id != expected_session.session_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            start, end = _timeline_turn_message_range(history, prepared, expected_session.continuation)
            final_cursor = replace(next_cursor, message_count=end)
        elif successful:
            assert self._recovery_run_store is not None and recovery_run is not None
            assert recovery_agent_run_checkpoint is not None
            _checkpoint, prepared_recovery = await self._recovery_run_store.prepare_run_checkpoint(
                recovery_run, recovery_agent_run_checkpoint,
            )
        recovered = RecoveryCheckpoint(
            execution.execution_id, (None if execution.agent_run_seq == 0 else agent_run_id(
                namespace=self._namespace, tenant_id=self._tenant_id,
                execution_id=execution.execution_id, agent_run_seq=execution.agent_run_seq,
            )),
            RecoveryCheckpointState.COMPLETED, 2, reservation.execution.created_at, execution.updated_at,
        )
        resolved: ExecutionTerminalCommitResult | None = None
        capability_prefix = (await self._capability_writes.prepare_terminal(
            execution.execution_id, capability_run_id, stores[0].storage_group,
        ) if self._capability_writes is not None else None)
        if capability_prefix is not None:
            stores = tuple(_dedupe_stores((*stores, *capability_prefix.stores)))

        fresh_record_keys: dict[StateStore, tuple[bytes, ...]] = {}
        if (isinstance(stores[0].storage_group, GroupRecordReader)
                and expected_session is not None and prepared is not None and archive is not None
                and conversation_agent_run is not None
                and isinstance(self._conversation_history, ConversationHistoryRepositoryImpl)):
            entries = [
                (repository.state_store, repository.pending_turn_record_keys(reservation)),
                (self._conversation.state_store, (self._conversation.session_record_key(expected_session.session_id),
                 self._conversation_history.history_record_key(expected_session.history_id))),
                (self._recovery.state_store, (self._recovery.checkpoint_record_key(execution.execution_id),)),
                *(await archive.checkpoint_publication_record_keys(conversation_agent_run, prepared[0])).items(),
            ]
            if all(store.storage_group is stores[0].storage_group for store, _keys in entries):
                for store, keys in entries:
                    fresh_record_keys[store] = tuple(dict.fromkeys((*fresh_record_keys.get(store, ()), *keys)))
                stores = tuple(_dedupe_stores((*stores, *fresh_record_keys)))

        async def mutate(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
            if context_write is not None:
                content_archive, archive_records = context_write
                await content_archive.store.mutate(lambda tx: content_archive.publish(tx, archive_records))
            if fresh_record_keys and isinstance(group, GroupRecordReader):
                # Mutable owners still use their CAS; immutable records still compare complete bytes.
                await group.get_records(fresh_record_keys)
            execution_transaction = group.transaction(repository.state_store)
            winner = await self._read_pending_turn_in_group(group, reservation, expected_session)
            if winner is not None:
                if capability_prefix is not None and not await capability_prefix.readback():
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return winner
            if expected_session is not None:
                conversation_transaction = group.transaction(self._conversation.state_store)
                current = await self._conversation.get_in_transaction(
                    conversation_transaction, expected_session.session_id, tenant_id=self._tenant_id,
                )
                if current != expected_session:
                    raise AIError(ErrorCode.SESSION_BUSY)
                if successful:
                    assert history is not None and self._conversation_history is not None
                    current_history = await self._conversation_history.get_in_transaction(
                        conversation_transaction, history.history_id, tenant_id=self._tenant_id,
                    )
                    if current_history != history:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
            final = await repository.insert_pending_turn_in_transaction(
                execution_transaction, reservation, commit, audit_events=tuple(audit_events),
            )
            if expected_session is not None:
                if successful:
                    assert archive is not None and conversation_agent_run is not None and prepared is not None
                    assert final_cursor is not None
                    await archive.materialize_checkpoint_in_transaction(
                        group.transaction(archive.state_store), conversation_agent_run, prepared[0],
                    )
                await self._conversation.commit_pending_turn_in_transaction(
                    conversation_transaction, expected=expected_session, execution_id=execution.execution_id,
                    next_cursor=final_cursor, start_message_index=start, end_message_index=end,
                    stored_user_input=turn_payloads[0], output=turn_payloads[1],
                    status=commit.execution.status, created_at=commit.execution.created_at,
                    updated_at=commit.execution.updated_at, error_code=commit.execution.error_code,
                    safe_error_details=commit.execution.safe_error_details,
                )
            elif successful:
                assert self._recovery_run_store is not None and recovery_run is not None
                assert prepared_recovery is not None
                await self._recovery_run_store.materialize_checkpoint_in_transaction(
                    group.transaction(self._recovery_run_store.state_store), recovery_run, prepared_recovery[0],
                )
            await self._recovery.admit_in_transaction(group.transaction(self._recovery.state_store), recovered)
            if capability_prefix is not None:
                await capability_prefix.apply(group)
            return final

        async def operation() -> object:
            nonlocal resolved
            resolved = await stores[0].storage_group.mutate(stores, mutate)
            return resolved

        async def readback() -> CommitObservation[object]:
            nonlocal resolved
            try:
                resolved = await self.read_pending_turn(reservation, expected_session=expected_session)
            except AIError as error:
                if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
                raise
            if resolved is not None and capability_prefix is not None and not await capability_prefix.readback():
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            return CommitObservation(
                DurableCommitState.COMMITTED if resolved is not None else DurableCommitState.NOT_COMMITTED,
                value=resolved,
            )

        try:
            outcome = await run_durable_commit(operation, readback, background_tasks=self._background_tasks)
            if outcome.state is DurableCommitState.COMMITTED:
                assert resolved is not None
                if capability_prefix is not None:
                    await capability_prefix.settle()
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return resolved
            if outcome.state is DurableCommitState.NOT_COMMITTED:
                if outcome.error is not None:
                    raise outcome.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error
        finally:
            if capability_prefix is not None:
                await capability_prefix.abort()

    async def _prepare_conversation_checkpoint(
        self,
        owner: StateStepArchive,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        history_id: str,
        *,
        archive: PreparedArchive | None = None,
    ) -> tuple[AgentRunCheckpoint, PreparedAgentRunCheckpointBatch, ConversationHistoryRecord | None]:
        history_owner = self._conversation_history
        if history_owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        preparation_keys = None
        if (type(history_owner) is ConversationHistoryRepositoryImpl
                and history_owner.namespace == self._namespace and history_owner.tenant_id == self._tenant_id
                and history_owner.state_store is self._conversation.state_store
                and history_owner.state_store is owner.state_store):
            preparation_keys = await owner.conversation_preparation_record_keys(run, checkpoint)
        if preparation_keys is None:
            relocated, prepared = await owner.prepare_conversation_checkpoint(run, checkpoint, archive=archive)
            history = await history_owner.get(history_id, tenant_id=self._tenant_id)
        else:
            requests = dict(preparation_keys)
            requests[history_owner.state_store] = tuple(dict.fromkeys((
                *requests.get(history_owner.state_store, ()), history_owner.history_record_key(history_id),
            )))

            async def capture_preparation(group: StateGroupTransaction):
                if not isinstance(group, GroupRecordReader):
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                captured = await group.get_records(requests)
                descriptor = await history_owner.get_in_transaction(
                    group.transaction(history_owner.state_store), history_id, tenant_id=self._tenant_id,
                )
                observed = {store: {key: captured[store].get(key) for key in keys}
                            for store, keys in preparation_keys.items()}
                return descriptor, observed

            history, observed_records = await owner.state_store.storage_group.read_many(
                tuple(requests), capture_preparation,
            )
            relocated, prepared = await owner.prepare_conversation_checkpoint(
                run, checkpoint, archive=archive, observed_records=observed_records,
            )
        return relocated, prepared, history

    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        expected_execution: ExecutionRecord | None = None,
        capability_run_id: str | None = None,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
        next_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        conversation_message_range: tuple[int, int] | None = None,
        recovery_checkpoint: RecoveryCheckpoint | None = None,
        recovery_run: AgentRunRecord | None = None,
        recovery_agent_run_checkpoint: AgentRunCheckpoint | None = None,
        execution_run: AgentRunRecord | None = None,
        execution_events: Sequence[StepEvent] = (),
        execution_checkpoints: Sequence[AgentRunCheckpoint] = (),
        execution_projections: Sequence[PreparedExecutionProjection] = (),
        audit_events: Sequence[ExecutionEventAppend] = (),
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> ExecutionTerminalCommitResult:
        if expected_execution is not None and (
            expected_execution.execution_id != commit.execution.execution_id
            or expected_execution.revision != commit.expected_revision
            or expected_execution.event_seq != commit.expected_event_seq
        ):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        turn_payloads = (await self.retain_conversation_turn_payloads(commit.execution, commit.result)
                         if session_id is not None and commit.execution.conversation else None)
        effective_commit = commit
        if not commit.execution.durable_trace and (
            execution_run is not None or execution_events or execution_checkpoints or execution_projections
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_projections and (
            self._execution_run_store is None or execution_run is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is None and (conversation_agent_run is not None or conversation_checkpoint is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is None and (recovery_run is not None or recovery_agent_run_checkpoint is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (recovery_run is None) != (recovery_agent_run_checkpoint is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "terminal checkpoint requested: execution=%s session=%s recovery=%s",
            commit.execution.execution_id,
            session_id,
            recovery_agent_run_checkpoint is not None,
        )
        owner_tasks = (
            self._background_tasks
            if background_tasks is None
            else background_tasks
        )
        stores = [self._execution.state_store]
        if session_id is not None:
            self._require_conversation()
            stores.append(self._conversation.state_store)
        if recovery_checkpoint is not None:
            self._require_recovery()
            stores.append(self._recovery.state_store)
        if execution_run is not None:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.extend(self._execution_run_store.transaction_stores)
        if conversation_agent_run is not None or conversation_checkpoint is not None:
            if self._conversation_run_store is None or conversation_agent_run is None or conversation_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.extend(self._conversation_run_store.transaction_stores)
        if recovery_run is not None or recovery_agent_run_checkpoint is not None:
            if self._recovery_run_store is None or recovery_run is None or recovery_agent_run_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.extend(self._recovery_run_store.transaction_stores)
        terminal_archives = tuple(
            archive for archive, participates in (
                (self._execution_run_store, execution_run is not None and bool(execution_projections or execution_checkpoints)),
                (self._conversation_run_store, conversation_agent_run is not None),
                (self._recovery_run_store, recovery_run is not None),
            ) if participates and archive is not None
        )
        shared_archive = None
        if _same_group(stores) and terminal_archives:
            roots = tuple([
                (await archive.transcript_repository.content_archive(create=True)).empty_root()
                for archive in terminal_archives
            ])
            same_content_scope = all(root == roots[0] for root in roots)
            shared_archive = _copy_archive_union(tuple(
                archive for projection in execution_projections
                for archive in (
                    projection.archive,
                    *(checkpoint.archive for checkpoint in projection.checkpoints),
                    *((projection.observation.archive,) if projection.observation is not None else ()),
                )
            )) if same_content_scope else None
        prepared_conversation = ()
        timeline_history = None
        if (
            self._conversation_run_store is not None
            and conversation_agent_run is not None
            and conversation_checkpoint is not None
        ):
            if next_cursor is None or next_cursor.history_id is None or self._conversation_history is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            conversation_checkpoint, prepared_conversation, timeline_history = await self._prepare_conversation_checkpoint(
                self._conversation_run_store, conversation_agent_run, conversation_checkpoint,
                next_cursor.history_id, archive=shared_archive,
            )
        timeline_range: tuple[int, int] | None = conversation_message_range
        if prepared_conversation:
            if (session_id is None or next_cursor is None or timeline_history is None
                    or timeline_history.session_id != session_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            timeline_range = _timeline_turn_message_range(
                timeline_history,
                prepared_conversation,
                expected_cursor,
            )
            next_cursor = replace(
                next_cursor,
                message_count=timeline_range[1],
            )

        prepared_recovery = ()
        if (
            self._recovery_run_store is not None
            and recovery_run is not None
            and recovery_agent_run_checkpoint is not None
        ):
            recovery_agent_run_checkpoint, prepared_recovery = await self._recovery_run_store.prepare_run_checkpoint(
                recovery_run,
                recovery_agent_run_checkpoint,
                archive=shared_archive,
            )
        prepared_execution: PreparedAgentRunCheckpointBatch | tuple[()] = ()
        if (
            not execution_projections
            and self._execution_run_store is not None
            and execution_run is not None
            and execution_checkpoints
        ):
            prepared_execution = await self._execution_run_store.prepare_checkpoints(
                execution_run,
                execution_checkpoints,
                archive=shared_archive,
            )
        seal = _execution_history_seal(
            commit,
            audit_events=audit_events,
            projections=execution_projections,
            current_run=execution_run,
            current_events=execution_events,
            current_batch=(
                prepared_execution
                if isinstance(prepared_execution, PreparedAgentRunCheckpointBatch)
                else None
            ),
        ) if commit.execution.durable_trace else None
        if _same_group(stores):
            terminal_projections = execution_projections
            if shared_archive is not None:
                shared_archive = _copy_archive_union((
                    shared_archive,
                    *(checkpoint.archive for batch in (
                        prepared_conversation, prepared_recovery, prepared_execution,
                    ) for checkpoint in batch),
                ))
                if prepared_conversation:
                    prepared_conversation = replace(prepared_conversation, checkpoints=tuple(
                        replace(checkpoint, archive=PreparedArchive()) for checkpoint in prepared_conversation
                    ))
                if prepared_recovery:
                    prepared_recovery = replace(prepared_recovery, checkpoints=tuple(
                        replace(checkpoint, archive=PreparedArchive()) for checkpoint in prepared_recovery
                    ))
                if prepared_execution:
                    prepared_execution = replace(prepared_execution, checkpoints=tuple(
                        replace(checkpoint, archive=PreparedArchive()) for checkpoint in prepared_execution
                    ))
                terminal_projections = tuple(replace(
                    projection,
                    archive=PreparedArchive(),
                    checkpoints=tuple(replace(checkpoint, archive=PreparedArchive())
                                      for checkpoint in projection.checkpoints),
                    observation=(None if projection.observation is None else
                                 replace(projection.observation, archive=PreparedArchive())),
                ) for projection in execution_projections)
            prepared = (await self._capability_writes.prepare_terminal(
                commit.execution.execution_id, capability_run_id, stores[0].storage_group,
            ) if self._capability_writes is not None else None)
            if prepared is not None:
                stores.extend(prepared.stores)
            fresh_record_keys: dict[StateStore, tuple[bytes, ...]] = {}
            if (shared_archive is not None
                    and isinstance(stores[0].storage_group, GroupRecordReader)
                    and type(self._execution) is ExecutionRepositoryImpl
                    and type(self._conversation) is SessionRepositoryImpl
                    and type(self._conversation_history) is ConversationHistoryRepositoryImpl
                    and type(self._recovery) is RecoveryCheckpointRepositoryImpl
                    and type(self._conversation_run_store) in {StateStepArchive, ModelInteractionStateStepArchive}
                    and conversation_agent_run is not None and prepared_conversation
                    and session_id is not None and next_cursor is not None):
                content = await terminal_archives[0].transcript_repository.content_archive(create=True)
                entries = [
                    (self._execution.state_store, self._execution.terminal_record_keys(
                        commit.execution, include_history_seal=seal is not None,
                    )),
                    (content.store, content.publication_record_keys(shared_archive)),
                    (self._conversation.state_store, (
                        self._conversation.session_record_key(session_id),
                        self._conversation_history.history_record_key(next_cursor.history_id),
                    )),
                    *(await self._conversation_run_store.checkpoint_publication_record_keys(
                        conversation_agent_run, prepared_conversation[0],
                    )).items(),
                ]
                if recovery_checkpoint is not None:
                    entries.append((self._recovery.state_store, (
                        self._recovery.checkpoint_record_key(recovery_checkpoint.execution_id),
                    )))
                if all(store.storage_group is stores[0].storage_group for store, _keys in entries):
                    for store, keys in entries:
                        fresh_record_keys[store] = tuple(dict.fromkeys((*fresh_record_keys.get(store, ()), *keys)))
                    stores = list(_dedupe_stores((*stores, *fresh_record_keys)))
            try:
                async def callback(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
                    nonlocal effective_commit
                    if fresh_record_keys and isinstance(group, GroupRecordReader):
                        await group.get_records(fresh_record_keys)
                    execution_transaction = group.transaction(self._execution.state_store)
                    closed_result = None
                    if (type(self._execution) is ExecutionRepositoryImpl
                            and not commit.execution.durable_trace and execution_run is None):
                        effective_commit = await self._effective_terminal_commit(
                            execution_transaction, commit, expected_execution=expected_execution,
                        )
                        # Establish the execution-first fence and final control CAS
                        # before dependent writes in this same atomic transaction.
                        closed_result = await self._execution.commit_closed_terminal_in_transaction(
                            execution_transaction, effective_commit, pending_events=audit_events,
                        )
                    else:
                        head, head_record = await self._execution.require_open_history_head_in_transaction(
                            execution_transaction,
                            commit.execution.execution_id,
                            expected_producer_generation=commit.producer_generation,
                        )
                        effective_commit = await self._effective_terminal_commit(
                            execution_transaction,
                            commit,
                            expected_execution=expected_execution,
                        )
                    if shared_archive is not None:
                        await terminal_archives[0].transcript_repository.publish_archive(shared_archive)
                    if session_id is not None:
                        conversation_transaction = group.transaction(self._conversation.state_store)
                        if conversation_agent_run is None or conversation_checkpoint is None:
                            if turn_payloads is not None:
                                await self._conversation.commit_timeline_turn_in_transaction(
                                    conversation_transaction, session_id, tenant_id=self._tenant_id,
                                    execution_id=commit.execution.execution_id,
                                    start_message_index=0 if timeline_range is None else timeline_range[0],
                                    end_message_index=0 if timeline_range is None else timeline_range[1],
                                    stored_user_input=turn_payloads[0], output=turn_payloads[1],
                                    status=effective_commit.execution.status, created_at=effective_commit.execution.created_at,
                                    updated_at=effective_commit.execution.updated_at, error_code=effective_commit.execution.error_code,
                                    safe_error_details=effective_commit.execution.safe_error_details,
                                    allow_unadmitted=commit.execution.started_at is None,
                                )
                            await self._conversation.release_execution_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                                execution_id=commit.execution.execution_id,
                                history_unavailable=commit.execution.invalidates_session_history,
                                require_turn_commit=commit.execution.conversation,
                            )
                        else:
                            if next_cursor is None or self._conversation_run_store is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            await self._conversation_run_store.materialize_checkpoint_in_transaction(
                                group.transaction(self._conversation_run_store.state_store),
                                conversation_agent_run,
                                prepared_conversation[0],
                            )
                            session = await self._conversation.get_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                            )
                            await self._promote_history_in_transaction(
                                conversation_transaction,
                                session,
                                prepared_conversation[0],
                            )
                            await self._conversation.complete_execution_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                                execution_id=commit.execution.execution_id,
                                expected=expected_cursor,
                                next_cursor=next_cursor,
                                history_quality="complete",
                            )
                            if timeline_range is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            await self._conversation.commit_timeline_turn_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                                execution_id=commit.execution.execution_id,
                                start_message_index=timeline_range[0],
                                end_message_index=timeline_range[1],
                                stored_user_input=turn_payloads[0], output=turn_payloads[1],
                                status=effective_commit.execution.status, created_at=effective_commit.execution.created_at,
                                updated_at=effective_commit.execution.updated_at, error_code=effective_commit.execution.error_code,
                                safe_error_details=effective_commit.execution.safe_error_details,
                            )
                    if recovery_checkpoint is not None:
                        if recovery_run is not None or recovery_agent_run_checkpoint is not None:
                            if self._recovery_run_store is None or recovery_run is None or recovery_agent_run_checkpoint is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            await self._recovery_run_store.materialize_checkpoint_in_transaction(
                                group.transaction(self._recovery_run_store.state_store),
                                recovery_run,
                                prepared_recovery[0],
                            )
                        await self._recovery.compare_and_swap_in_transaction(
                            group.transaction(self._recovery.state_store),
                            recovery_checkpoint.execution_id,
                            tenant_id=self._tenant_id,
                            expected_revision=recovery_checkpoint.revision - 1,
                            next_record=recovery_checkpoint,
                        )
                    if execution_run is not None:
                        execution_transaction_for_steps = group.transaction(
                            self._execution_run_store.state_store
                        )
                        if execution_projections:
                            for projection in terminal_projections:
                                await self._execution_run_store.transcript_repository.publish_archive(projection.archive)
                                await self._execution_run_store.sync_projection_in_transaction(
                                    execution_transaction_for_steps,
                                    projection.run,
                                    events=projection.events,
                                    checkpoints=projection.checkpoints,
                                    interactions=projection.interactions,
                                    observation=projection.observation,
                                    producer_generation=projection.producer_generation,
                                    execution_id=commit.execution.execution_id,
                                    history_head_guard=(head, head_record),
                                )
                        else:
                            await self._execution_run_store.sync_projection_in_transaction(
                                execution_transaction_for_steps,
                                execution_run,
                                events=execution_events,
                                checkpoints=(
                                    prepared_execution.checkpoints
                                    if isinstance(
                                        prepared_execution,
                                        PreparedAgentRunCheckpointBatch,
                                    )
                                    else ()
                                ),
                                execution_id=commit.execution.execution_id,
                                history_head_guard=(head, head_record),
                            )
                    if prepared is not None:
                        await prepared.apply(group)
                    if closed_result is not None:
                        return closed_result
                    if seal is not None:
                        await self._execution.put_history_seal_in_transaction(execution_transaction, seal)
                    await self._execution.replace_history_head_in_transaction(
                        execution_transaction,
                        head_record,
                        replace(
                            head,
                            state=ExecutionHistoryState.SEALED if seal is not None else ExecutionHistoryState.CLOSED,
                            revision=head.revision + 1,
                            seal_digest=None if seal is None else seal.seal_digest,
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
                            effective_commit,
                            expected_execution=expected_execution,
                            pending_event_count=len(audit_events),
                            session_id=session_id,
                            next_cursor=next_cursor,
                            conversation_agent_run=conversation_agent_run,
                            conversation_checkpoint=conversation_checkpoint,
                            recovery_checkpoint=recovery_checkpoint,
                            recovery_run=recovery_run,
                            recovery_agent_run_checkpoint=recovery_agent_run_checkpoint,
                            execution_run=execution_run,
                            execution_events=execution_events,
                            execution_checkpoints=execution_checkpoints,
                            execution_projections=execution_projections,
                            audit_events=audit_events,
                        )
                        if not visible:
                            return CommitObservation(DurableCommitState.NOT_COMMITTED)
                        if prepared is not None and not await prepared.readback():
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                            )
                        execution = await self._load_execution(
                            commit.execution.execution_id,
                            tenant_id=self._tenant_id,
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
                        if error.code is ErrorCode.EXECUTION_RESULT_CONFLICT:
                            return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
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
                    if prepared is not None:
                        await prepared.settle()
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
            finally:
                if prepared is not None:
                    await prepared.abort()
        recovery_pending = False
        if recovery_checkpoint is not None:
            actual_recovery = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=self._tenant_id,
            )
            if actual_recovery is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if actual_recovery == recovery_checkpoint:
                if not await self._terminal_targets_visible(
                    effective_commit,
                    expected_execution=expected_execution,
                    pending_event_count=len(audit_events),
                    session_id=session_id,
                    next_cursor=next_cursor,
                    conversation_agent_run=conversation_agent_run,
                    conversation_checkpoint=conversation_checkpoint,
                    recovery_checkpoint=recovery_checkpoint,
                    recovery_run=recovery_run,
                    recovery_agent_run_checkpoint=recovery_agent_run_checkpoint,
                    execution_run=execution_run,
                    execution_events=execution_events,
                    execution_checkpoints=execution_checkpoints,
                    execution_projections=execution_projections,
                    audit_events=audit_events,
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution = await self._load_execution(
                    commit.execution.execution_id,
                    tenant_id=self._tenant_id,
                )
                if execution is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if self._capability_writes is not None:
                    prefix = await self._capability_writes.prepare_terminal(
                        commit.execution.execution_id, capability_run_id, self._execution.state_store.storage_group,
                    )
                    try:
                        if not await prefix.readback():
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        await prefix.settle()
                    finally:
                        await prefix.abort()
                return ExecutionTerminalCommitResult(execution, commit.result)
            if not _recovery_completion_predecessor(actual_recovery, recovery_checkpoint):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            recovery_pending = True
        has_recovery_run_target = (
            recovery_run is not None or recovery_agent_run_checkpoint is not None
        )
        recovery_run_store_merged = False
        if session_id is not None and conversation_agent_run is not None and conversation_checkpoint is not None:
            if next_cursor is None or self._conversation_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            conversation_stores = [
                self._conversation.state_store,
                self._conversation_run_store.state_store,
            ]

            async def commit_conversation(group: StateGroupTransaction) -> None:
                conversation_transaction = group.transaction(self._conversation.state_store)
                await self._conversation_run_store.materialize_checkpoint_in_transaction(
                    group.transaction(self._conversation_run_store.state_store),
                    conversation_agent_run,
                    prepared_conversation[0],
                )
                session = await self._conversation.get_in_transaction(
                    conversation_transaction,
                    session_id,
                    tenant_id=self._tenant_id,
                )
                await self._promote_history_in_transaction(
                    conversation_transaction,
                    session,
                    prepared_conversation[0],
                )
                await self._conversation.advance_continuation_in_transaction(
                    group.transaction(self._conversation.state_store),
                    session_id,
                    tenant_id=self._tenant_id,
                    execution_id=commit.execution.execution_id,
                    expected=expected_cursor,
                    next_cursor=next_cursor,
                    release_execution=False,
                    history_quality="complete",
                )
                if timeline_range is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
                            tenant_id=self._tenant_id,
                            execution_id=commit.execution.execution_id,
                            next_cursor=next_cursor,
                            run=conversation_agent_run,
                            checkpoint=conversation_checkpoint,
                        ):
                            break
            else:
                await self._materialize_prepared_checkpoint_with_reconciliation(
                    self._conversation_run_store,
                    conversation_agent_run,
                    prepared_conversation,
                )

                async def commit_conversation_state(
                    transaction: StateTransaction,
                ) -> None:
                    session = await self._conversation.get_in_transaction(
                        transaction,
                        session_id,
                        tenant_id=self._tenant_id,
                    )
                    await self._promote_history_in_transaction(
                        transaction,
                        session,
                        prepared_conversation[0],
                    )
                    await self._conversation.advance_continuation_in_transaction(
                        transaction,
                        session_id,
                        tenant_id=self._tenant_id,
                        execution_id=commit.execution.execution_id,
                        expected=expected_cursor,
                        next_cursor=next_cursor,
                        release_execution=False,
                        history_quality="complete",
                    )
                    if timeline_range is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

                for attempt in range(2):
                    try:
                        await self._conversation.state_store.mutate(
                            commit_conversation_state
                        )
                        break
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                            raise
                        session = await self._load_session(
                            session_id,
                            tenant_id=self._tenant_id,
                        )
                        if session is not None and session.continuation == next_cursor:
                            break

        execution_stores = [self._execution.state_store]
        if execution_run is not None:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution_stores.extend(self._execution_run_store.transaction_stores)
        recovery_state_merged = (
            recovery_pending
            and recovery_checkpoint is not None
            and self._recovery.state_store.storage_group is self._execution.state_store.storage_group
        )
        if recovery_state_merged:
            if has_recovery_run_target:
                if self._recovery_run_store is None or recovery_run is None or recovery_agent_run_checkpoint is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                recovery_run_store_merged = (
                    self._recovery_run_store.state_store.storage_group
                    is self._execution.state_store.storage_group
                )
            execution_stores.append(self._recovery.state_store)
            if recovery_run_store_merged:
                execution_stores.extend(self._recovery_run_store.transaction_stores)
        if recovery_pending and has_recovery_run_target and not recovery_run_store_merged:
            if self._recovery_run_store is None or recovery_run is None or recovery_agent_run_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._materialize_prepared_checkpoint_with_reconciliation(
                self._recovery_run_store,
                recovery_run,
                prepared_recovery,
            )

        execution_run_store_in_transaction = False
        execution_prefix: "PreparedCapabilityWrites | None" = None

        async def commit_execution(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
            nonlocal effective_commit
            execution_transaction = group.transaction(self._execution.state_store)
            head, head_record = await self._execution.require_open_history_head_in_transaction(
                execution_transaction,
                commit.execution.execution_id,
                expected_producer_generation=commit.producer_generation,
            )
            effective_commit = await self._effective_terminal_commit(
                execution_transaction, commit, expected_execution=expected_execution,
            )
            if execution_run is not None and execution_run_store_in_transaction:
                execution_transaction_for_steps = group.transaction(
                    self._execution_run_store.state_store
                )
                if execution_projections:
                    for projection in execution_projections:
                        await self._execution_run_store.transcript_repository.publish_archive(projection.archive)
                        await self._execution_run_store.sync_projection_in_transaction(
                            execution_transaction_for_steps,
                            projection.run,
                            events=projection.events,
                            checkpoints=projection.checkpoints,
                            interactions=projection.interactions,
                            observation=projection.observation,
                            producer_generation=projection.producer_generation,
                            execution_id=commit.execution.execution_id,
                            history_head_guard=(head, head_record),
                        )
                else:
                    await self._execution_run_store.sync_projection_in_transaction(
                        execution_transaction_for_steps,
                        execution_run,
                        events=execution_events,
                        checkpoints=(
                            prepared_execution.checkpoints
                            if isinstance(
                                prepared_execution,
                                PreparedAgentRunCheckpointBatch,
                            )
                            else ()
                        ),
                        execution_id=commit.execution.execution_id,
                        history_head_guard=(head, head_record),
                    )
            if seal is not None:
                await self._execution.put_history_seal_in_transaction(execution_transaction, seal)
            await self._execution.replace_history_head_in_transaction(
                execution_transaction,
                head_record,
                replace(
                    head,
                    state=ExecutionHistoryState.SEALED if seal is not None else ExecutionHistoryState.CLOSED,
                    revision=head.revision + 1,
                    seal_digest=None if seal is None else seal.seal_digest,
                ),
            )
            if recovery_state_merged:
                if recovery_run_store_merged:
                    await self._recovery_run_store.materialize_checkpoint_in_transaction(
                        group.transaction(self._recovery_run_store.state_store),
                        recovery_run,
                        prepared_recovery[0],
                    )
                await self._recovery.compare_and_swap_in_transaction(
                    group.transaction(self._recovery.state_store),
                    recovery_checkpoint.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=recovery_checkpoint.revision - 1,
                    next_record=recovery_checkpoint,
                )
            if execution_prefix is not None:
                await execution_prefix.apply(group)
            return await self._execution.commit_terminal_in_transaction(
                execution_transaction,
                effective_commit,
                pending_events=audit_events,
            )

        async def execution_target_visible() -> bool:
            return await self._terminal_targets_visible(
                effective_commit,
                expected_execution=expected_execution,
                pending_event_count=len(audit_events),
                session_id=session_id,
                next_cursor=next_cursor,
                conversation_agent_run=conversation_agent_run,
                conversation_checkpoint=conversation_checkpoint,
                recovery_checkpoint=recovery_checkpoint if recovery_state_merged else None,
                recovery_run=recovery_run if recovery_run_store_merged else None,
                recovery_agent_run_checkpoint=(
                    recovery_agent_run_checkpoint if recovery_run_store_merged else None
                ),
                execution_run=execution_run,
                execution_events=execution_events,
                execution_checkpoints=execution_checkpoints,
                execution_projections=execution_projections,
                audit_events=audit_events,
                require_session_release=False,
                require_recovery=recovery_state_merged,
            )

        async def commit_execution_with_reconciliation(
            stores: Sequence[StateStore],
        ) -> ExecutionTerminalCommitResult:
            nonlocal execution_prefix
            execution_prefix = (await self._capability_writes.prepare_terminal(
                commit.execution.execution_id, capability_run_id, self._execution.state_store.storage_group,
            ) if self._capability_writes is not None else None)
            if execution_prefix is not None:
                stores = tuple(_dedupe_stores((*stores, *execution_prefix.stores)))
            try:
                async def readback() -> CommitObservation[ExecutionTerminalCommitResult]:
                    try:
                        if not await execution_target_visible():
                            return CommitObservation(
                                DurableCommitState.NOT_COMMITTED
                            )
                        if execution_prefix is not None and not await execution_prefix.readback():
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                            )
                        execution = await self._load_execution(
                            commit.execution.execution_id,
                            tenant_id=self._tenant_id,
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
                        if error.code is ErrorCode.EXECUTION_RESULT_CONFLICT:
                            return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
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
                    if execution_prefix is not None:
                        await execution_prefix.settle()
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

            finally:
                if execution_prefix is not None:
                    await execution_prefix.abort()

        if await execution_target_visible():
            if self._capability_writes is not None:
                prefix = await self._capability_writes.prepare_terminal(
                    commit.execution.execution_id, capability_run_id, self._execution.state_store.storage_group,
                )
                try:
                    if not await prefix.readback():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await prefix.settle()
                finally:
                    await prefix.abort()
            execution = await self._load_execution(
                commit.execution.execution_id,
                tenant_id=self._tenant_id,
            )
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = ExecutionTerminalCommitResult(execution, commit.result)
        elif _same_group(execution_stores):
            execution_run_store_in_transaction = execution_run is not None
            result = await commit_execution_with_reconciliation(execution_stores)
        else:
            if execution_projections:
                for projection in execution_projections:
                    await self._sync_prepared_projection_with_reconciliation(
                        self._execution_run_store,
                        projection,
                        execution_id=commit.execution.execution_id,
                    )
            elif execution_run is not None:
                await self._sync_projection_with_reconciliation(
                    execution_run,
                    execution_id=commit.execution.execution_id,
                    events=execution_events,
                    checkpoints=execution_checkpoints,
                )
            terminal_stores = [self._execution.state_store]
            if recovery_state_merged:
                terminal_stores.append(self._recovery.state_store)
                if recovery_run_store_merged:
                    terminal_stores.extend(self._recovery_run_store.transaction_stores)
            result = await commit_execution_with_reconciliation(terminal_stores)
        if recovery_state_merged and recovery_checkpoint is not None:
            actual_recovery = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=self._tenant_id,
            )
            if actual_recovery == recovery_checkpoint:
                recovery_pending = False
            elif actual_recovery is not None and not _recovery_completion_predecessor(
                actual_recovery,
                recovery_checkpoint,
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            async def release_conversation(transaction: StateTransaction) -> None:
                if turn_payloads is not None:
                    await self._conversation.commit_timeline_turn_in_transaction(
                        transaction, session_id, tenant_id=self._tenant_id,
                        execution_id=commit.execution.execution_id,
                        start_message_index=0 if timeline_range is None else timeline_range[0],
                        end_message_index=0 if timeline_range is None else timeline_range[1],
                        stored_user_input=turn_payloads[0], output=turn_payloads[1],
                        status=effective_commit.execution.status, created_at=effective_commit.execution.created_at,
                        updated_at=effective_commit.execution.updated_at, error_code=effective_commit.execution.error_code,
                        safe_error_details=effective_commit.execution.safe_error_details,
                        allow_unadmitted=commit.execution.started_at is None,
                    )
                await self._conversation.release_execution_in_transaction(
                    transaction, session_id, tenant_id=self._tenant_id,
                    execution_id=commit.execution.execution_id,
                    history_unavailable=commit.execution.invalidates_session_history,
                    require_turn_commit=commit.execution.conversation,
                )

            for attempt in range(2):
                try:
                    await self._conversation.state_store.mutate(release_conversation)
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    session = await self._load_session(session_id, tenant_id=self._tenant_id)
                    if session is not None and session.active_execution_id is None:
                        if turn_payloads is not None:
                            await self.verify_conversation_turn(effective_commit.execution, effective_commit.result)
                        break
        if recovery_pending and recovery_checkpoint is not None:
            await self._complete_recovery_checkpoint(recovery_checkpoint)
        return result

    async def finish_conversation_turn(
        self, execution: ExecutionRecord, result: ResultRecord | None,
        *, message_range: tuple[int, int] | None = None,
    ) -> None:
        """Finish a retained turn from its durable terminal outcome before releasing admission."""
        if execution.session_id is None or not execution.conversation:
            return
        if execution.status is ExecutionStatus.SUCCEEDED and message_range is None:
            await self.verify_conversation_turn(execution, result)
            return
        payloads = await self.retain_conversation_turn_payloads(execution, result)
        start, end = (0, 0) if message_range is None else message_range

        async def finish(transaction: StateTransaction) -> None:
            await self._conversation.commit_timeline_turn_in_transaction(
                transaction, execution.session_id, tenant_id=self._tenant_id,
                execution_id=execution.execution_id, start_message_index=start,
                end_message_index=end, stored_user_input=payloads[0], output=payloads[1],
                status=execution.status, created_at=execution.created_at,
                updated_at=execution.updated_at, error_code=execution.error_code,
                safe_error_details=execution.safe_error_details,
                allow_unadmitted=execution.started_at is None,
            )
            await self._conversation.release_execution_in_transaction(
                transaction, execution.session_id, tenant_id=self._tenant_id,
                execution_id=execution.execution_id,
                history_unavailable=execution.invalidates_session_history,
                require_turn_commit=True,
            )

        try:
            await self._conversation.state_store.mutate(finish)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            await self.verify_conversation_turn(execution, result)

    async def verify_conversation_turn(
        self, execution: ExecutionRecord, result: ResultRecord | None,
    ) -> None:
        if execution.session_id is None or not execution.conversation:
            return
        if result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._conversation.state_store.read(
            lambda transaction: self._conversation.pending_turn_commit_in_transaction(
                transaction, session_id=execution.session_id, execution_id=execution.execution_id,
                allow_unadmitted=execution.started_at is None,
            )
        )
        if value is None:
            return
        expected_output = None if result is None else result.output
        if (value.status is not execution.status
                or value.created_at != execution.created_at
                or value.updated_at > execution.updated_at
                or value.updated_at < result.created_at
                or value.error_code != execution.error_code
                or value.safe_error_details != execution.safe_error_details
                or value.stored_user_input.view != execution.stored_user_input.view
                or value.stored_user_input.digest != execution.stored_user_input.digest
                or (value.output is None) != (expected_output is None)
                or value.output is not None and value.output.digest != expected_output.digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _materialize_prepared_checkpoint_with_reconciliation(
        self,
        archive: StateStepArchive,
        run: AgentRunRecord,
        batch: PreparedAgentRunCheckpointBatch,
        *,
        execution_id: str | None = None,
    ) -> None:
        if len(batch) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = batch[0]
        if await self._prepared_checkpoint_visible(archive, run, prepared):
            return

        async def materialize(transaction: StateTransaction) -> None:
            await archive.materialize_checkpoint_in_transaction(
                transaction,
                run,
                prepared,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                visible = await self._prepared_checkpoint_visible(archive, run, prepared)
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
            await archive.transcript_repository.publish_archive(projection.archive)
            await archive.sync_projection_in_transaction(
                transaction,
                projection.run,
                events=projection.events,
                checkpoints=projection.checkpoints,
                interactions=projection.interactions,
                observation=projection.observation,
                producer_generation=projection.producer_generation,
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

    async def _prepared_checkpoint_visible(
        self,
        archive: StateStepArchive,
        run: AgentRunRecord,
        prepared: PreparedAgentRunCheckpoint,
    ) -> bool:
        if await archive.get_agent_run(agent_run_id=run.agent_run_id) != run:
            return False
        checkpoint = await archive.latest_checkpoint(
            agent_run_id=run.agent_run_id,
            include_interrupted=True,
        )
        if checkpoint is None:
            return False
        projection = await archive.transcript_repository.load_projection(
            prepared.owner_id
        )
        return (
            checkpoint.step_index == prepared.stored.step_index
            and checkpoint.timestamp == prepared.stored.timestamp
            and checkpoint.state == prepared.stored.state
            and projection is not None
            and projection.digest == prepared.projection.digest
        )

    async def _step_checkpoint_visible(
        self,
        archive: StateStepArchive,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
    ) -> bool:
        return (
            await archive.get_agent_run(agent_run_id=run.agent_run_id) == run
            and await archive.verify_checkpoint_projection(
                agent_run_id=run.agent_run_id,
                checkpoint=checkpoint,
            )
        )

    async def _conversation_target_visible(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        next_cursor: ConversationCursor,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
    ) -> bool:
        session = await self._load_session(session_id, tenant_id=tenant_id)
        return (
            session is not None
            and session.active_execution_id == execution_id
            and session.continuation == next_cursor
            and self._conversation_run_store is not None
            and await self._step_checkpoint_visible(
                self._conversation_run_store,
                run,
                checkpoint,
            )
        )

    async def _sync_projection_with_reconciliation(
        self,
        run: AgentRunRecord,
        *,
        execution_id: str,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
    ) -> None:
        if self._execution_run_store is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def operation() -> None:
            await self._execution_run_store.sync_projection(
                run,
                events=events,
                checkpoints=checkpoints,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                stored_run = await self._execution_run_store.get_agent_run(agent_run_id=run.agent_run_id)
                stored_events = await self._execution_run_store.list_events(
                    agent_run_id=run.agent_run_id
                )
                stored_checkpoint = await self._execution_run_store.latest_checkpoint(
                    agent_run_id=run.agent_run_id,
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
                and (not checkpoints or stored_checkpoint == checkpoints[-1])
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
        current = await self._recovery.get(target.execution_id, tenant_id=self._tenant_id)
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
                    tenant_id=self._tenant_id,
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
                observed = await self._recovery.get(target.execution_id, tenant_id=self._tenant_id)
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
        *,
        expected_execution: ExecutionRecord | None = None,
    ) -> ExecutionTerminalCommit:
        if expected_execution is not None:
            current = await self._execution.get_in_transaction(
                transaction, commit.execution.execution_id, tenant_id=self._tenant_id,
            )
            if current is None or not _same_execution_except_hold_metadata(current, expected_execution):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            commit = replace(
                commit,
                expected_revision=current.revision,
                execution=replace(
                    commit.execution,
                    dependency_hold_ids=current.dependency_hold_ids,
                    updated_at=max(commit.execution.updated_at, current.updated_at),
                ),
            )
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
        execution = await self._load_execution(
            claim.execution_id,
            tenant_id=self._tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.status is ExecutionStatus.PENDING_START
            and execution.revision == claim.expected_revision
            and execution.event_seq == claim.expected_event_seq
        ):
            return None
        if (
            execution.status is not ExecutionStatus.STARTED
            or execution.revision != claim.expected_revision + 1
            or execution.event_seq != claim.expected_event_seq + 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._execution.get_start_idempotency(claim)
        if idempotency is None or idempotency.status is not IdempotencyStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        started_events = await self._events.list(
            claim.execution_id,
            tenant_id=self._tenant_id,
            after_event_seq=claim.expected_event_seq,
            limit=1,
        )
        if (
            len(started_events.items) != 1
            or started_events.items[0].event_seq != claim.expected_event_seq + 1
            or started_events.items[0].event_type
            != ExecutionEventType.EXECUTION_STARTED.value
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if recovery_checkpoint is not None:
            actual = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=self._tenant_id,
            )
            if (
                actual is None
                or actual.execution_id != recovery_checkpoint.execution_id
                or actual.agent_run_id is not None
                or actual.state is not RecoveryCheckpointState.ADMITTED
                or actual.handoff_phase is not RecoveryHandoffPhase.NONE
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            session = await self._load_session(session_id, tenant_id=self._tenant_id)
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
        expected_execution: ExecutionRecord | None = None,
        pending_event_count: int,
        session_id: str | None,
        next_cursor: ConversationCursor | None,
        conversation_agent_run: AgentRunRecord | None,
        conversation_checkpoint: AgentRunCheckpoint | None,
        recovery_checkpoint: RecoveryCheckpoint | None,
        recovery_run: AgentRunRecord | None,
        recovery_agent_run_checkpoint: AgentRunCheckpoint | None,
        execution_run: AgentRunRecord | None,
        execution_events: Sequence[StepEvent],
        execution_checkpoints: Sequence[AgentRunCheckpoint],
        execution_projections: Sequence[PreparedExecutionProjection],
        audit_events: Sequence[ExecutionEventAppend],
        require_session_release: bool = True,
        require_recovery: bool = True,
    ) -> bool:
        execution = await self._load_execution(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.status not in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }
            if expected_execution is not None
            else execution.revision == commit.expected_revision
            and execution.event_seq == commit.expected_event_seq
        ):
            head = await self._execution.get_history_head(
                execution.execution_id, tenant_id=self._tenant_id,
            )
            seal = await self._execution.get_history_seal(
                execution.execution_id, tenant_id=self._tenant_id,
            )
            if head is None or head.state is not ExecutionHistoryState.OPEN or seal is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if recovery_checkpoint is not None and require_recovery:
                recovery = await self._recovery.get(
                    execution.execution_id, tenant_id=self._tenant_id,
                )
                if recovery == recovery_checkpoint:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if expected_execution is not None and not _same_execution_except_hold_metadata(execution, expected_execution):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            return False
        expected_revision = commit.expected_revision + pending_event_count + 1
        expected_event_seq = commit.expected_event_seq + pending_event_count + 1
        expected_terminal = replace(
            commit.execution,
            revision=expected_revision,
            event_seq=expected_event_seq,
            result=commit.result,
        )
        if not _same_execution_except_hold_metadata(execution, expected_terminal):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_seal = await self._execution.get_history_seal(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
        ) if execution.durable_trace else None
        if execution.durable_trace and history_seal is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_head = await self._execution.get_history_head(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if (
            history_head is None
            or history_head.state is not (ExecutionHistoryState.SEALED if execution.durable_trace else ExecutionHistoryState.CLOSED)
            or history_head.seal_digest != (None if history_seal is None else history_seal.seal_digest)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.durable_trace and (execution_projections or execution_run is None):
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
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for projection in execution_projections:
                if not await self._execution_run_store.verify_execution_projection_head(
                    projection
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._execution.get_terminal_idempotency(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
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
                tenant_id=self._tenant_id,
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
            tenant_id=self._tenant_id,
            after_event_seq=commit.expected_event_seq,
            limit=pending_event_count + 1,
        )
        expected_events = tuple(audit_events) + (
            ExecutionEventAppend(commit.terminal_event_type, commit.terminal_event_payload),
        )
        if len(events.items) != len(expected_events) or any(
            actual.event_seq != commit.expected_event_seq + index + 1
            or actual.event_type != expected.event_type
            or actual.payload != expected.payload
            for index, (actual, expected) in enumerate(
                zip(events.items, expected_events, strict=True)
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            session = await self._load_session(
                session_id,
                tenant_id=self._tenant_id,
            )
            if session is None or require_session_release and session.active_execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                require_session_release
                and commit.execution.invalidates_session_history
                and session.history_quality != "unavailable"
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if conversation_agent_run is not None and (
                next_cursor is None or session.continuation != next_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None and commit.execution.conversation and require_session_release:
            await self.verify_conversation_turn(commit.execution, commit.result)
        if recovery_checkpoint is not None and require_recovery:
            actual = await self._recovery.get(
                recovery_checkpoint.execution_id,
                tenant_id=self._tenant_id,
            )
            if actual != recovery_checkpoint:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for archive, run, checkpoint in (
            (
                self._conversation_run_store,
                conversation_agent_run,
                conversation_checkpoint,
            ),
            (
                self._recovery_run_store,
                recovery_run,
                recovery_agent_run_checkpoint,
            ),
        ):
            if run is None and checkpoint is None:
                continue
            if archive is None or run is None or checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_run = await archive.get_agent_run(agent_run_id=run.agent_run_id)
            if not await archive.verify_checkpoint_projection(
                agent_run_id=run.agent_run_id,
                checkpoint=checkpoint,
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if stored_run != run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_run is not None:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_run = await self._execution_run_store.get_agent_run(agent_run_id=execution_run.agent_run_id)
            stored_events = await self._execution_run_store.list_events(agent_run_id=execution_run.agent_run_id)
            if stored_run != execution_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution_events and tuple(stored_events[-len(execution_events) :]) != tuple(execution_events):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_checkpoint = await self._execution_run_store.latest_checkpoint(
                agent_run_id=execution_run.agent_run_id,
                include_interrupted=True,
            )
            if execution_checkpoints and stored_checkpoint != execution_checkpoints[-1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution_checkpoints and not await self._execution_run_store.verify_checkpoint_projection(
                agent_run_id=execution_run.agent_run_id,
                checkpoint=execution_checkpoints[-1],
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
                history_unavailable=owner.invalidates_session_history,
                require_turn_commit=owner.conversation,
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


def _require_durable_record(
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
            pending_operation_id=target.pending_operation_id,
            revision=target.revision,
            updated_at=target.updated_at,
        )
        == target
    )


__all__ = ["StateCommands"]
