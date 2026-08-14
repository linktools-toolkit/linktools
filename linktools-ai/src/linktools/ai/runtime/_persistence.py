#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence contracts and immutable records.

This module contains no backend, filesystem, database, or workflow code.  It
is the single semantic boundary shared by the local and SQL implementations.
"""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    EvaluationStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    StopReason,
    ToolOperationStatus,
    validate_persistence_namespace,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, StorageMetrics
from ..task import TaskGraph, TaskGraphView, TaskLease, TaskNodeView, TaskTerminalRecord
from ._domain import RuntimeDomain
from ._tool import ToolStateRepository


@dataclass(frozen=True, slots=True)
class ConversationCursor:
    step_run_id: str


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    tenant_id: str
    owner_principal_id: str
    binding_digest: str
    status: SessionStatus
    revision: int
    resource_generation: int
    cwd: "str | None"
    metadata: Mapping[str, JsonValue]
    created_at: datetime
    updated_at: datetime
    closed_at: "datetime | None"
    continuation: "ConversationCursor | None" = None


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    execution_id: str
    tenant_id: str
    session_id: "str | None"
    binding_digest: str
    parent_execution_id: "str | None"
    root_execution_id: str
    source_execution_id: "str | None"
    base_execution_id: "str | None"
    lineage_kind: ExecutionLineageKind
    status: ExecutionStatus
    revision: int
    event_sequence: int
    agent_run_sequence: int
    error_code: "str | None"
    safe_error_details: Mapping[str, JsonValue]
    created_at: datetime
    updated_at: datetime
    memory_scope: "str | None" = None
    conversation_step_run_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class ExecutionStartClaim:
    execution_id: str
    tenant_id: str
    expected_revision: int
    expected_event_sequence: int
    scope: str
    key_hash: str
    request_digest: str
    started_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionStartUnknownCommit:
    execution_id: str
    tenant_id: str
    expected_revision: int
    expected_event_sequence: int
    scope: str
    key_hash: str
    request_digest: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionCancelRequestCommit:
    execution_id: str
    tenant_id: str
    expected_revision: int
    expected_event_sequence: int
    operation_id: str
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionStartReservation:
    execution: ExecutionRecord
    idempotency: "IdempotencyRecord"


@dataclass(frozen=True, slots=True)
class ExecutionStartReservationResult:
    execution: ExecutionRecord
    idempotency: "IdempotencyRecord"
    created: bool


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    tenant_id: str
    runtime_domain: RuntimeDomain
    scope: str
    key_hash: str
    request_digest: str
    resource_kind: ResourceKind
    resource_id: str
    status: IdempotencyStatus
    result_digest: "str | None"
    error_code: "str | None"
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        expected = {
            RuntimeDomain.EXECUTION: ResourceKind.EXECUTION,
            RuntimeDomain.EVALUATION: ResourceKind.EVALUATION,
        }.get(self.runtime_domain)
        if expected is None or self.resource_kind is not expected:
            raise ValueError("idempotency resource identity does not match runtime domain")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("idempotency timestamps require timezone awareness")


@dataclass(frozen=True, slots=True)
class ResultRecord:
    execution_id: str
    tenant_id: str
    output_schema_id: "str | None"
    output_schema_revision: "int | None"
    output_schema_fingerprint: "str | None"
    object_ref: "ObjectRef | None"
    stop_reason: StopReason
    input_tokens: int
    output_tokens: int
    total_cost_micros: int
    created_at: datetime

    def __post_init__(self) -> None:
        output_fields = (self.output_schema_id, self.output_schema_revision, self.output_schema_fingerprint)
        if any(value is None for value in output_fields) and any(value is not None for value in output_fields):
            raise ValueError("result output schema fields must be all null or all present")
        if self.object_ref is not None and any(value is None for value in output_fields):
            raise ValueError("result object requires output schema")
        if min(self.input_tokens, self.output_tokens, self.total_cost_micros) < 0:
            raise ValueError("terminal accounting must be non-negative")
        if self.created_at.tzinfo is None:
            raise ValueError("terminal accounting requires an aware timestamp")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    tenant_id: str
    memory_scope_key: str
    content_ref: ObjectRef
    content_digest: str
    metadata: Mapping[str, JsonValue]
    revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    evaluation_id: str
    tenant_id: str
    execution_id: str
    dataset_id: str
    dataset_revision: int
    evaluator_id: str
    evaluator_revision: int
    binding_digest: str
    output_schema_fingerprint: str
    artifact_digest: "str | None"
    status: EvaluationStatus
    revision: int
    metrics: Mapping[str, float | int]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    execution_id: str
    tenant_id: str
    producer: str
    media_type: str
    size: int
    digest: str
    object_ref: ObjectRef
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionTerminalCommit:
    expected_revision: int
    expected_event_sequence: int
    execution: ExecutionRecord
    result: ResultRecord
    terminal_event_type: ExecutionEventType
    terminal_event_payload: Mapping[str, JsonValue]
    idempotency: "IdempotencyTerminalUpdate | None" = None
    operation: "OperationTerminalUpdate | None" = None

    def __post_init__(self) -> None:
        status = self.execution.status
        if status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise ValueError("terminal commit requires a terminal Execution")
        if self.result.execution_id != self.execution.execution_id or self.result.tenant_id != self.execution.tenant_id:
            raise ValueError("terminal result identity mismatch")
        output_fields = (self.result.output_schema_id, self.result.output_schema_revision, self.result.output_schema_fingerprint)
        has_output = all(value is not None for value in output_fields) and self.result.object_ref is not None
        partial_output = any(value is not None for value in output_fields) or self.result.object_ref is not None
        if status is ExecutionStatus.SUCCEEDED and not has_output:
            raise ValueError("successful terminal result requires output")
        if status is not ExecutionStatus.SUCCEEDED and partial_output:
            raise ValueError("failed terminal result cannot contain output")


@dataclass(frozen=True, slots=True)
class IdempotencyTerminalUpdate:
    scope: str
    key_hash: str
    expected_status: IdempotencyStatus
    next_status: IdempotencyStatus
    request_digest: str
    result_digest: "str | None"
    error_code: "str | None"


@dataclass(frozen=True, slots=True)
class OperationTerminalUpdate:
    operation_id: str
    expected_status: OperationStatus
    next_status: OperationStatus
    result_ref: "str | None"
    result_digest: "str | None"
    error_code: "str | None"


@dataclass(frozen=True, slots=True)
class ExecutionTerminalCommitResult:
    execution: ExecutionRecord
    result: ResultRecord


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    approval_id: str
    execution_id: str
    tenant_id: str
    operation_id: str
    status: ApprovalStatus
    idempotency_key_hash: "str | None"
    decision: "ApprovalDecision | None"
    decided_by: "str | None"
    decision_digest: "str | None"
    created_at: datetime
    decided_at: "datetime | None"


@dataclass(frozen=True, slots=True)
class ExternalCallRecord:
    call_id: str
    execution_id: str
    tenant_id: str
    operation_id: str
    status: ExternalCallStatus
    idempotency_key_hash: "str | None"
    object_ref: "ObjectRef | None"
    payload_digest: "str | None"
    created_at: datetime
    supplied_at: "datetime | None"


@dataclass(frozen=True, slots=True)
class RecoveryCheckpoint:
    execution_id: str
    tenant_id: str
    input: "RecoveryExecutionInput"
    step_run_id: str
    agent_run_sequence: int
    state: "RecoveryCheckpointState"
    handoff_phase: "RecoveryHandoffPhase"
    terminal_handoff: "RecoveryTerminalHandoff | None"
    handoff_contract_digest: "str | None"
    pending_operation_id: "str | None"
    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.handoff_phase is RecoveryHandoffPhase.NONE:
            if self.terminal_handoff is not None or self.handoff_contract_digest is not None:
                raise ValueError("unprepared recovery checkpoint cannot contain a handoff")
            if self.state is RecoveryCheckpointState.HANDOFF:
                raise ValueError("unprepared recovery checkpoint cannot be in handoff state")
        elif self.handoff_phase is RecoveryHandoffPhase.COMPLETED:
            if self.state is not RecoveryCheckpointState.COMPLETED:
                raise ValueError("completed recovery checkpoint must be completed")
        elif self.terminal_handoff is None or self.handoff_contract_digest is None:
            raise ValueError("prepared recovery checkpoint requires a handoff contract")
        elif self.state is not RecoveryCheckpointState.HANDOFF:
            raise ValueError("active recovery checkpoint handoff must be in handoff state")


class RecoveryCheckpointState(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    HANDOFF = "handoff"
    COMPLETED = "completed"


class RecoveryHandoffPhase(StrEnum):
    NONE = "none"
    PREPARED = "prepared"
    EXECUTION_COMMITTED = "execution_committed"
    CONVERSATION_RESOLVED = "conversation_resolved"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RecoveryExecutionInput:
    prompt: str
    principal_id: str
    principal_kind: str
    session_id: "str | None"
    memory_scope: "str | None"
    agent_id: str
    prompt_id: str
    binding_digest: str
    lineage_kind: str
    parent_execution_id: "str | None"
    root_execution_id: str
    source_execution_id: "str | None"
    base_execution_id: "str | None"
    idempotency: "RecoveryIdempotencyInput | None"


@dataclass(frozen=True, slots=True)
class RecoveryIdempotencyInput:
    scope: str
    key_hash: str
    request_digest: str


@dataclass(frozen=True, slots=True)
class RecoveryTerminalOutcome:
    terminal_status: ExecutionStatus
    error_code: "str | None"
    safe_error_details: Mapping[str, JsonValue]
    stop_reason: StopReason
    output_schema_id: "str | None"
    output_schema_revision: "int | None"
    output_schema_fingerprint: "str | None"
    recovery_object_ref: "ObjectRef | None"
    input_tokens: int
    output_tokens: int
    total_cost_micros: int
    terminal_event_type: ExecutionEventType
    terminal_event_payload: Mapping[str, JsonValue]
    result_created_at: datetime

    def __post_init__(self) -> None:
        output_fields = (self.output_schema_id, self.output_schema_revision, self.output_schema_fingerprint)
        has_output = all(value is not None for value in output_fields) and self.recovery_object_ref is not None
        partial_output = any(value is not None for value in output_fields) or self.recovery_object_ref is not None
        if self.terminal_status is ExecutionStatus.SUCCEEDED and not has_output:
            raise ValueError("successful recovery outcome requires output")
        if self.terminal_status is not ExecutionStatus.SUCCEEDED and partial_output:
            raise ValueError("failed recovery outcome cannot contain output")
        if min(self.input_tokens, self.output_tokens, self.total_cost_micros) < 0:
            raise ValueError("recovery accounting must be non-negative")
        if self.result_created_at.tzinfo is None:
            raise ValueError("recovery outcome requires an aware timestamp")


@dataclass(frozen=True, slots=True)
class RecoveryConversationIntent:
    session_id: str
    expected_cursor: ConversationCursor | None
    next_cursor: ConversationCursor


@dataclass(frozen=True, slots=True)
class RecoveryTerminalHandoff:
    outcome: RecoveryTerminalOutcome
    source_step_run_id: str
    conversation: RecoveryConversationIntent | None


class RuntimeRepository(Protocol):
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...


class RuntimeTransactionCoordinator(Protocol):
    """Owner-fixed mutation boundary implemented by concrete adapters."""

    @property
    def owner_domain(self) -> RuntimeDomain: ...

    def mutation(self) -> AbstractAsyncContextManager[None]: ...

    def mark_changed(self) -> None: ...


class RuntimeObjectRouter(Protocol):
    """Resolve the ObjectStore selected by Runtime composition."""

    def object_store(self, domain: RuntimeDomain) -> ObjectStore: ...

    def working_object_store(self, domain: RuntimeDomain, *, owner_scope: str) -> ObjectStore: ...

    async def release_object_scope(self, domain: RuntimeDomain, *, owner_scope: str) -> None: ...


class RuntimeRetentionController(Protocol):
    async def release_execution_handoff(self, execution_id: str, *, tenant_id: str) -> None: ...
    async def release_session(self, session_id: str, *, tenant_id: str, continuation: "ConversationCursor | None") -> None: ...
    async def release_task_graph(self, graph_id: str, *, tenant_id: str) -> None: ...
    async def release_evaluation(self, evaluation_id: str, *, tenant_id: str) -> None: ...
    async def close(self) -> None: ...


class SessionRepository(RuntimeRepository, Protocol):
    async def create(self, record: SessionRecord) -> SessionRecord: ...
    async def list(self, *, tenant_id: str, owner_principal_id: str | None = None) -> tuple[SessionRecord, ...]: ...
    async def get_header(self, session_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None: ...
    async def compare_and_swap(self, session_id: str, *, tenant_id: str, expected_revision: int, next_record: SessionRecord) -> SessionRecord: ...
    async def advance_continuation(self, session_id: str, *, tenant_id: str, expected: "ConversationCursor | None", next_cursor: ConversationCursor) -> SessionRecord: ...


class ExecutionRepository(RuntimeRepository, Protocol):
    async def create(self, record: ExecutionRecord) -> ExecutionRecord: ...
    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def get(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None: ...
    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord) -> ExecutionRecord: ...
    async def list_by_session(self, session_id: str, *, tenant_id: str, statuses: frozenset[ExecutionStatus] | None = None) -> tuple[ExecutionRecord, ...]: ...
    async def list_children(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, ...]: ...
    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord: ...
    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult: ...
    async def claim_next_agent_run(self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int) -> ExecutionRecord: ...
    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord: ...
    async def request_cancel(self, commit: ExecutionCancelRequestCommit) -> ExecutionRecord: ...
    async def advance_event_sequence(self, execution_id: str, *, tenant_id: str, expected_sequence: int) -> ExecutionRecord: ...
    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult: ...
    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None: ...


class IdempotencyRepository(RuntimeRepository, Protocol):
    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord: ...
    async def get(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None: ...
    async def list_by_resource(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str) -> tuple[IdempotencyRecord, ...]: ...
    async def compare_and_swap(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord: ...


class EventRepository(RuntimeRepository, Protocol):
    async def append(self, execution_id: str, *, tenant_id: str, expected_sequence: int, event_type: ExecutionEventType, payload: JsonValue) -> "ExecutionEventRecord": ...
    async def list(self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int) -> Page["ExecutionEventRecord"]: ...


@dataclass(frozen=True, slots=True)
class ExecutionEventRecord:
    execution_id: str
    tenant_id: str
    sequence: int
    event_type: ExecutionEventType
    payload: JsonValue


class ApprovalRepository(RuntimeRepository, Protocol):
    async def get_header(self, approval_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create(self, record: ApprovalRecord) -> ApprovalRecord: ...
    async def get(self, approval_id: str, *, tenant_id: str) -> ApprovalRecord | None: ...
    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, idempotency_key_hash: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord: ...
    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]: ...


class ExternalCallRepository(RuntimeRepository, Protocol):
    async def get_header(self, call_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create_call(self, record: ExternalCallRecord) -> ExternalCallRecord: ...
    async def get(self, call_id: str, *, tenant_id: str) -> ExternalCallRecord | None: ...
    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, idempotency_key_hash: str, object_ref: ObjectRef, payload_digest: str, supplied_at: datetime) -> ExternalCallRecord: ...
    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalCallRecord, ...]: ...


class RecoveryCheckpointRepository(RuntimeRepository, Protocol):
    async def create(self, record: RecoveryCheckpoint) -> RecoveryCheckpoint: ...
    async def get(self, execution_id: str, *, tenant_id: str) -> RecoveryCheckpoint | None: ...
    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]: ...
    async def compare_and_swap(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: RecoveryCheckpoint) -> RecoveryCheckpoint: ...


class OperationLedgerRepository(RuntimeRepository, Protocol):
    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord: ...
    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None: ...
    async def compare_and_swap(self, operation_id: str, *, tenant_id: str, expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord: ...
    async def list_pending(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, limit: int) -> tuple[OperationLedgerRecord, ...]: ...
    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str: ...


class TaskRepository(RuntimeRepository, Protocol):
    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create_graph(self, graph: TaskGraph, *, tenant_id: str) -> "TaskGraphView": ...
    async def get_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView | None": ...
    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView": ...
    async def claim(self, graph_id: str, node_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease: ...
    async def renew(self, lease: TaskLease, *, tenant_id: str, lease_seconds: int) -> TaskLease: ...
    async def complete(self, lease: TaskLease, *, tenant_id: str, execution_id: "str | None", result_digest: str) -> "TaskTerminalRecord": ...
    async def fail(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> "TaskTerminalRecord": ...
    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]: ...


class EvaluationRepository(RuntimeRepository, Protocol):
    async def get_header(self, evaluation_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create(self, record: EvaluationRecord) -> EvaluationRecord: ...
    async def get(self, evaluation_id: str, *, tenant_id: str) -> EvaluationRecord | None: ...
    async def compare_and_swap(self, evaluation_id: str, *, tenant_id: str, expected_revision: int, next_record: EvaluationRecord) -> EvaluationRecord: ...
    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]: ...


class MemoryRepository(RuntimeRepository, Protocol):
    async def get_header(self, memory_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def put(self, record: MemoryRecord, *, expected_revision: int | None) -> MemoryRecord: ...
    async def put_with_operation(self, record: MemoryRecord, *, expected_revision: int | None, operation: OperationLedgerInput | None) -> "tuple[MemoryRecord | None, bool]": ...
    async def get(self, memory_id: str, *, tenant_id: str) -> MemoryRecord | None: ...
    async def list(self, *, tenant_id: str, memory_scope_key: str, cursor: "str | None", limit: int) -> "Page[MemoryRecord]": ...
    async def delete(self, memory_id: str, *, tenant_id: str, expected_revision: int) -> None: ...
    async def delete_with_operation(self, memory_id: str, *, tenant_id: str, expected_revision: int | None, operation: OperationLedgerInput | None) -> "tuple[bool, bool]": ...


class ArtifactRepository(RuntimeRepository, Protocol):
    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord: ...
    async def get_header(self, artifact_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None: ...
    async def list_by_execution(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[ArtifactRecord]: ...


@dataclass(frozen=True, slots=True)
class ConversationStore:
    sessions: SessionRepository
    operations: OperationLedgerRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.sessions, self.operations


@dataclass(frozen=True, slots=True)
class ExecutionStore:
    executions: ExecutionRepository
    idempotency: IdempotencyRepository
    events: EventRepository
    operations: OperationLedgerRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.executions, self.idempotency, self.events, self.operations

    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        return await self.executions.commit_terminal(commit)

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        return await self.executions.get_result(execution_id, tenant_id=tenant_id)


@dataclass(frozen=True, slots=True)
class MemoryStore:
    records: MemoryRepository
    operations: OperationLedgerRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.records, self.operations


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    records: ArtifactRepository
    operations: OperationLedgerRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.records, self.operations


@dataclass(frozen=True, slots=True)
class TaskStore:
    tasks: TaskRepository
    operations: OperationLedgerRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.tasks, self.operations


@dataclass(frozen=True, slots=True)
class EvaluationStore:
    records: EvaluationRepository
    idempotency: IdempotencyRepository
    operations: OperationLedgerRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.records, self.idempotency, self.operations


@dataclass(frozen=True, slots=True)
class RecoveryStore:
    approvals: ApprovalRepository
    external_calls: ExternalCallRepository
    checkpoints: RecoveryCheckpointRepository
    operations: OperationLedgerRepository
    tools: ToolStateRepository

    def components(self) -> tuple[RuntimeRepository, ...]:
        return self.approvals, self.external_calls, self.checkpoints, self.operations, self.tools


@dataclass(frozen=True, slots=True)
class RuntimeDomainStates:
    namespace: str
    conversation: ConversationStore
    execution: ExecutionStore
    memory: MemoryStore
    artifact: ArtifactStore
    task: TaskStore
    evaluation: EvaluationStore
    recovery: RecoveryStore
    object_router: "RuntimeObjectRouter | None" = field(default=None, compare=False, repr=False)
    metrics: StorageMetrics = field(default_factory=StorageMetrics, compare=False, repr=False)

    def __post_init__(self) -> None:
        try:
            validate_persistence_namespace(self.namespace)
        except AIError as error:
            raise ValueError("runtime persistence namespace is invalid") from error
        stores = (self.conversation, self.execution, self.memory, self.artifact, self.task, self.evaluation, self.recovery)
        if any(store is None for store in stores):
            raise ValueError("runtime persistence requires every repository")

    def components(self) -> tuple[RuntimeRepository, ...]:
        return tuple(component for store in (self.conversation, self.execution, self.memory, self.artifact, self.task, self.evaluation, self.recovery) for component in store.components())

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        if self.object_router is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self.object_router.object_store(domain)

    def working_object_store(self, domain: RuntimeDomain, *, owner_scope: str) -> ObjectStore:
        if self.object_router is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self.object_router.working_object_store(domain, owner_scope=owner_scope)

    async def release_object_scope(self, domain: RuntimeDomain, *, owner_scope: str) -> None:
        if self.object_router is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self.object_router.release_object_scope(domain, owner_scope=owner_scope)

    async def initialize(self) -> None:
        seen: set[int] = set()
        for component in self.components():
            if id(component) not in seen:
                await component.initialize()
                seen.add(id(component))

    async def close(self) -> None:
        seen: set[int] = set()
        for component in reversed(self.components()):
            if id(component) not in seen:
                await component.close()
                seen.add(id(component))

__all__ = [
    "ApprovalRecord", "ApprovalRepository", "ArtifactRecord", "ArtifactRepository", "ConversationCursor",
    "ArtifactStore", "ConversationStore", "EvaluationRecord", "EvaluationRepository", "EvaluationStore", "ExecutionEventRecord", "ExecutionRecord", "ExecutionStore",
    "ExecutionRepository", "ExecutionStartClaim", "ExecutionStartReservation", "ExecutionStartReservationResult", "ExecutionStartUnknownCommit", "ExecutionCancelRequestCommit", "ExecutionTerminalCommit", "ExecutionTerminalCommitResult", "ExternalCallRecord", "IdempotencyTerminalUpdate", "OperationTerminalUpdate",
    "ExternalCallRepository", "IdempotencyRecord", "IdempotencyRepository", "MemoryRecord", "MemoryRepository", "MemoryStore", "RecoveryCheckpoint", "RecoveryCheckpointRepository", "RecoveryStore",
    "OperationLedgerInput", "OperationLedgerRecord", "OperationLedgerRepository", "ResultRecord", "RuntimeRepository", "RuntimeDomainStates",
    "RecoveryCheckpointState", "RecoveryConversationIntent", "RecoveryExecutionInput", "RecoveryHandoffPhase", "RecoveryIdempotencyInput", "RecoveryTerminalHandoff", "RecoveryTerminalOutcome", "SessionRecord", "SessionRepository", "RuntimeDomain",
    "TaskLease", "TaskNodeView", "TaskRepository", "ToolOperationStatus", "RuntimeObjectRouter",
]
