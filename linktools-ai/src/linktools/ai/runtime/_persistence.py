#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence contracts and immutable records.

This module contains no backend, filesystem, database, or workflow code.  It
is the single semantic boundary shared by the local and SQL implementations.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    BlobStatus,
    EvaluationStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationStatus,
    Page,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    StopReason,
    TaskStatus,
    ToolOperationStatus,
)
from ..task import TaskGraph, TaskGraphView, TaskTerminalRecord
from ._tool import ToolStateStore


class RuntimePersistenceMode(StrEnum):
    IN_MEMORY = "MEMORY"
    FILESYSTEM = "FILE"
    SQL = "SQL"


class RuntimeBackend(StrEnum):
    IN_MEMORY = "memory"
    FILESYSTEM = "file"
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


def backend_mode(backend: RuntimeBackend) -> RuntimePersistenceMode:
    if backend is RuntimeBackend.IN_MEMORY:
        return RuntimePersistenceMode.IN_MEMORY
    if backend is RuntimeBackend.FILESYSTEM:
        return RuntimePersistenceMode.FILESYSTEM
    return RuntimePersistenceMode.SQL


@dataclass(frozen=True, slots=True)
class BlobRef:
    tenant_id: str
    digest: str
    size: int
    locator: str


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
    head_execution_id: "str | None" = None


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
    result_ref: "str | None"
    result_digest: "str | None"
    error_code: "str | None"
    safe_error_details: Mapping[str, JsonValue]
    created_at: datetime
    updated_at: datetime
    memory_namespace: "str | None" = None


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
    scope: str
    key_hash: str
    request_digest: str
    execution_id: str
    status: IdempotencyStatus
    result_digest: "str | None"
    error_code: "str | None"
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ResultRecord:
    execution_id: str
    tenant_id: str
    status: ExecutionStatus
    output_schema_id: str
    output_schema_revision: int
    output_schema_fingerprint: str
    payload_ref: "str | None"
    payload_digest: "str | None"
    stop_reason: StopReason
    input_tokens: int
    output_tokens: int
    total_cost_micros: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class OperationLedgerRecord:
    operation_id: str
    tenant_id: str
    resource_kind: ResourceKind
    resource_id: str
    execution_id: "str | None"
    kind: OperationKind
    status: OperationStatus
    request_digest: str
    result_ref: "str | None"
    result_digest: "str | None"
    error_code: "str | None"
    compactable: bool
    sequence: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class OperationLedgerInput:
    operation_id: str
    tenant_id: str
    resource_kind: ResourceKind
    resource_id: str
    execution_id: "str | None"
    kind: OperationKind
    status: OperationStatus
    request_digest: str
    result_ref: "str | None"
    result_digest: "str | None"
    error_code: "str | None"
    compactable: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    tenant_id: str
    owner_id: str
    kind: str
    content_ref: str
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
    blob_ref: str
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
    session_head: "SessionHeadAdvance | None" = None


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
class SessionHeadAdvance:
    session_id: str
    expected_head_execution_id: "str | None"
    next_head_execution_id: str


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
    decision_id: "str | None"
    decision: "ApprovalDecision | None"
    decided_by: "str | None"
    decision_digest: "str | None"
    created_at: datetime
    decided_at: "datetime | None"


@dataclass(frozen=True, slots=True)
class ExternalResultRecord:
    call_id: str
    execution_id: str
    tenant_id: str
    operation_id: str
    status: ExternalCallStatus
    result_id: "str | None"
    payload_ref: "str | None"
    payload_digest: "str | None"
    created_at: datetime
    supplied_at: "datetime | None"


@dataclass(frozen=True, slots=True)
class TaskLease:
    graph_id: str
    task_id: str
    tenant_id: str
    owner: str
    fence: int
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class TaskNodeView:
    graph_id: str
    task_id: str
    dependencies: "tuple[str, ...]"
    status: TaskStatus
    owner: "str | None"
    fence: int
    lease_expires_at: "datetime | None"
    result_digest: "str | None"
    error_code: "str | None"
    error_digest: "str | None"
    execution_id: "str | None" = None


class RuntimeRepository(Protocol):
    @property
    def mode(self) -> RuntimePersistenceMode: ...

    @property
    def backend(self) -> RuntimeBackend: ...

    @property
    def namespace(self) -> str: ...

    @property
    def atomic_domain_id(self) -> str: ...

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...


class SessionRepository(RuntimeRepository, Protocol):
    async def create(self, record: SessionRecord) -> SessionRecord: ...
    async def list(self, *, tenant_id: str, owner_principal_id: str | None = None) -> tuple[SessionRecord, ...]: ...
    async def get_header(self, session_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None: ...
    async def compare_and_swap(self, session_id: str, *, tenant_id: str, expected_revision: int, next_record: SessionRecord) -> SessionRecord: ...


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
    async def advance_sequence(self, execution_id: str, *, tenant_id: str, kind: str, expected_sequence: int) -> ExecutionRecord: ...
    async def commit_terminal(self, execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord) -> ExecutionRecord: ...


class IdempotencyRepository(RuntimeRepository, Protocol):
    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord: ...
    async def get(self, scope: str, key_hash: str, *, tenant_id: str) -> IdempotencyRecord | None: ...
    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[IdempotencyRecord, ...]: ...
    async def compare_and_swap(self, scope: str, key_hash: str, *, tenant_id: str, expected_status: IdempotencyStatus, next_record: IdempotencyRecord) -> IdempotencyRecord: ...


class ResultRepository(RuntimeRepository, Protocol):
    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult: ...
    async def get(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None: ...


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
    async def decide(self, approval_id: str, *, tenant_id: str, expected_status: ApprovalStatus, decision_id: str, decision: ApprovalDecision, principal_id: str, decision_digest: str, decided_at: datetime) -> ApprovalRecord: ...
    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]: ...


class ExternalResultRepository(RuntimeRepository, Protocol):
    async def get_header(self, call_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create_call(self, record: ExternalResultRecord) -> ExternalResultRecord: ...
    async def get(self, call_id: str, *, tenant_id: str) -> ExternalResultRecord | None: ...
    async def supply(self, call_id: str, *, tenant_id: str, expected_status: ExternalCallStatus, result_id: str, payload_ref: str, payload_digest: str, supplied_at: datetime) -> ExternalResultRecord: ...
    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalResultRecord, ...]: ...


class OperationLedgerRepository(RuntimeRepository, Protocol):
    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord: ...
    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None: ...
    async def compare_and_swap(self, operation_id: str, *, tenant_id: str, expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord: ...
    async def list_pending(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, limit: int) -> tuple[OperationLedgerRecord, ...]: ...
    async def compact_terminal(self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str, through_sequence: int) -> str: ...


class TaskRepository(RuntimeRepository, Protocol):
    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create_plan(self, graph: TaskGraph, *, tenant_id: str) -> "TaskGraphView": ...
    async def get_plan(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView | None": ...
    async def reconcile_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView": ...
    async def claim(self, graph_id: str, task_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease: ...
    async def renew(self, lease: TaskLease, *, tenant_id: str) -> TaskLease: ...
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
    async def list(self, *, tenant_id: str, owner_id: str, cursor: str | None, limit: int) -> Page[MemoryRecord]: ...
    async def delete(self, memory_id: str, *, tenant_id: str, expected_revision: int) -> None: ...
    async def delete_with_operation(self, memory_id: str, *, tenant_id: str, expected_revision: int | None, operation: OperationLedgerInput | None) -> "tuple[bool, bool]": ...


class ArtifactRepository(RuntimeRepository, Protocol):
    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord: ...
    async def get_header(self, artifact_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None: ...
    async def list_by_execution(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[ArtifactRecord]: ...


class BlobStore(RuntimeRepository, Protocol):
    async def put_bytes(self, *, tenant_id: str, data: bytes, expected_digest: str | None = None) -> BlobRef: ...
    async def put_stream(self, *, tenant_id: str, chunks: AsyncIterator[bytes], expected_size: int, expected_digest: str) -> BlobRef: ...
    async def stat(self, ref: BlobRef, *, tenant_id: str) -> BlobRef | None: ...
    def open(self, ref: BlobRef, *, tenant_id: str) -> AsyncIterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class RuntimePersistence:
    mode: RuntimePersistenceMode
    backend: RuntimeBackend
    namespace: str
    sessions: SessionRepository
    executions: ExecutionRepository
    results: ResultRepository
    idempotency: IdempotencyRepository
    events: EventRepository
    tasks: TaskRepository
    evaluations: EvaluationRepository
    memories: MemoryRepository
    artifacts: ArtifactRepository
    approvals: ApprovalRepository
    externals: ExternalResultRepository
    operations: OperationLedgerRepository
    tools: ToolStateStore
    blobs: BlobStore
    local_tenant_id: "str | None" = None

    @property
    def atomic_domain_id(self) -> str:
        return self.sessions.atomic_domain_id

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("runtime persistence namespace is required")
        if self.mode is RuntimePersistenceMode.FILESYSTEM and not self.local_tenant_id:
            raise ValueError("filesystem runtime requires local_tenant_id")
        components = (
            self.sessions, self.executions, self.results, self.idempotency, self.events,
            self.tasks, self.evaluations, self.memories, self.artifacts,
            self.approvals, self.externals, self.operations, self.tools, self.blobs,
        )
        if any(component is None for component in components):
            raise ValueError("runtime persistence requires every repository")
        identities = {(component.mode, component.backend, component.namespace, component.atomic_domain_id) for component in components}
        identity = next(iter(identities), None)
        if identity != (self.mode, self.backend, self.namespace, self.atomic_domain_id) or len(identities) != 1:
            raise ValueError("runtime persistence components must share one atomic domain")


__all__ = [
    "ApprovalRecord", "ApprovalRepository", "ArtifactRecord", "ArtifactRepository", "BlobRef", "BlobStatus",
    "BlobStore", "EvaluationRecord", "EvaluationRepository", "ExecutionEventRecord", "ExecutionRecord",
    "ExecutionRepository", "ExecutionStartClaim", "ExecutionStartReservation", "ExecutionStartReservationResult", "ExecutionStartUnknownCommit", "ExecutionCancelRequestCommit", "ExecutionTerminalCommit", "ExecutionTerminalCommitResult", "ExternalResultRecord", "IdempotencyTerminalUpdate", "OperationTerminalUpdate", "SessionHeadAdvance",
    "ExternalResultRepository", "IdempotencyRecord", "IdempotencyRepository", "MemoryRecord", "MemoryRepository",
    "OperationLedgerInput", "OperationLedgerRecord", "OperationLedgerRepository", "ResultRecord", "ResultRepository", "RuntimeBackend", "RuntimePersistence",
    "RuntimePersistenceMode", "RuntimeRepository", "SessionRecord", "SessionRepository", "backend_mode",
    "TaskLease", "TaskNodeView", "TaskRepository", "ToolOperationStatus",
]
