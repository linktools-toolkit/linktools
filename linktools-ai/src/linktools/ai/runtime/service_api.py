#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime service protocols and transport-neutral request values."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    EvaluationStatus,
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionStatus,
    JsonValue,
    Page,
    Principal,
    SessionStatus,
    UsageMetrics,
    validate_idempotency_key,
    validate_memory_scope,
    validate_resource_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..observe import RunSnapshot
from ..storage import ObjectRef
from ..task import CancelGraphRequest, TaskGraphHandle, TaskGraphRequest, TaskGraphResult, TaskGraphView


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    user_prompt: str
    principal: Principal
    idempotency_key: "str | None" = None
    memory_scope: "str | None" = None
    planning: bool = False
    thinking: bool = False

    def __post_init__(self) -> None:
        validate_user_prompt(self.user_prompt)
        if self.idempotency_key is None:
            raise AIError(ErrorCode.IDEMPOTENCY_KEY_INVALID)
        validate_idempotency_key(self.idempotency_key)
        if self.memory_scope is not None:
            validate_memory_scope(self.memory_scope)
        if not isinstance(self.planning, bool) or not isinstance(self.thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class RetryExecutionRequest:
    user_prompt: str
    principal: Principal
    idempotency_key: str

    def __post_init__(self) -> None:
        validate_user_prompt(self.user_prompt)
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ForkExecutionRequest:
    user_prompt: str
    principal: Principal
    idempotency_key: str

    def __post_init__(self) -> None:
        validate_user_prompt(self.user_prompt)
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CancelExecutionRequest:
    principal: Principal
    idempotency_key: str
    force: bool = False

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CancelExecutionResult:
    execution_id: str
    cancelled: bool


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    execution_id: str


@dataclass(frozen=True, slots=True)
class ExecutionView:
    execution_id: str
    status: ExecutionStatus


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    execution_id: str
    status: ExecutionStatus
    output: JsonValue | None
    output_schema_id: str
    output_schema_revision: int
    output_schema_fingerprint: str
    usage: UsageMetrics


@dataclass(frozen=True, slots=True)
class ExecutionTraceItem:
    execution_id: str
    sequence: int
    payload: JsonValue


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    execution_id: str
    sequence: int
    text: str


@dataclass(frozen=True, slots=True)
class ExecutionHistoryItem:
    execution_id: str
    sequence: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.item_kind not in {"system", "user", "assistant", "thinking", "tool_call", "tool_result", "retry"}:
            raise ValueError("execution history item is invalid")


@dataclass(frozen=True, slots=True)
class SessionHistoryItem:
    sequence: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.sequence < 1 or self.item_kind not in {
            "system", "user", "assistant", "thinking", "tool_call", "tool_result", "retry"
        }:
            raise ValueError("session history item is invalid")


class ExecutionHistoryReader(Protocol):
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: "str | None",
        limit: int,
    ) -> Page[ExecutionHistoryItem]: ...

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> "Page[ExecutionTraceItem]": ...

    async def transcript(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[TranscriptItem]: ...


class SessionHistoryReader(Protocol):
    async def history(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation_step_run_id: "str | None",
        continuation_history_id: "str | None" = None,
        cursor: "str | None",
        limit: int,
    ) -> "Page[SessionHistoryItem]": ...


@dataclass(frozen=True, slots=True)
class CreateSessionRequest:
    principal: Principal
    session_id: str
    idempotency_key: str
    cwd: "str | None" = None
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ListSessionRequest:
    principal: Principal
    cursor: "str | None" = None
    limit: int = 100


@dataclass(frozen=True, slots=True)
class ResumeSessionRequest:
    principal: Principal
    user_prompt: str
    idempotency_key: str = ""
    memory_scope: "str | None" = None
    planning: bool = False
    thinking: bool = False

    def __post_init__(self) -> None:
        validate_user_prompt(self.user_prompt)
        validate_idempotency_key(self.idempotency_key)
        if self.memory_scope is not None:
            validate_memory_scope(self.memory_scope)
        if not isinstance(self.planning, bool) or not isinstance(self.thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class ForkSessionRequest:
    principal: Principal
    new_session_id: str
    idempotency_key: str = ""
    cwd: "str | None" = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class UpdateSessionRequest:
    principal: Principal
    expected_revision: int
    idempotency_key: str
    metadata: "Mapping[str, JsonValue]"
    cwd: "str | None" = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CloseSessionRequest:
    principal: Principal
    idempotency_key: str
    force: bool = False
    wait_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        if not 1 <= self.wait_timeout_seconds <= 300:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class SessionView:
    session_id: str
    binding_digest: str
    status: SessionStatus
    revision: int = 0
    resource_generation: int = 0
    cwd: "str | None" = None
    active_execution_ids: "tuple[str, ...]" = ()
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)
    history_quality: str = "complete"


@dataclass(frozen=True, slots=True)
class LoadedSession:
    view: SessionView
    active_execution_ids: "tuple[str, ...]"


@dataclass(frozen=True, slots=True)
class RunEvaluationRequest:
    principal: Principal
    dataset_digest: str
    memory_scope: str
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        validate_memory_scope(self.memory_scope)
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CompareEvaluationRequest:
    principal: Principal
    baseline_id: str
    candidate_id: str
    dataset_id: "str | None" = None
    dataset_revision: "int | None" = None
    evaluator_contract_id: "str | None" = None
    evaluator_contract_revision: "int | None" = None
    target_kind: "str | None" = None
    metric_contract_revision: "int | None" = None
    snapshot_digest: "str | None" = None
    artifact_digest: "str | None" = None
    output_schema_fingerprint: "str | None" = None


@dataclass(frozen=True, slots=True)
class ReplayEvaluationRequest:
    principal: Principal
    memory_scope: str
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        validate_memory_scope(self.memory_scope)
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class EvaluationHandle:
    evaluation_id: str


@dataclass(frozen=True, slots=True)
class EvaluationView:
    evaluation_id: str
    status: EvaluationStatus


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    baseline_id: str
    candidate_id: str
    compatible: bool
    pass_rate: float = 0.0
    error_rate: float = 0.0
    refusal_rate: float = 0.0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.baseline_id.strip()
            or not self.candidate_id.strip()
            or min(self.pass_rate, self.error_rate, self.refusal_rate, self.total_cost, self.latency_p50, self.latency_p95) < 0
            or min(self.retry_count, self.input_tokens, self.output_tokens) < 0
        ):
            raise ValueError("evaluation comparison contains invalid metrics")


@dataclass(frozen=True, slots=True)
class ApprovalView:
    approval_id: str
    status: ApprovalStatus


@dataclass(frozen=True, slots=True)
class ApprovalDecisionRequest:
    principal: Principal
    approval_id: str
    idempotency_key: str
    decision: ApprovalDecision

    def __post_init__(self) -> None:
        validate_resource_id(self.approval_id)
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ApprovalDecisionResult:
    approval_id: str
    idempotency_key: str
    decision: ApprovalDecision


@dataclass(frozen=True, slots=True)
class ExternalSupplyRequest:
    principal: Principal
    call_id: str
    idempotency_key: str
    object_ref: "ObjectRef"
    payload_digest: str

    def __post_init__(self) -> None:
        validate_resource_id(self.call_id)
        validate_idempotency_key(self.idempotency_key)
        if len(self.payload_digest) != 64:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class ExternalSupplyResult:
    call_id: str
    idempotency_key: str
    object_ref: "ObjectRef"
    payload_digest: str


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    execution_id: str
    sequence: int
    event_type: ExecutionEventType
    payload: JsonValue


@dataclass(frozen=True, slots=True)
class ExecutionStreamEvent:
    execution_id: str
    durable_sequence: int | None
    event_type: "ExecutionEventType | ExecutionDeltaType"
    payload: JsonValue


@dataclass(frozen=True, slots=True)
class ArtifactView:
    artifact_id: str
    execution_id: str
    size: int


@dataclass(frozen=True, slots=True)
class ArtifactDownload:
    artifact_id: str
    url: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class WorkflowUpdateResult:
    workflow_id: str
    status: str


@dataclass(frozen=True, slots=True)
class WorkflowQueryResult:
    workflow_id: str
    status: str
    payload: "Mapping[str, JsonValue]"


@dataclass(frozen=True, slots=True)
class BudgetReservationRequest:
    execution_id: str
    idempotency_key: str
    amount: int

    def __post_init__(self) -> None:
        if not self.execution_id.strip() or not self.idempotency_key.strip() or self.amount < 1:
            raise ValueError("budget reservation is invalid")


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    amount: int


@dataclass(frozen=True, slots=True)
class BudgetSettlement:
    reservation_id: str
    amount: int


@dataclass(frozen=True, slots=True)
class BudgetSettlementRequest:
    reservation_id: str
    amount: int

    def __post_init__(self) -> None:
        if not self.reservation_id.strip() or self.amount < 0:
            raise ValueError("budget settlement is invalid")


class WorkflowGateway(Protocol):
    async def start_execution(
        self,
        workflow_id: str,
        request: ExecutionRequest,
        *,
        binding_digest: str,
        binding: Mapping[str, JsonValue],
    ) -> ExecutionHandle: ...

    async def update_execution(self, workflow_id: str, operation: str, payload: 'Mapping[str, JsonValue]') -> WorkflowUpdateResult: ...
    async def query_execution(self, workflow_id: str, query: str) -> WorkflowQueryResult: ...
    async def cancel_execution(self, workflow_id: str) -> CancelExecutionResult: ...
    async def start_task_graph(self, workflow_id: str, request: TaskGraphRequest) -> TaskGraphHandle: ...
    async def cancel_task_graph(self, workflow_id: str, idempotency_key: str) -> TaskGraphView: ...


class ExecutionService(Protocol):
    async def run(self, binding_digest: str, request: ExecutionRequest) -> ExecutionHandle: ...
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView: ...
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult: ...
    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> ExecutionResult: ...
    async def run_and_wait(self, binding_digest: str, request: ExecutionRequest, *, timeout_seconds: "float | None" = None) -> ExecutionResult: ...
    async def retry(self, binding_digest: str, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle: ...
    async def fork(self, binding_digest: str, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle: ...
    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult: ...
    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ExecutionTraceItem]': ...
    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TranscriptItem]': ...
    async def history(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ExecutionHistoryItem]': ...


class SessionService(Protocol):
    async def create(self, binding_digest: str, request: CreateSessionRequest) -> SessionView: ...
    async def get(self, session_id: str, *, principal: Principal) -> SessionView: ...
    async def list(self, request: ListSessionRequest) -> 'Page[SessionView]': ...
    async def history(self, session_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[SessionHistoryItem]': ...
    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession: ...
    async def resume(self, binding_digest: str, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...
    async def fork(self, binding_digest: str, session_id: str, request: ForkSessionRequest) -> SessionView: ...
    async def update(self, binding_digest: str, session_id: str, request: UpdateSessionRequest) -> SessionView: ...
    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView: ...


class TaskService(Protocol):
    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult: ...
    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult: ...
    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView: ...
    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult: ...
    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView: ...


class EvaluationService(Protocol):
    async def run(self, binding_digest: str, output_schema_fingerprint: str, request: RunEvaluationRequest) -> EvaluationHandle: ...
    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView: ...
    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison: ...
    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> RunSnapshot: ...
    async def replay(self, binding_digest: str, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle: ...


class ApprovalService(Protocol):
    async def list(self, execution_id: str, *, principal: Principal) -> 'tuple[ApprovalView, ...]': ...
    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult: ...


class ExternalService(Protocol):
    async def supply(self, execution_id: str, request: ExternalSupplyRequest) -> ExternalSupplyResult: ...


class EventService(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> 'Page[ExecutionEvent]': ...

    def stream(self, execution_id: str, *, principal: Principal, after_sequence: int = 0) -> 'AsyncIterator[ExecutionStreamEvent]': ...


class ArtifactService(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ArtifactView]': ...
    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload: ...


class BudgetService(Protocol):
    async def reserve(self, request: BudgetReservationRequest) -> BudgetReservation: ...
    async def settle(self, request: BudgetSettlementRequest) -> BudgetSettlement: ...


__all__ = [
    "ApprovalDecisionRequest", "ApprovalDecisionResult", "ApprovalService", "ApprovalView",
    "ExternalSupplyRequest", "ExternalSupplyResult", "ExternalService",
    "ArtifactDownload", "ArtifactService", "ArtifactView", "BudgetService", "CancelExecutionRequest",
    "BudgetReservation", "BudgetReservationRequest", "BudgetSettlement", "BudgetSettlementRequest",
    "CancelExecutionResult", "CancelGraphRequest", "CloseSessionRequest", "CompareEvaluationRequest",
    "CreateSessionRequest", "EvaluationComparison", "EvaluationHandle", "EvaluationService",
    "EvaluationView", "EventService", "ExecutionEvent", "ExecutionStreamEvent", "ExecutionHandle",
    "ExecutionRequest", "ExecutionResult", "ExecutionService", "ExecutionView",
    "ExecutionHistoryItem", "ExecutionHistoryReader", "SessionHistoryItem", "SessionHistoryReader",
    "ForkExecutionRequest", "ForkSessionRequest", "ListSessionRequest", "LoadedSession", "Page",
    "ReplayEvaluationRequest", "ResumeSessionRequest", "RetryExecutionRequest", "RunEvaluationRequest",
    "SessionService", "SessionView", "TaskService", "ExecutionTraceItem", "TranscriptItem", "UpdateSessionRequest",
    "WorkflowGateway", "WorkflowQueryResult", "WorkflowUpdateResult",
]
