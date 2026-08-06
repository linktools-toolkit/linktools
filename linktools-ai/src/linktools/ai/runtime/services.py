#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime service protocols and transport-neutral request values."""

from dataclasses import dataclass
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from ..core import ExecutionProfile, Page, Principal
from ..core.json import JsonValue
from ..observe.snapshot import RunSnapshot
from ..task.model import (
    CancelGraphRequest,
    TaskGraphRequest,
    TaskGraphHandle,
    TaskGraphResult,
    TaskGraphView,
)
from ..agent.context import AgentBinding


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    prompt: str
    principal: Principal
    requested_profile: ExecutionProfile = ExecutionProfile.LOCAL_CODING
    idempotency_key: "str | None" = None


@dataclass(frozen=True, slots=True)
class RetryExecutionRequest:
    principal: Principal
    idempotency_key: "str | None" = None


@dataclass(frozen=True, slots=True)
class ForkExecutionRequest:
    principal: Principal
    idempotency_key: "str | None" = None


@dataclass(frozen=True, slots=True)
class CancelExecutionRequest:
    principal: Principal
    force: bool = False


@dataclass(frozen=True, slots=True)
class CancelExecutionResult:
    execution_id: str
    cancelled: bool


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    execution_id: str
    profile: ExecutionProfile


@dataclass(frozen=True, slots=True)
class ExecutionView:
    execution_id: str
    status: str
    profile: ExecutionProfile


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    execution_id: str
    status: str
    output: str


@dataclass(frozen=True, slots=True)
class TraceItem:
    execution_id: str
    sequence: int
    payload: JsonValue


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    execution_id: str
    sequence: int
    text: str


@dataclass(frozen=True, slots=True)
class CreateSessionRequest:
    principal: Principal
    session_id: str


@dataclass(frozen=True, slots=True)
class ListSessionRequest:
    principal: Principal
    cursor: "str | None" = None
    limit: int = 100


@dataclass(frozen=True, slots=True)
class ResumeSessionRequest:
    principal: Principal
    prompt: str


@dataclass(frozen=True, slots=True)
class ForkSessionRequest:
    principal: Principal
    new_session_id: str


@dataclass(frozen=True, slots=True)
class UpdateSessionRequest:
    principal: Principal
    metadata: "Mapping[str, JsonValue]"


@dataclass(frozen=True, slots=True)
class CloseSessionRequest:
    principal: Principal
    force: bool = False


@dataclass(frozen=True, slots=True)
class SessionView:
    session_id: str
    binding_digest: str
    status: str


@dataclass(frozen=True, slots=True)
class LoadedSession:
    view: SessionView
    active_execution_ids: "tuple[str, ...]"


@dataclass(frozen=True, slots=True)
class RunEvaluationRequest:
    principal: Principal
    dataset_digest: str


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


@dataclass(frozen=True, slots=True)
class EvaluationHandle:
    evaluation_id: str


@dataclass(frozen=True, slots=True)
class EvaluationView:
    evaluation_id: str
    status: str


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
            or min(
                self.pass_rate,
                self.error_rate,
                self.refusal_rate,
                self.total_cost,
                self.latency_p50,
                self.latency_p95,
            ) < 0
            or min(self.retry_count, self.input_tokens, self.output_tokens) < 0
        ):
            raise ValueError("evaluation comparison contains invalid metrics")


@dataclass(frozen=True, slots=True)
class ApprovalView:
    approval_id: str
    status: str


@dataclass(frozen=True, slots=True)
class ApprovalDecisionRequest:
    principal: Principal
    approval_id: str
    decision_id: str
    decision: str


@dataclass(frozen=True, slots=True)
class ApprovalDecisionResult:
    approval_id: str
    decision_id: str
    decision: str


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    execution_id: str
    sequence: int
    event_type: str
    payload: JsonValue


@dataclass(frozen=True, slots=True)
class ExecutionStreamItem:
    event: ExecutionEvent
    live: bool


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
class PayloadRef:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("payload reference must not be empty")


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
    async def start_execution(self, workflow_id: str, request: ExecutionRequest) -> ExecutionHandle: ...
    async def update_execution(self, workflow_id: str, operation: str, payload: 'Mapping[str, JsonValue]') -> WorkflowUpdateResult: ...
    async def query_execution(self, workflow_id: str, query: str) -> WorkflowQueryResult: ...
    async def cancel_execution(self, workflow_id: str) -> CancelExecutionResult: ...
    async def start_task_graph(self, workflow_id: str, request: TaskGraphRequest) -> TaskGraphHandle: ...


class ExecutionService(Protocol):
    async def run(self, binding: AgentBinding, request: ExecutionRequest) -> ExecutionHandle: ...
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView: ...
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult: ...
    async def retry(self, binding: AgentBinding, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle: ...
    async def fork(self, binding: AgentBinding, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle: ...
    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult: ...
    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TraceItem]': ...
    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TranscriptItem]': ...


class SessionService(Protocol):
    async def create(self, binding: AgentBinding, request: CreateSessionRequest) -> SessionView: ...
    async def get(self, session_id: str, *, principal: Principal) -> SessionView: ...
    async def list(self, request: ListSessionRequest) -> 'Page[SessionView]': ...
    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession: ...
    async def resume(self, binding: AgentBinding, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle: ...
    async def fork(self, binding: AgentBinding, session_id: str, request: ForkSessionRequest) -> SessionView: ...
    async def update(self, binding: AgentBinding, session_id: str, request: UpdateSessionRequest) -> SessionView: ...
    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView: ...


class TaskService(Protocol):
    async def run_graph(self, binding: AgentBinding, request: TaskGraphRequest) -> TaskGraphResult: ...
    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView: ...
    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView: ...


class EvaluationService(Protocol):
    async def run(self, binding: AgentBinding, request: RunEvaluationRequest) -> EvaluationHandle: ...
    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView: ...
    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison: ...
    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> RunSnapshot: ...
    async def replay(self, binding: AgentBinding, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle: ...


class ApprovalService(Protocol):
    async def list(self, execution_id: str, *, principal: Principal) -> 'tuple[ApprovalView, ...]': ...
    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult: ...


class EventService(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> 'Page[ExecutionEvent]': ...
    def stream(self, execution_id: str, *, principal: Principal, after_sequence: int = 0) -> 'AsyncIterator[ExecutionStreamItem]': ...


class ArtifactService(Protocol):
    async def list(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[ArtifactView]': ...
    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload: ...


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    execution: ExecutionService
    session: SessionService
    task: TaskService
    evaluation: EvaluationService
    approval: ApprovalService
    event: EventService
    artifact: ArtifactService

    def __post_init__(self) -> None:
        if any(
            service is None
            for service in (
                self.execution,
                self.session,
                self.task,
                self.evaluation,
                self.approval,
                self.event,
                self.artifact,
            )
        ):
            raise ValueError("RuntimeServices requires all seven services")


class PayloadService(Protocol):
    async def load(self, ref: PayloadRef) -> bytes: ...
    async def store(self, data: bytes) -> PayloadRef: ...


class EventRepository(Protocol):
    async def append(self, execution_id: str, event: ExecutionEvent, *, expected_sequence: int) -> ExecutionEvent: ...


class ResultRepository(Protocol):
    async def commit(self, execution_id: str, result: ExecutionResult, *, idempotency_key: str) -> str: ...


class BudgetService(Protocol):
    async def reserve(self, request: BudgetReservationRequest) -> BudgetReservation: ...
    async def settle(self, request: BudgetSettlementRequest) -> BudgetSettlement: ...


__all__ = [
    "ApprovalDecisionRequest", "ApprovalDecisionResult", "ApprovalService", "ApprovalView",
    "ArtifactDownload", "ArtifactService", "ArtifactView", "BudgetService", "CancelExecutionRequest",
    "BudgetReservation", "BudgetReservationRequest", "BudgetSettlement", "BudgetSettlementRequest",
    "CancelExecutionResult",
    "CancelGraphRequest", "CloseSessionRequest", "CompareEvaluationRequest",
    "CreateSessionRequest", "EvaluationComparison", "EvaluationHandle", "EvaluationService",
    "EvaluationView", "EventRepository", "EventService", "ExecutionEvent", "ExecutionHandle",
    "ExecutionRequest", "ExecutionResult", "ExecutionService", "ExecutionStreamItem", "ExecutionView",
    "ForkExecutionRequest", "ForkSessionRequest", "ListSessionRequest", "LoadedSession", "Page",
    "PayloadRef", "PayloadService", "ReplayEvaluationRequest", "ResultRepository", "ResumeSessionRequest",
    "RetryExecutionRequest", "RunEvaluationRequest", "RuntimeServices", "SessionService", "SessionView",
    "TaskService", "TraceItem", "TranscriptItem", "UpdateSessionRequest",
    "WorkflowGateway", "WorkflowQueryResult", "WorkflowUpdateResult",
]
