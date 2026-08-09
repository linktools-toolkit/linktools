#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime containers with explicit query/mutation separation."""

from dataclasses import dataclass
from collections.abc import AsyncIterator

from linktools.core import environ

from ..agent import AgentBinding, AgentCatalogView, BindingDependencies, BindingExecutionRegistry, build_binding_plan
from pydantic_ai_harness.step_persistence import StepStore
from ..capability import MCPToolProvider, SkillProvider, ToolPolicy, ToolStateStore, Sandbox
from ..core import Page, Principal, PrincipalProvider
from ..errors import ErrorCode, AIError
from ..model import ModelResolver
from ..observe import MiddlewarePipeline
from ..observe import RunSnapshot
from ..spec import AgentSpec, OutputTypeRegistry, PromptSpec
from ..task import CancelGraphRequest, TaskGraphRequest, TaskGraphResult, TaskGraphView
from ..task import TaskApi, TaskQueryApi
from ..runtime import ApprovalApi, ApprovalQueryApi
from ..runtime import ArtifactApi
from ..runtime import EventApi
from ..runtime import EvaluationApi, EvaluationQueryApi, validate_compare_request
from ..runtime import ExecutionApi, ExecutionQueryApi
from ..runtime import (
    ApprovalDecisionRequest,
    ApprovalDecisionResult,
    ApprovalService,
    ApprovalView,
    ArtifactDownload,
    ArtifactService,
    ArtifactView,
    CancelExecutionRequest,
    CancelExecutionResult,
    CloseSessionRequest,
    CompareEvaluationRequest,
    CreateSessionRequest,
    EvaluationComparison,
    EvaluationHandle,
    EvaluationService,
    EvaluationView,
    EventService,
    ExecutionEvent,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ExecutionStreamItem,
    ExecutionView,
    ForkExecutionRequest,
    ForkSessionRequest,
    ListSessionRequest,
    LoadedSession,
    ReplayEvaluationRequest,
    ResumeSessionRequest,
    RetryExecutionRequest,
    RunEvaluationRequest,
    RuntimeServiceIdentity,
    RuntimeServices,
    SessionService,
    SessionView,
    TaskService,
    TraceItem,
    TranscriptItem,
    UpdateSessionRequest,
)
from ..runtime import SessionApi, SessionQueryApi


@dataclass(frozen=True, slots=True)
class Runtime:
    service_identity: RuntimeServiceIdentity
    binding: AgentBinding
    execution: ExecutionApi
    session: SessionApi
    task: TaskApi
    evaluation: EvaluationApi
    approval: ApprovalApi
    event: EventApi
    artifact: ArtifactApi


@dataclass(frozen=True, slots=True)
class RuntimeAccess:
    service_identity: RuntimeServiceIdentity
    execution: ExecutionQueryApi
    session: SessionQueryApi
    task: TaskQueryApi
    evaluation: EvaluationQueryApi
    approval: ApprovalQueryApi
    event: EventApi
    artifact: ArtifactApi


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    model_resolver: ModelResolver
    skill_provider: SkillProvider
    mcp_provider: MCPToolProvider
    middleware: MiddlewarePipeline
    sandbox: Sandbox
    tool_policy: ToolPolicy
    output_types: OutputTypeRegistry
    tool_state: ToolStateStore
    principal_provider: PrincipalProvider
    services: RuntimeServices
    binding_registry: "BindingExecutionRegistry | None" = None

    @property
    def binding(self) -> BindingDependencies:
        return BindingDependencies(
            self.model_resolver,
            self.skill_provider,
            self.mcp_provider,
            self.middleware,
            self.sandbox,
            self.tool_policy,
            self.output_types,
        )


def build_runtime(
    spec: AgentSpec,
    prompt: PromptSpec,
    *,
    dependencies: RuntimeDependencies,
) -> Runtime:
    if any(
        value is None
        for value in (
            dependencies.model_resolver,
            dependencies.skill_provider,
            dependencies.mcp_provider,
            dependencies.middleware,
            dependencies.sandbox,
            dependencies.tool_policy,
            dependencies.output_types,
            dependencies.tool_state,
            dependencies.principal_provider,
            dependencies.services,
        )
    ):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not spec.id or not prompt.id or spec.revision < 1 or prompt.revision < 1:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "invalid Agent or Prompt revision")
    plan = build_binding_plan(spec, prompt, dependencies=dependencies.binding)
    if dependencies.binding_registry is not None:
        dependencies.binding_registry.register(plan)
    binding = plan.binding
    logger = environ.get_logger("ai.app.facade")
    logger.debug("runtime binding prepared agent=%s model=%s route=%s", spec.id, spec.model, plan.model_route.route_id)
    return Runtime(
        dependencies.services.identity,
        binding,
        _ExecutionApi(dependencies.services.execution, binding),
        _SessionApi(dependencies.services.session, binding),
        _TaskApi(dependencies.services.task, binding),
        _EvaluationApi(dependencies.services.evaluation, binding),
        _ApprovalApi(dependencies.services.approval),
        _EventApi(dependencies.services.event),
        _ArtifactApi(dependencies.services.artifact),
    )


@dataclass(frozen=True, slots=True)
class LocalRuntimeDependencies:
    binding: BindingDependencies
    binding_registry: BindingExecutionRegistry
    agent_catalog: AgentCatalogView
    steps: StepStore
    tool_state: ToolStateStore
    principal_provider: PrincipalProvider
    services: RuntimeServices


def build_local_runtime(
    spec: AgentSpec,
    prompt: PromptSpec,
    *,
    dependencies: LocalRuntimeDependencies,
) -> Runtime:
    plan = build_binding_plan(spec, prompt, dependencies=dependencies.binding)
    dependencies.binding_registry.register(plan)
    binding = plan.binding
    _logger = environ.get_logger("ai.app.facade")
    _logger.info("local runtime binding registered: agent=%s binding=%s", spec.id, binding.digest)
    return Runtime(
        dependencies.services.identity,
        binding,
        _ExecutionApi(dependencies.services.execution, binding),
        _SessionApi(dependencies.services.session, binding),
        _TaskApi(dependencies.services.task, binding),
        _EvaluationApi(dependencies.services.evaluation, binding),
        _ApprovalApi(dependencies.services.approval),
        _EventApi(dependencies.services.event),
        _ArtifactApi(dependencies.services.artifact),
    )


def build_runtime_access(services: RuntimeServices) -> RuntimeAccess:
    if services is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return RuntimeAccess(
        services.identity,
        _ExecutionAccess(services.execution),
        _SessionAccess(services.session),
        _TaskAccess(services.task),
        _EvaluationAccess(services.evaluation),
        _ApprovalAccess(services.approval),
        _EventApi(services.event),
        _ArtifactApi(services.artifact),
    )


class _ExecutionApi(ExecutionApi):
    def __init__(self, service: ExecutionService, binding: AgentBinding) -> None:
        self._service = service
        self._binding = binding

    async def run(self, request: ExecutionRequest) -> ExecutionHandle:
        return await self._service.run(self._binding.digest, request)

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.inspect(execution_id, principal=principal)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        return await self._service.result(execution_id, principal=principal)

    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> ExecutionResult:
        return await self._service.wait(execution_id, principal=principal, timeout_seconds=timeout_seconds)

    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TraceItem]":
        return await self._service.trace(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TranscriptItem]":
        return await self._service.transcript(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def history(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[ExecutionHistoryItem]":
        return await self._service.history(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def run_and_wait(self, request: ExecutionRequest, *, timeout_seconds: "float | None" = None) -> ExecutionResult:
        return await self._service.run_and_wait(self._binding.digest, request, timeout_seconds=timeout_seconds)

    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        return await self._service.retry(self._binding.digest, execution_id, request)

    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        return await self._service.fork(self._binding.digest, execution_id, request)

    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult:
        return await self._service.cancel(execution_id, request)


class _ExecutionAccess(ExecutionQueryApi):
    def __init__(self, service: ExecutionService) -> None:
        self._service = service

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        return await self._service.inspect(execution_id, principal=principal)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        return await self._service.result(execution_id, principal=principal)

    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> ExecutionResult:
        return await self._service.wait(execution_id, principal=principal, timeout_seconds=timeout_seconds)

    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TraceItem]":
        return await self._service.trace(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TranscriptItem]":
        return await self._service.transcript(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def history(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[ExecutionHistoryItem]":
        return await self._service.history(execution_id, principal=principal, cursor=cursor, limit=limit)


class _SessionApi(SessionApi):
    def __init__(self, service: SessionService, binding: AgentBinding) -> None:
        self._service = service
        self._binding = binding

    async def create(self, request: CreateSessionRequest) -> SessionView:
        return await self._service.create(self._binding.digest, request)

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        return await self._service.get(session_id, principal=principal)

    async def list(self, request: ListSessionRequest) -> "Page[SessionView]":
        return await self._service.list(request)

    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession:
        return await self._service.load(session_id, principal=principal)

    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:
        return await self._service.resume(self._binding.digest, session_id, request)

    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView:
        return await self._service.fork(self._binding.digest, session_id, request)

    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView:
        return await self._service.update(self._binding.digest, session_id, request)

    async def close(self, session_id: str, request: CloseSessionRequest) -> SessionView:
        return await self._service.close(session_id, request)


class _SessionAccess(SessionQueryApi):
    def __init__(self, service: SessionService) -> None:
        self._service = service

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        return await self._service.get(session_id, principal=principal)

    async def list(self, request: ListSessionRequest) -> "Page[SessionView]":
        return await self._service.list(request)

    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession:
        return await self._service.load(session_id, principal=principal)


class _TaskApi:
    def __init__(self, service: TaskService, binding: AgentBinding) -> None:
        self._service = service
        self._binding = binding

    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult:
        return await self._service.run_graph(self._binding.digest, request)

    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        return await self._service.run_graph_and_wait(self._binding.digest, request, timeout_seconds=timeout_seconds)

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        return await self._service.inspect_graph(graph_id, principal=principal)

    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        return await self._service.wait_graph(graph_id, principal=principal, timeout_seconds=timeout_seconds)

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        return await self._service.cancel_graph(graph_id, request)


class _TaskAccess(TaskQueryApi):
    def __init__(self, service: TaskService) -> None:
        self._service = service

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        return await self._service.inspect_graph(graph_id, principal=principal)

    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        return await self._service.wait_graph(graph_id, principal=principal, timeout_seconds=timeout_seconds)


class _EvaluationApi(EvaluationApi):
    def __init__(self, service: EvaluationService, binding: AgentBinding) -> None:
        self._service = service
        self._binding = binding

    async def run(self, request: RunEvaluationRequest) -> EvaluationHandle:
        return await self._service.run(self._binding.digest, self._binding.output_schema_fingerprint, request)

    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView:
        return await self._service.inspect(evaluation_id, principal=principal)

    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison:
        validate_compare_request(request)
        return await self._service.compare(request)

    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> "RunSnapshot":
        return await self._service.snapshot(evaluation_id, principal=principal)

    async def replay(self, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle:
        return await self._service.replay(self._binding.digest, snapshot_id, request)


class _EvaluationAccess(EvaluationQueryApi):
    def __init__(self, service: EvaluationService) -> None:
        self._service = service

    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView:
        return await self._service.inspect(evaluation_id, principal=principal)

    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison:
        validate_compare_request(request)
        return await self._service.compare(request)

    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> "RunSnapshot":
        return await self._service.snapshot(evaluation_id, principal=principal)


class _ApprovalApi(ApprovalApi):
    def __init__(self, service: ApprovalService) -> None:
        self._service = service

    async def list(self, execution_id: str, *, principal: Principal) -> 'tuple[ApprovalView, ...]':
        return await self._service.list(execution_id, principal=principal)

    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult:
        return await self._service.decide(execution_id, request)


class _ApprovalAccess(ApprovalQueryApi):
    def __init__(self, service: ApprovalService) -> None:
        self._service = service

    async def list(self, execution_id: str, *, principal: Principal) -> 'tuple[ApprovalView, ...]':
        return await self._service.list(execution_id, principal=principal)


class _EventApi:
    def __init__(self, service: EventService) -> None:
        self._service = service

    async def list(self, execution_id: str, *, principal: Principal, after_sequence: int = 0, limit: int = 100) -> "Page[ExecutionEvent]":
        return await self._service.list(execution_id, principal=principal, after_sequence=after_sequence, limit=limit)

    def stream(self, execution_id: str, *, principal: Principal, after_sequence: int = 0) -> 'AsyncIterator[ExecutionStreamItem]':
        return self._service.stream(execution_id, principal=principal, after_sequence=after_sequence)


class _ArtifactApi:
    def __init__(self, service: ArtifactService) -> None:
        self._service = service

    async def list(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[ArtifactView]":
        return await self._service.list(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def get(self, artifact_id: str, *, principal: Principal) -> ArtifactDownload:
        return await self._service.get(artifact_id, principal=principal)


__all__ = ["LocalRuntimeDependencies", "RuntimeDependencies", "build_local_runtime", "build_runtime", "build_runtime_access"]
