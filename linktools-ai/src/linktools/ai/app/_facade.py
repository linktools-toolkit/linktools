#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime containers with explicit query/mutation separation."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from linktools.core import environ

from ..agent import (
    AgentBinder,
    AgentBinding,
    AgentBindingRegistry,
    OutputTypeRegistry,
)
from ..capability import (
    CapabilityInjection,
    CapabilityResolver,
    CapabilityResolverRegistry,
)
from ..core import Page, Principal, canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelConnectionRegistry, ModelResolver
from ..observe import MiddlewarePipeline, RunSnapshot
from ..runtime import (
    ApprovalApi,
    ApprovalDecisionRequest,
    ApprovalDecisionResult,
    ApprovalQueryApi,
    ApprovalService,
    ApprovalView,
    ArtifactApi,
    ArtifactDownload,
    ArtifactService,
    ArtifactView,
    CancelExecutionRequest,
    CancelExecutionResult,
    CloseSessionRequest,
    CompareEvaluationRequest,
    CreateSessionRequest,
    EvaluationApi,
    EvaluationComparison,
    EvaluationHandle,
    EvaluationQueryApi,
    EvaluationService,
    EvaluationView,
    EventApi,
    EventService,
    ExecutionApi,
    ExecutionEvent,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionQueryApi,
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
    SessionApi,
    SessionQueryApi,
    SessionService,
    SessionView,
    TaskService,
    ToolPolicy,
    TraceItem,
    TranscriptItem,
    UpdateSessionRequest,
    validate_compare_request,
)
from ..spec import AgentSpec, PromptSpec
from ..task import (
    CancelGraphRequest,
    TaskApi,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphView,
)
from ..workspace import Sandbox


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

    async def run(self, request: ExecutionRequest, *, timeout_seconds: "float | None" = None) -> ExecutionResult:
        return await self.execution.run_and_wait(request, timeout_seconds=timeout_seconds)


@dataclass(frozen=True, slots=True)
class RuntimeAccess:
    service_identity: RuntimeServiceIdentity
    execution: ExecutionQueryApi
    session: SessionQueryApi
    task: TaskApi
    evaluation: EvaluationQueryApi
    approval: ApprovalQueryApi
    event: EventApi
    artifact: ArtifactApi


@dataclass(frozen=True, slots=True)
class LocalRuntimeServices:
    services: RuntimeServices
    binding_registry: "AgentBindingRegistry"

    def __post_init__(self) -> None:
        if self.services is None or self.binding_registry is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    model_resolver: ModelResolver
    capability_resolvers: "tuple[CapabilityResolver, ...]"
    model_connections: ModelConnectionRegistry
    middleware: MiddlewarePipeline
    sandbox: Sandbox
    tool_policy: ToolPolicy
    output_types: OutputTypeRegistry
    local: LocalRuntimeServices
    platform_capability_profile_fingerprint: str = canonical_sha256("linktools-ai-platform-v1")

    @property
    def execution_profile_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "tool_policy": self.tool_policy.fingerprint,
                "sandbox": self.sandbox.fingerprint,
                "middleware": self.middleware.fingerprint,
                "platform_capabilities": self.platform_capability_profile_fingerprint,
            }
        )


def _build_bound_runtime(binding: AgentBinding, local: LocalRuntimeServices) -> Runtime:
    return Runtime(
        local.services.identity,
        binding,
        _ExecutionApi(local.services.execution, binding),
        _SessionApi(local.services.session, binding),
        _TaskApi(local.services.task),
        _EvaluationApi(local.services.evaluation, binding),
        _ApprovalApi(local.services.approval),
        _EventApi(local.services.event),
        _ArtifactApi(local.services.artifact),
    )


def build_runtime(
    spec: AgentSpec,
    prompt: PromptSpec,
    *,
    dependencies: RuntimeDependencies,
    additional_capability_resolvers: "Sequence[CapabilityResolver]" = (),
    capability_injections: "Sequence[CapabilityInjection]" = (),
) -> Runtime:
    if any(
        value is None
        for value in (
            dependencies.model_resolver,
            dependencies.capability_resolvers,
            dependencies.model_connections,
            dependencies.middleware,
            dependencies.sandbox,
            dependencies.tool_policy,
            dependencies.output_types,
            dependencies.local,
            dependencies.execution_profile_fingerprint,
        )
    ):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not spec.id or not prompt.id or spec.revision < 1 or prompt.revision < 1:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "invalid Agent or Prompt revision")
    resolver_registry = CapabilityResolverRegistry((*dependencies.capability_resolvers, *additional_capability_resolvers))
    binder = AgentBinder(
        model_resolver=dependencies.model_resolver,
        model_connections=dependencies.model_connections,
        output_types=dependencies.output_types,
        capability_resolvers=resolver_registry,
        execution_profile_fingerprint=dependencies.execution_profile_fingerprint,
    )
    binding = binder.bind(spec, prompt, injections=tuple(capability_injections))
    dependencies.local.binding_registry.register(binding)
    binding = dependencies.local.binding_registry.resolve(binding.manifest.digest)
    logger = environ.get_logger("ai.app.facade")
    logger.debug("runtime binding prepared agent=%s model=%s route=%s", spec.id, spec.model, binding.model_route.route_id)
    return _build_bound_runtime(binding, dependencies.local)


def build_local_runtime(
    spec: AgentSpec,
    prompt: PromptSpec,
    *,
    binder: AgentBinder,
    local: LocalRuntimeServices,
) -> Runtime:
    binding = binder.bind(spec, prompt)
    local.binding_registry.register(binding)
    binding = local.binding_registry.resolve(binding.manifest.digest)
    _logger = environ.get_logger("ai.app.facade")
    _logger.info("local runtime binding registered: agent=%s binding=%s", spec.id, binding.manifest.digest)
    return _build_bound_runtime(binding, local)


def build_runtime_access(services: RuntimeServices) -> RuntimeAccess:
    if services is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return RuntimeAccess(
        services.identity,
        _ExecutionAccess(services.execution),
        _SessionAccess(services.session),
        _TaskApi(services.task),
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
        return await self._service.run(self._binding.manifest.digest, request)

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
        return await self._service.run_and_wait(self._binding.manifest.digest, request, timeout_seconds=timeout_seconds)

    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle:
        return await self._service.retry(self._binding.manifest.digest, execution_id, request)

    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle:
        return await self._service.fork(self._binding.manifest.digest, execution_id, request)

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
        return await self._service.create(self._binding.manifest.digest, request)

    async def get(self, session_id: str, *, principal: Principal) -> SessionView:
        return await self._service.get(session_id, principal=principal)

    async def list(self, request: ListSessionRequest) -> "Page[SessionView]":
        return await self._service.list(request)

    async def load(self, session_id: str, *, principal: Principal) -> LoadedSession:
        return await self._service.load(session_id, principal=principal)

    async def resume(self, session_id: str, request: ResumeSessionRequest) -> ExecutionHandle:
        return await self._service.resume(self._binding.manifest.digest, session_id, request)

    async def fork(self, session_id: str, request: ForkSessionRequest) -> SessionView:
        return await self._service.fork(self._binding.manifest.digest, session_id, request)

    async def update(self, session_id: str, request: UpdateSessionRequest) -> SessionView:
        return await self._service.update(self._binding.manifest.digest, session_id, request)

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
    def __init__(self, service: TaskService) -> None:
        self._service = service

    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult:
        return await self._service.run_graph(request)

    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        return await self._service.run_graph_and_wait(request, timeout_seconds=timeout_seconds)

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        return await self._service.inspect_graph(graph_id, principal=principal)

    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        return await self._service.wait_graph(graph_id, principal=principal, timeout_seconds=timeout_seconds)

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        return await self._service.cancel_graph(graph_id, request)


class _EvaluationApi(EvaluationApi):
    def __init__(self, service: EvaluationService, binding: AgentBinding) -> None:
        self._service = service
        self._binding = binding

    async def run(self, request: RunEvaluationRequest) -> EvaluationHandle:
        return await self._service.run(self._binding.manifest.digest, self._binding.manifest.output_schema_fingerprint, request)

    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView:
        return await self._service.inspect(evaluation_id, principal=principal)

    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison:
        validate_compare_request(request)
        return await self._service.compare(request)

    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> "RunSnapshot":
        return await self._service.snapshot(evaluation_id, principal=principal)

    async def replay(self, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle:
        return await self._service.replay(self._binding.manifest.digest, snapshot_id, request)


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


__all__ = ["LocalRuntimeServices", "RuntimeDependencies", "build_local_runtime", "build_runtime", "build_runtime_access"]
