#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime containers with explicit query/mutation separation."""

from dataclasses import dataclass
from collections.abc import AsyncIterator

from linktools.core import environ

from ..agent.binding import AgentBinding
from ..capability import MCPToolProvider, SkillProvider, SubagentProvider, ToolPolicy, ToolStateStore, Sandbox
from ..core import Page, Principal, PrincipalProvider
from ..core.errors import ErrorCode, AIError
from ..core.ids import canonical_sha256
from ..model import ModelResolver
from ..observe.middleware import MiddlewarePipeline
from ..observe.snapshot import RunSnapshot
from ..spec import AgentSpec, OutputTypeRegistry, PromptSpec
from ..task.graph import CancelGraphRequest, TaskGraphRequest, TaskGraphResult, TaskGraphView
from ..task.service import TaskApi, TaskQueryApi
from ..runtime.approval import ApprovalApi, ApprovalQueryApi
from ..runtime.artifact import ArtifactApi
from ..runtime.event import EventApi
from ..runtime.evaluation import EvaluationApi, EvaluationQueryApi, validate_compare_request
from ..runtime.execution import ExecutionApi, ExecutionQueryApi
from ..runtime.services import (
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
from ..runtime.session import SessionApi, SessionQueryApi


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
    subagent_provider: SubagentProvider
    middleware: MiddlewarePipeline
    sandbox: Sandbox
    tool_policy: ToolPolicy
    output_types: OutputTypeRegistry
    tool_state: ToolStateStore
    principal_provider: PrincipalProvider
    services: RuntimeServices


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
            dependencies.subagent_provider,
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
    route = dependencies.model_resolver.resolve(spec.model)
    model_revision = dependencies.model_resolver.snapshot().revision
    spec_fingerprint = canonical_sha256(
        {
            "id": spec.id,
            "revision": spec.revision,
            "model": spec.model,
            "features": [
                {"kind": feature.kind, "id": feature.id, "revision": feature.revision, "required": feature.required, "config": dict(feature.config)}
                for feature in spec.features
            ],
            "output_schema": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "instructions": list(spec.instructions),
        }
    )
    prompt_fingerprint = canonical_sha256(
        {"id": prompt.id, "revision": prompt.revision, "system": prompt.system, "instructions": list(prompt.instructions), "variables": list(prompt.variables)}
    )
    output_schema_fingerprint = dependencies.output_types.fingerprint(spec.output_schema, spec.output_schema_revision)
    capability_manifest_digest = _capability_digest(spec, dependencies)
    binding = AgentBinding(
        spec,
        prompt,
        spec_fingerprint,
        prompt_fingerprint,
        model_revision,
        output_schema_fingerprint,
        capability_manifest_digest,
        dependencies.tool_policy.fingerprint,
        dependencies.sandbox.fingerprint,
        dependencies.middleware.fingerprint,
    )
    logger = environ.get_logger("ai.app.facade")
    logger.debug("runtime binding prepared agent=%s model=%s route=%s", spec.id, spec.model, route.route_id)
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


def _capability_digest(spec: AgentSpec, dependencies: RuntimeDependencies) -> str:
    manifests: list[dict[str, str | int | bool]] = []
    for feature in spec.features:
        provider_digest = ""
        try:
            if feature.kind == "skill":
                resolved = dependencies.skill_provider.resolve_ref(feature.id, feature.revision)
                provider_digest = dependencies.skill_provider.manifest()
                resolved_revision = resolved.revision
                fingerprint = canonical_sha256({"id": resolved.id, "revision": resolved.revision, "content": resolved.content})
            elif feature.kind == "mcp":
                resolved = dependencies.mcp_provider.resolve_ref(feature.id, feature.revision)
                provider_digest = dependencies.mcp_provider.manifest()
                resolved_revision = resolved.revision
                fingerprint = canonical_sha256({"id": resolved.id, "revision": resolved.revision, "command": resolved.command, "args": list(resolved.args)})
            elif feature.kind == "subagent":
                resolved = dependencies.subagent_provider.resolve_ref(feature.id, feature.revision)
                provider_digest = dependencies.subagent_provider.manifest()
                resolved_revision = feature.revision or 1
                fingerprint = canonical_sha256(resolved)
            elif feature.kind in {"tool", "sandbox", "middleware"}:
                resolved_revision = feature.revision or 1
                fingerprint = canonical_sha256({"kind": feature.kind, "id": feature.id})
            else:
                raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING if feature.required else ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        except AIError as error:
            if feature.required:
                raise AIError(ErrorCode.FEATURE_REQUIRED_MISSING) from error
            resolved_revision = 0
            fingerprint = "UNRESOLVED"
            provider_digest = "UNRESOLVED"
        manifests.append({"kind": feature.kind, "id": feature.id, "requested_revision": feature.revision or 0, "resolved_revision": resolved_revision, "config_digest": canonical_sha256(dict(feature.config)), "fingerprint": fingerprint, "provider_manifest_digest": provider_digest, "required": feature.required})
    return canonical_sha256({"features": sorted(manifests, key=lambda value: (str(value["kind"]), str(value["id"]), int(value["resolved_revision"])))})


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

    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TraceItem]":
        return await self._service.trace(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TranscriptItem]":
        return await self._service.transcript(execution_id, principal=principal, cursor=cursor, limit=limit)

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

    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TraceItem]":
        return await self._service.trace(execution_id, principal=principal, cursor=cursor, limit=limit)

    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> "Page[TranscriptItem]":
        return await self._service.transcript(execution_id, principal=principal, cursor=cursor, limit=limit)


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

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        return await self._service.inspect_graph(graph_id, principal=principal)

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        return await self._service.cancel_graph(graph_id, request)


class _TaskAccess(TaskQueryApi):
    def __init__(self, service: TaskService) -> None:
        self._service = service

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        return await self._service.inspect_graph(graph_id, principal=principal)


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


__all__ = ["RuntimeDependencies", "build_runtime", "build_runtime_access"]
