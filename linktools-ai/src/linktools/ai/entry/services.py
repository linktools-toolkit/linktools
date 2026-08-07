#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唯一 Composition Root for process-local Runtime services."""

from dataclasses import dataclass
from typing import Protocol

from linktools.core import environ

from ..asset import AssetCodecRegistry, AssetStore
from ..capability import MCPToolProvider, Sandbox, SkillProvider, SubagentProvider, ToolPolicy, ToolStateStore
from ..model import ModelRegistry, ModelResolver
from ..observe import MiddlewarePipeline
from ..runtime import (
    DefaultApprovalService,
    DefaultArtifactService,
    DefaultEventService,
    DefaultEvaluationService,
    DefaultExecutionService,
    DefaultSessionService,
    DefaultTaskService,
    Runtime,
    RuntimeAccess,
    RuntimeServices,
    build_runtime_access,
)
from ..runtime.persistence import RuntimePersistence
from ..runtime.services import ExecutionHandle, ExecutionRequest, WorkflowGateway, new_runtime_service_identity
from ..core import AuthorizationPolicy, ExecutionProfile, HmacCursorSigner, PrincipalProvider
from ..core.errors import ErrorCode, LinktoolsAIError
from ..spec import AgentSpecCodec, PromptSpecCodec
from ..capability.codec import MCPServerSpecCodec, SkillSpecCodec
from ..spec.output import OutputTypeRegistry

_logger = environ.get_logger("ai.entry.services")

@dataclass(frozen=True, slots=True)
class EntryServices:
    runtime_services: RuntimeServices
    access: RuntimeAccess
    runtime_factory: "EntryRuntimeFactory"
    principal_provider: "PrincipalProvider | None" = None


class EntryRuntimeFactory(Protocol):
    async def build_for_request(self, request: ExecutionRequest) -> Runtime: ...


class _WorkflowExecutionLauncher:
    def __init__(self, gateway: WorkflowGateway) -> None:
        self._gateway = gateway

    async def start(self, binding: "AgentBinding", request: ExecutionRequest, execution: "ExecutionRecord") -> None:
        await self._gateway.start_execution(execution.execution_id, request)

    async def cancel(self, execution: "ExecutionRecord") -> None:
        await self._gateway.cancel_execution(execution.execution_id)


@dataclass(frozen=True, slots=True)
class AgentServices:
    asset_store: AssetStore
    model_registry: ModelRegistry
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
    runtime_services: RuntimeServices
    runtime_access: RuntimeAccess


def build_asset_codecs() -> AssetCodecRegistry:
    registry = AssetCodecRegistry()
    registry.register(AgentSpecCodec())
    registry.register(PromptSpecCodec())
    registry.register(SkillSpecCodec())
    registry.register(MCPServerSpecCodec())
    manifest = registry.freeze()
    _logger.info("asset codecs frozen: entries=%s digest=%s", len(manifest.entries), manifest.digest)
    return registry


def build_agent_services(
    asset_store: AssetStore,
    model_registry: ModelRegistry,
    model_resolver: ModelResolver,
    skill_provider: SkillProvider,
    mcp_provider: MCPToolProvider,
    subagent_provider: SubagentProvider,
    middleware: MiddlewarePipeline,
    sandbox: Sandbox,
    tool_policy: ToolPolicy,
    output_types: OutputTypeRegistry,
    tool_state: ToolStateStore,
    principal_provider: PrincipalProvider,
    runtime_services: RuntimeServices,
) -> AgentServices:
    if any(
        value is None
        for value in (
            asset_store,
            model_registry,
            model_resolver,
            skill_provider,
            mcp_provider,
            subagent_provider,
            middleware,
            sandbox,
            tool_policy,
            output_types,
            tool_state,
            principal_provider,
            runtime_services,
        )
    ):
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    registry_snapshot = model_registry.snapshot()
    resolver_snapshot = model_resolver.snapshot()
    if (
        resolver_snapshot.revision != registry_snapshot.revision
        or resolver_snapshot.digest != registry_snapshot.digest
    ):
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not output_types.frozen:
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not asset_store.codec_manifest.entries or not asset_store.codec_manifest.digest:
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not skill_provider.manifest() or not mcp_provider.manifest() or not subagent_provider.manifest():
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    services = AgentServices(
        asset_store,
        model_registry,
        model_resolver,
        skill_provider,
        mcp_provider,
        subagent_provider,
        middleware,
        sandbox,
        tool_policy,
        output_types,
        tool_state,
        principal_provider,
        runtime_services,
        build_runtime_access(runtime_services),
    )
    _logger.info("agent services composed: model_revision=%s", model_registry.snapshot().revision)
    return services


def build_services(
    services: RuntimeServices,
    runtime_factory: EntryRuntimeFactory,
    principal_provider: "PrincipalProvider | None" = None,
) -> EntryServices:
    if runtime_factory is None:
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return EntryServices(services, build_runtime_access(services), runtime_factory, principal_provider)


def build_default_runtime_services(
    persistence: RuntimePersistence,
    authorization: "AuthorizationPolicy",
    *,
    profile: "ExecutionProfile",
    temporal_enabled: bool,
    grant_key: bytes,
    schema_digest: str = "runtime",
    workflow_gateway: "WorkflowGateway | None" = None,
) -> RuntimeServices:
    """Compose all default services from one persistence and authorization root."""
    identity = new_runtime_service_identity(
        mode=persistence.mode.value,
        namespace=persistence.namespace,
        atomic_domain_id=persistence.atomic_domain_id,
        schema_digest=schema_digest,
        profile=profile,
        temporal_enabled=temporal_enabled,
    )
    if profile is ExecutionProfile.PRODUCTION_SERVICE and workflow_gateway is None:
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    launcher = None if workflow_gateway is None else _WorkflowExecutionLauncher(workflow_gateway)
    execution = DefaultExecutionService(persistence, authorization, launcher=launcher)
    services = RuntimeServices(
        identity,
        execution,
        DefaultSessionService(persistence, authorization, execution, HmacCursorSigner("session", grant_key)),
        DefaultTaskService(persistence, authorization, workflow_gateway),
        DefaultEvaluationService(persistence, authorization, execution),
        DefaultApprovalService(persistence, authorization, workflow_gateway),
        DefaultEventService(persistence, authorization),
        DefaultArtifactService(persistence, authorization, grant_key=grant_key, cursor_signer=HmacCursorSigner("artifact", grant_key)),
    )
    _logger.info("runtime services composed: profile=%s mode=%s namespace=%s", profile, persistence.mode, persistence.namespace)
    return services


__all__ = ["AgentServices", "EntryRuntimeFactory", "EntryServices", "build_agent_services", "build_asset_codecs", "build_default_runtime_services", "build_services"]
