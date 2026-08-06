#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唯一 Composition Root for process-local Runtime services."""

from dataclasses import dataclass

from linktools.core import environ

from ..asset import AssetCodecRegistry, AssetStore
from ..capability import MCPToolProvider, Sandbox, SkillProvider, SubagentProvider, ToolPolicy, ToolStateStore
from ..model import ModelRegistry, ModelResolver
from ..observe import MiddlewarePipeline
from ..runtime import Runtime, RuntimeAccess, RuntimeServices, build_runtime_access
from ..core import PrincipalProvider
from ..core.errors import ErrorCode, LinktoolsAIError
from ..spec import AgentSpecCodec, PromptSpecCodec
from ..capability.codec import MCPServerSpecCodec, SkillSpecCodec
from ..spec.output import OutputTypeRegistry

_logger = environ.get_logger("ai.entry.services")

@dataclass(frozen=True, slots=True)
class EntryServices:
    runtime: Runtime
    access: RuntimeAccess
    principal_provider: "PrincipalProvider | None" = None


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
    runtime: Runtime,
    services: RuntimeServices,
    principal_provider: "PrincipalProvider | None" = None,
) -> EntryServices:
    return EntryServices(runtime, build_runtime_access(services), principal_provider)


__all__ = ["AgentServices", "EntryServices", "build_agent_services", "build_asset_codecs", "build_services"]
