#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from linktools.core import environ

from ..adapter import (
    PydanticMCPRuntime,
    RuntimeMemoryStore,
    StepExecutionHistoryReader,
    StepSessionHistoryReader,
)
from ..agent import (
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
    AgentCompiler,
    AgentDefinition,
    AgentDefinitionCatalog,
)
from ..asset import (
    AssetDiscoveryStatus,
    AssetPathAdapter,
    AssetRepository,
    AssetStore,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetTypeRegistrySnapshot,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from ..capability import (
    SKILL_TOOL_NAMES,
    CapabilityBinding,
    CapabilityProvider,
    MCPCapabilityProvider,
    RuntimeCapability,
    SkillCapabilityProvider,
    validate_fingerprint,
)
from ..core import HmacCursorSigner, TenantAuthorizationPolicy, canonical_sha256, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry, ModelResolver
from ..runtime import Runtime, RuntimeState, build_local_runtime
from ..runtime.state import ExecutionReadModelRepository, RuntimeDomain, RuntimeStatePlan, RuntimeStateRoute
from ..spec import AgentSpec, builtin_asset_bindings
from ..storage import ObjectStore, StorageLayer, StorageOverlay
from ._root import Workspace
from ._tools import build_workspace_capabilities

_logger = environ.get_logger("ai.workspace.runtime")


def _build_asset_registry(
    asset_bindings: Sequence[AssetTypeBinding[object]],
) -> AssetTypeRegistrySnapshot:
    bindings = {binding.kind: binding for binding in builtin_asset_bindings()}
    seen: set[str] = set()
    for binding in asset_bindings:
        if binding.kind in seen:
            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)
        seen.add(binding.kind)
        previous = bindings.get(binding.kind)
        if previous is not None and previous.value_type is not binding.value_type:
            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)
        bindings[binding.kind] = binding
    registry = AssetTypeRegistry()
    for kind in sorted(bindings):
        registry.register(bindings[kind])
    return registry.freeze()


async def _build_asset_repository(
    workspace: Workspace,
    *,
    asset: AssetStore | None,
    snapshot: AssetTypeRegistrySnapshot,
    path_adapter: "AssetPathAdapter | None",
) -> AssetRepository:
    if asset is None:
        if path_adapter is None:
            prefixes = {
                "agent": "agents",
                "skill": "skills",
                **{kind: kind for kind in snapshot.kinds if kind not in {"agent", "skill"}},
            }
            path_adapter = PrefixAssetPathAdapter(prefixes)
        path_adapter.validate(snapshot.kinds)
        source = DirectoryAssetBackend(
            str(workspace.storage_root),
            path_adapter=path_adapter,
            kinds=snapshot.kinds,
        )
        writable = InMemoryAssetBackend()
        asset = AssetStore(
            StorageOverlay(
                source,
                writer=writable,
                layers=(StorageLayer("workspace-defaults", writable),),
            )
        )
        await asset.initialize()
    else:
        if path_adapter is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not asset.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return AssetRepository(asset, snapshot)


def _build_providers(
    providers: Sequence[CapabilityProvider],
) -> "tuple[CapabilityProvider, ...]":
    selected: list[CapabilityProvider] = [
        SkillCapabilityProvider(),
        MCPCapabilityProvider(PydanticMCPRuntime()),
    ]
    selected.extend(providers)
    by_name: dict[str, CapabilityProvider] = {}
    by_type: dict[type[object], CapabilityProvider] = {}
    for provider in selected:
        name = provider.provider
        value_type = provider.value_type
        revision = provider.revision
        if (
            not isinstance(name, str)
            or not name.strip()
            or name == "runtime"
            or not isinstance(value_type, type)
            or value_type is AgentSpec
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 1
        ):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        if name in by_name or value_type in by_type:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        by_name[name] = provider
        by_type[value_type] = provider
    return tuple(sorted(selected, key=lambda value: value.provider))


async def _bind_asset_capabilities(
    assets: AssetRepository,
    snapshot: AssetTypeRegistrySnapshot,
    providers: Sequence[CapabilityProvider],
) -> "tuple[CapabilityBinding, ...]":
    result: list[CapabilityBinding] = []
    for provider in providers:
        refs = []
        for kind in sorted(snapshot.kinds):
            binding = snapshot.binding(kind)
            if binding.value_type is not provider.value_type:
                continue
            entries = await assets.discover(kind=kind)
            if any(entry.status is not AssetDiscoveryStatus.RESOLVABLE for entry in entries):
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
            refs.extend(entry.ref for entry in entries)
        ordered = tuple(sorted(refs, key=lambda ref: (ref.kind, ref.id)))
        if not ordered:
            continue
        capability = await provider.bind(ordered, assets=assets)
        if capability.provider != provider.provider:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        validate_fingerprint(capability.fingerprint)
        result.append(capability)
    return tuple(result)


async def _discover_agent_specs(
    assets: AssetRepository,
    snapshot: AssetTypeRegistrySnapshot,
) -> "dict[str, AgentSpec]":
    specs: dict[str, AgentSpec] = {}
    for kind in sorted(snapshot.kinds):
        if snapshot.binding(kind).value_type is not AgentSpec:
            continue
        entries = await assets.discover(kind=kind)
        if any(entry.status is not AssetDiscoveryStatus.RESOLVABLE for entry in entries):
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        for entry in entries:
            resolved = await assets.resolve(entry.ref)
            if type(resolved.spec) is not AgentSpec:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            spec = resolved.spec
            if spec.id in specs:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            specs[spec.id] = spec
    specs.setdefault("default", AgentSpec("default", 1, "default"))
    return specs


def _active_provider_fingerprint(
    providers: Sequence[CapabilityProvider],
    bindings: Sequence[CapabilityBinding],
) -> str:
    active = {binding.provider for binding in bindings}
    selected = tuple(provider for provider in providers if provider.provider in active)
    if active != {provider.provider for provider in selected}:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return canonical_sha256(
        {
            "version": 1,
            "providers": [
                {
                    "provider": provider.provider,
                    "value_type": f"{provider.value_type.__module__}.{provider.value_type.__qualname__}",
                    "revision": provider.revision,
                }
                for provider in selected
            ],
        }
    )


def _agent_catalog_source_fingerprint(
    specs: "dict[str, AgentSpec]",
    models: ModelResolver,
) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "agents": [
                {
                    "spec": _agent_spec_payload(specs[agent_id]),
                    "model_fingerprint": models.resolve(specs[agent_id].model).fingerprint,
                }
                for agent_id in sorted(specs)
            ],
        }
    )


def _agent_spec_payload(spec: AgentSpec) -> "dict[str, object]":
    return {
        "id": spec.id,
        "revision": spec.revision,
        "model": spec.model,
        "system_prompt": spec.system_prompt,
        "instructions": list(spec.instructions),
        "allow_tools": list(spec.allow_tools),
        "metadata": dict(spec.metadata),
        "usage_limits": None
        if spec.usage_limits is None
        else {
            "model_requests": spec.usage_limits.model_requests,
            "tool_calls": spec.usage_limits.tool_calls,
            "input_tokens": spec.usage_limits.input_tokens,
            "output_tokens": spec.usage_limits.output_tokens,
            "total_tokens": spec.usage_limits.total_tokens,
        },
    }


def _platform_policy_fingerprint() -> str:
    return canonical_sha256(
        {
            "version": 1,
            "runtime_contract_revision": 1,
        }
    )


def _trusted_tool_classes(
    asset_capabilities: Sequence[CapabilityBinding],
) -> "dict[str, str]":
    result = {
        name: (
            "filesystem.read"
            if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
            else "filesystem.write"
        )
        for name in WORKSPACE_FILESYSTEM_TOOL_NAMES
    }
    result.update({name: "shell" for name in WORKSPACE_SHELL_TOOL_NAMES})
    if any(binding.provider == "skill" for binding in asset_capabilities):
        result.update({name: "control" for name in SKILL_TOOL_NAMES})
    return result


def _runtime_fingerprint(
    *,
    active_provider_fingerprint: str,
    agent_catalog_source_fingerprint: str,
    platform_policy_fingerprint: str,
) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "active_provider_fingerprint": active_provider_fingerprint,
            "agent_catalog_source_fingerprint": agent_catalog_source_fingerprint,
            "platform_policy_fingerprint": platform_policy_fingerprint,
        }
    )


def _build_default_models(workspace: Workspace) -> ModelRegistry:
    configured = workspace.config.get("model")
    model = configured.strip() if isinstance(configured, str) and configured.strip() else os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    return ModelRegistry.openai(model=model, base_url=base_url, api_key=api_key)


def _default_runtime_state(workspace: Workspace) -> RuntimeState:
    runtime_root = workspace.storage_root / "runtime"
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(runtime_root / "conversation", transaction_root=runtime_root),
        execution=RuntimeStateRoute.filesystem(runtime_root / "execution", transaction_root=runtime_root),
        recovery=RuntimeStateRoute.filesystem(runtime_root / "recovery", transaction_root=runtime_root),
    )
    return RuntimeState.from_plan(plan)


@asynccontextmanager
async def open_workspace_runtime(
    workspace: Workspace,
    *,
    tenant_id: str | None = None,
    asset: "AssetStore | None" = None,
    state: "RuntimeState | None" = None,
    models: "ModelRegistry | None" = None,
    asset_bindings: "Sequence[AssetTypeBinding[object]]" = (),
    capability_providers: "Sequence[CapabilityProvider]" = (),
    capabilities: "Sequence[RuntimeCapability]" = (),
    asset_path_adapter: "AssetPathAdapter | None" = None,
) -> AsyncIterator[Runtime]:
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace is required")
    if any(not isinstance(capability, RuntimeCapability) for capability in capabilities):
        raise TypeError("capabilities must contain RuntimeCapability values")
    effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)
    snapshot = _build_asset_registry(asset_bindings)
    selected_assets = await _build_asset_repository(
        workspace,
        asset=asset,
        snapshot=snapshot,
        path_adapter=asset_path_adapter,
    )
    selected_models = models or _build_default_models(workspace)
    model_resolver = selected_models.snapshot()
    providers = _build_providers(capability_providers)
    initial_revision = await selected_assets.current_revision()
    asset_capabilities = await _bind_asset_capabilities(selected_assets, snapshot, providers)
    runtime_capabilities = tuple(capabilities)
    platform_capabilities = build_workspace_capabilities(workspace.root)
    global_capabilities: tuple[CapabilityBinding, ...] = (
        *asset_capabilities,
        *runtime_capabilities,
    )
    specs = await _discover_agent_specs(selected_assets, snapshot)
    catalog_fingerprint = _agent_catalog_source_fingerprint(specs, model_resolver)
    active_provider_fingerprint = _active_provider_fingerprint(providers, asset_capabilities)
    platform_policy_fingerprint = _platform_policy_fingerprint()
    runtime_fingerprint = _runtime_fingerprint(
        active_provider_fingerprint=active_provider_fingerprint,
        agent_catalog_source_fingerprint=catalog_fingerprint,
        platform_policy_fingerprint=platform_policy_fingerprint,
    )
    compiler = AgentCompiler(
        model_resolver=model_resolver,
        capabilities=global_capabilities,
        platform_capabilities=platform_capabilities,
        runtime_fingerprint=runtime_fingerprint,
        trusted_tool_classes=_trusted_tool_classes(asset_capabilities),
        trusted_mcp_tools=any(
            binding.provider == "mcp" for binding in asset_capabilities
        ),
    )
    catalog = _build_catalog(specs, compiler)
    if await selected_assets.current_revision() != initial_revision:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    selected_runtime = state or _default_runtime_state(workspace)
    try:
        await selected_runtime.initialize(namespace=workspace.workspace_id, tenant_id=effective_tenant_id)
        runtime_value = await _compose_runtime(
            workspace,
            catalog,
            compiler=compiler,
            assets=selected_assets,
            tenant_id=effective_tenant_id,
            state=selected_runtime,
        )
    except BaseException:
        await selected_runtime.close()
        raise
    _logger.info(
        "workspace Runtime opened: workspace=%s tenant=%s active_providers=%s capabilities=%s agents=%s",
        workspace.workspace_id,
        effective_tenant_id,
        tuple(binding.provider for binding in asset_capabilities),
        tuple(
            (capability.provider, capability.id)
            for capability in (*global_capabilities, *platform_capabilities)
        ),
        catalog.root_ids,
    )
    try:
        yield runtime_value
    except BaseException as body_error:
        try:
            await runtime_value.close()
        except BaseException as close_error:
            raise close_error from body_error
        raise
    else:
        await runtime_value.close()


async def _compose_runtime(
    workspace: Workspace,
    catalog: AgentDefinitionCatalog,
    *,
    compiler: AgentCompiler,
    assets: AssetRepository,
    tenant_id: str,
    state: RuntimeState,
) -> Runtime:
    def memory_store_factory(
        tenant_id: str,
        execution_id: str,
        memory_scope: str,
        object_store: ObjectStore,
        transient: bool,
    ) -> RuntimeMemoryStore:
        return RuntimeMemoryStore(
            state.memory,
            object_store=object_store,
            namespace=workspace.workspace_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            memory_scope=memory_scope,
            transient=transient,
        )

    grant_key = _grant_key(workspace)
    execution_history = StepExecutionHistoryReader(
        namespace=workspace.workspace_id,
        executions=state.execution.executions,
        store=state.steps.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("execution-history", grant_key),
        read_model=ExecutionReadModelRepository(
            state.execution.executions.state_store,
            namespace=workspace.workspace_id,
            tenant_id=tenant_id,
        ),
    )
    session_history = StepSessionHistoryReader(
        store=state.steps.read_store(RuntimeDomain.CONVERSATION),
        cursor_signer=HmacCursorSigner("session-history", grant_key),
    )
    return await build_local_runtime(
        state=state,
        catalog=catalog,
        compiler=compiler,
        assets=assets,
        authorization=TenantAuthorizationPolicy(tenant_id),
        tenant_id=tenant_id,
        namespace=workspace.workspace_id,
        execution_root=workspace.root,
        history_reader=execution_history,
        session_history_reader=session_history,
        memory_store_factory=memory_store_factory,
        grant_key=grant_key,
    )


def _build_catalog(
    specs: "dict[str, AgentSpec]",
    compiler: AgentCompiler,
) -> AgentDefinitionCatalog:
    roots: dict[str, AgentDefinition] = {
        agent_id: compiler.compile(specs[agent_id])
        for agent_id in sorted(specs)
    }
    _logger.info("workspace Agent catalog frozen: agents=%s", tuple(sorted(roots)))
    return AgentDefinitionCatalog(roots)


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


__all__ = ["open_workspace_runtime"]
