#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

from linktools.core import environ

from ..adapter import (
    PydanticMCPRuntime,
    RuntimeMemoryStore,
    StepExecutionHistoryReader,
    StepSessionHistoryReader,
)
from ..agent import AgentCompiler, AgentDefinition, AgentDefinitionCatalog
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
    CapabilityBinding,
    CapabilityProvider,
    MCPCapabilityProvider,
    RuntimeCapability,
    SkillCapabilityProvider,
    SubagentCapabilityProvider,
    validate_fingerprint,
)
from ..core import HmacCursorSigner, TenantAuthorizationPolicy, canonical_sha256, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
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
    selected: dict[str, CapabilityProvider] = {
        "agent": SubagentCapabilityProvider(),
        "skill": SkillCapabilityProvider(),
        "mcp": MCPCapabilityProvider(PydanticMCPRuntime()),
    }
    custom_names: set[str] = set()
    for provider in providers:
        name = provider.provider
        if not isinstance(name, str) or not name.strip() or name == "runtime" or name in custom_names:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        if not isinstance(provider.value_type, type):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        custom_names.add(name)
        previous = selected.get(name)
        if previous is not None and previous.value_type is not provider.value_type:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        selected[name] = provider
    by_type: dict[type[object], str] = {}
    for name, provider in selected.items():
        previous_name = by_type.get(provider.value_type)
        if previous_name is not None and previous_name != name:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        by_type[provider.value_type] = name
    return tuple(selected[name] for name in sorted(selected))


async def _bind_asset_capabilities(
    assets: AssetRepository,
    snapshot: AssetTypeRegistrySnapshot,
    providers: Sequence[CapabilityProvider],
) -> "dict[str, CapabilityBinding]":
    bindings: dict[str, CapabilityBinding] = {}
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
        bindings[provider.provider] = capability
    return bindings


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
    asset_bindings: "Sequence[AssetTypeBinding[object]]" = (),
    asset_path_adapter: "AssetPathAdapter | None" = None,
    state: "RuntimeState | None" = None,
    models: "ModelRegistry | None" = None,
    capability_providers: "Sequence[CapabilityProvider]" = (),
    capabilities: "Sequence[RuntimeCapability]" = (),
) -> AsyncIterator[Runtime]:
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace is required")
    effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)
    snapshot = _build_asset_registry(asset_bindings)
    selected_assets = await _build_asset_repository(
        workspace,
        asset=asset,
        snapshot=snapshot,
        path_adapter=asset_path_adapter,
    )
    selected_models = models or _build_default_models(workspace)
    providers = _build_providers(capability_providers)
    initial_revision = await selected_assets.current_revision()
    family_bindings = await _bind_asset_capabilities(selected_assets, snapshot, providers)
    runtime_capabilities = (*build_workspace_capabilities(workspace.root), *tuple(capabilities))
    default_capabilities: tuple[CapabilityBinding, ...] = (
        *((family_bindings["skill"],) if "skill" in family_bindings else ()),
        *runtime_capabilities,
    )
    compiler = _build_compiler(
        capabilities=default_capabilities,
        models=selected_models,
        workspace=workspace,
        asset_registry=snapshot,
        capability_providers=providers,
        capability_bindings=family_bindings,
    )
    catalog = await _build_catalog(selected_assets, snapshot=snapshot, compiler=compiler)
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
            asset_registry=snapshot,
            capability_providers=providers,
            capability_bindings=family_bindings,
            tenant_id=effective_tenant_id,
            state=selected_runtime,
        )
    except BaseException:
        await selected_runtime.close()
        raise
    _logger.info(
        "workspace Runtime opened: workspace=%s tenant=%s available_capabilities=%s default_asset_capabilities=%s",
        workspace.workspace_id,
        effective_tenant_id,
        tuple(sorted(family_bindings)),
        ("skill",) if "skill" in family_bindings else (),
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
    asset_registry: AssetTypeRegistrySnapshot,
    capability_providers: Sequence[CapabilityProvider],
    capability_bindings: Mapping[str, CapabilityBinding],
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
        asset_registry=asset_registry,
        capability_providers=capability_providers,
        capability_bindings=capability_bindings,
        authorization=TenantAuthorizationPolicy(tenant_id),
        tenant_id=tenant_id,
        namespace=workspace.workspace_id,
        execution_root=workspace.root,
        history_reader=execution_history,
        session_history_reader=session_history,
        memory_store_factory=memory_store_factory,
        grant_key=grant_key,
    )


def _build_compiler(
    *,
    capabilities: Sequence[CapabilityBinding],
    models: ModelRegistry,
    workspace: Workspace,
    asset_registry: AssetTypeRegistrySnapshot,
    capability_providers: Sequence[CapabilityProvider],
    capability_bindings: Mapping[str, CapabilityBinding],
) -> AgentCompiler:
    profile = canonical_sha256(
        {
            "version": 7,
            "workspace_id": workspace.workspace_id,
            "platform_capabilities": {
                "filesystem": {"version": 1},
                "shell": {"version": 1},
                "memory": {"version": 1},
                "subagent": {"version": 1, "tool": "delegate_task", "nested": False},
                "planning": {"version": 1},
                "thinking": {"version": 1},
            },
            "global_capabilities": [
                {"provider": capability.provider, "id": capability.id, "fingerprint": capability.fingerprint}
                for capability in sorted(capabilities, key=lambda value: (value.provider, value.id))
            ],
            "asset_capability_families": [
                {"provider": name, "fingerprint": binding.fingerprint}
                for name, binding in sorted(capability_bindings.items())
            ],
        }
    )
    return AgentCompiler(
        model_resolver=models.snapshot(),
        capabilities=capabilities,
        execution_profile_fingerprint=profile,
        asset_registry=asset_registry,
        capability_providers=capability_providers,
        capability_bindings=capability_bindings,
    )


async def _build_catalog(
    assets: AssetRepository,
    *,
    snapshot: AssetTypeRegistrySnapshot,
    compiler: AgentCompiler,
) -> AgentDefinitionCatalog:
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
    roots: dict[str, AgentDefinition] = {
        agent_id: compiler.compile(specs[agent_id])
        for agent_id in sorted(specs)
    }
    subagents = {
        agent_id: compiler.derive_subagent(definition)
        for agent_id, definition in roots.items()
    }
    _logger.info("workspace Agent catalog frozen: agents=%s", tuple(sorted(roots)))
    return AgentDefinitionCatalog(roots, subagents)


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


__all__ = ["open_workspace_runtime"]
