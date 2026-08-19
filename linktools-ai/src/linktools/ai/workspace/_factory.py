#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

from linktools.core import environ

from ..adapter import (
    RuntimeMemoryStore,
    StepExecutionHistoryReader,
    StepSessionHistoryReader,
)
from ..agent import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AgentCompiler,
    AgentDefinition,
    AgentDefinitionCatalog,
    AssistantTextOutput,
    OutputTypeRegistry,
)
from ..asset import (
    AssetDiscoveryStatus,
    AssetRepository,
    AssetStore,
    AssetTypeRegistry,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from ..capability import RuntimeCapability, SkillCapabilityProvider
from ..core import (
    HmacCursorSigner,
    TenantAuthorizationPolicy,
    canonical_sha256,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
from ..runtime import Runtime, RuntimeState, build_local_runtime
from ..runtime.state import RuntimeDomain, RuntimeStatePlan, RuntimeStateRoute
from ..spec import AgentCapabilityRef, AgentSpec, builtin_asset_bindings
from ..storage import ObjectStore, StorageLayer, StorageOverlay
from ._root import Workspace
from ._source import CapabilitySource
from ._tools import build_workspace_capabilities

_logger = environ.get_logger("ai.workspace.runtime")


async def _build_asset_repository(
    workspace: Workspace,
    *,
    asset: AssetStore | None,
    sources: Sequence[CapabilitySource],
) -> AssetRepository:
    registry = AssetTypeRegistry()
    builtins = {binding.kind: binding for binding in builtin_asset_bindings()}
    bindings_by_kind = dict(builtins)
    bindings_by_kind.update(
        (source.asset_binding.kind, source.asset_binding)
        for source in sources
    )
    bindings = tuple(bindings_by_kind.values())
    for binding in bindings:
        registry.register(binding)
    snapshot = registry.freeze()
    path_adapter: PrefixAssetPathAdapter | None = None
    if asset is None:
        prefixes = {
            "agent": "agents",
            "skill": "skills",
            **{
                kind: kind
                for kind in snapshot.kinds
                if kind not in {"agent", "skill"}
            },
        }
        path_adapter = PrefixAssetPathAdapter(prefixes)
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
    elif not asset.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if path_adapter is not None:
        path_adapter.validate(snapshot.kinds)
    return AssetRepository(asset, snapshot)


def _validate_sources(sources: Sequence[CapabilitySource]) -> None:
    kinds: set[str] = set()
    providers: set[str] = set()
    for source in sources:
        kind = source.asset_binding.kind
        provider = source.provider.provider
        if kind == "agent" or kind in kinds:
            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)
        if provider == "runtime" or provider in providers:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        kinds.add(kind)
        providers.add(provider)


async def _source_refs(
    assets: AssetRepository,
    source: CapabilitySource,
) -> tuple[AgentCapabilityRef, ...]:
    entries = await assets.discover(kind=source.asset_binding.kind)
    if any(entry.status is not AssetDiscoveryStatus.RESOLVABLE for entry in entries):
        raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
    return tuple(
        AgentCapabilityRef(
            source.provider.provider,
            entry.ref.id,
            required=True,
        )
        for entry in entries
    )


def _merge_default_capabilities(
    explicit: Sequence[AgentCapabilityRef],
    discovered: Sequence[AgentCapabilityRef],
) -> tuple[AgentCapabilityRef, ...]:
    discovered_keys = {(ref.provider, ref.id) for ref in discovered}
    explicit_keys: set[tuple[str, str]] = set()
    merged: list[AgentCapabilityRef] = []
    for ref in explicit:
        key = ref.provider, ref.id
        merged.append(replace(ref, required=True) if key in discovered_keys else ref)
        explicit_keys.add(key)
    merged.extend(
        ref
        for ref in discovered
        if (ref.provider, ref.id) not in explicit_keys
    )
    return tuple(merged)


def _build_default_models(workspace: Workspace) -> ModelRegistry:
    configured = workspace.config.get("model")
    model = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else os.getenv("OPENAI_MODEL", "").strip()
    )
    if not model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    return ModelRegistry.openai(model=model, base_url=base_url, api_key=api_key)


def _default_runtime_state(workspace: Workspace) -> RuntimeState:
    runtime_root = workspace.storage_root / "runtime"
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(
            runtime_root / "conversation",
            transaction_root=runtime_root,
        ),
        execution=RuntimeStateRoute.filesystem(
            runtime_root / "execution",
            transaction_root=runtime_root,
        ),
        recovery=RuntimeStateRoute.filesystem(
            runtime_root / "recovery",
            transaction_root=runtime_root,
        ),
    )
    return RuntimeState.from_plan(plan)


def _build_output_types(
    output_types: OutputTypeRegistry | None,
) -> OutputTypeRegistry:
    selected = output_types or OutputTypeRegistry()
    if not selected.frozen:
        selected.register(
            ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
            ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
            AssistantTextOutput,
        )
        selected.freeze()
    else:
        resolved = selected.resolve(
            ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
            ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
        )
        if resolved is not AssistantTextOutput:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_DRIFT)
    return selected


@asynccontextmanager
async def open_workspace_runtime(
    workspace: Workspace,
    *,
    tenant_id: str | None = None,
    asset: "AssetStore | None" = None,
    state: "RuntimeState | None" = None,
    models: "ModelRegistry | None" = None,
    output_types: "OutputTypeRegistry | None" = None,
    capability_sources: "Sequence[CapabilitySource]" = (),
    capabilities: "Sequence[RuntimeCapability]" = (),
) -> AsyncIterator[Runtime]:
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace is required")
    effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)
    builtins = {binding.kind: binding for binding in builtin_asset_bindings()}
    sources = (
        CapabilitySource(builtins["skill"], SkillCapabilityProvider()),
        *tuple(capability_sources),
    )
    _validate_sources(sources)
    selected_assets = await _build_asset_repository(
        workspace,
        asset=asset,
        sources=sources,
    )
    selected_models = models or _build_default_models(workspace)
    output_registry = _build_output_types(output_types)
    effective_capabilities = (
        *build_workspace_capabilities(workspace.root),
        *tuple(capabilities),
    )
    initial_revision = await selected_assets.current_revision()
    compiler = _build_compiler(
        selected_assets,
        sources=sources,
        capabilities=effective_capabilities,
        models=selected_models,
        output_types=output_registry,
        workspace=workspace,
    )
    catalog = await _build_catalog(selected_assets, sources=sources, compiler=compiler)
    if await selected_assets.current_revision() != initial_revision:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    selected_runtime = state or _default_runtime_state(workspace)
    try:
        await selected_runtime.initialize(
            namespace=workspace.workspace_id,
            tenant_id=effective_tenant_id,
        )
        runtime_value = await _compose_runtime(
            workspace,
            catalog,
            tenant_id=effective_tenant_id,
            state=selected_runtime,
        )
    except BaseException:
        await selected_runtime.close()
        raise
    _logger.info(
        "workspace Runtime opened: workspace=%s tenant=%s",
        workspace.workspace_id,
        effective_tenant_id,
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
    )
    session_history = StepSessionHistoryReader(
        store=state.steps.read_store(RuntimeDomain.CONVERSATION),
        cursor_signer=HmacCursorSigner("session-history", grant_key),
    )
    return await build_local_runtime(
        state=state,
        catalog=catalog,
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
    assets: AssetRepository,
    *,
    sources: Sequence[CapabilitySource],
    capabilities: Sequence[RuntimeCapability],
    models: ModelRegistry,
    output_types: OutputTypeRegistry,
    workspace: Workspace,
) -> AgentCompiler:
    profile = canonical_sha256(
        {
            "version": 5,
            "workspace_id": workspace.workspace_id,
            "platform_capabilities": {
                "filesystem": {"version": 1},
                "shell": {"version": 1},
                "memory": {"version": 1},
                "subagent": {"version": 1, "tool": "delegate_task", "nested": False},
            },
            "direct_capabilities": [
                {
                    "id": capability.id,
                    "fingerprint": capability.fingerprint,
                    "inherit_to_subagents": capability.inherit_to_subagents,
                }
                for capability in sorted(capabilities, key=lambda value: value.id)
            ],
        }
    )
    return AgentCompiler(
        assets,
        model_resolver=models.snapshot(),
        output_types=output_types,
        capability_providers=tuple(source.provider for source in sources),
        capabilities=capabilities,
        execution_profile_fingerprint=profile,
    )


async def _build_catalog(
    assets: AssetRepository,
    *,
    sources: Sequence[CapabilitySource],
    compiler: AgentCompiler,
) -> AgentDefinitionCatalog:
    entries = await assets.discover(kind="agent")
    specs: dict[str, AgentSpec] = {}
    for entry in entries:
        if entry.status is not AssetDiscoveryStatus.RESOLVABLE:
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        resolved = await assets.resolve(entry.ref)
        if type(resolved.spec) is not AgentSpec:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        specs[entry.ref.id] = resolved.spec
    default = specs.get("default")
    source_refs: list[AgentCapabilityRef] = []
    for source in sources:
        source_refs.extend(await _source_refs(assets, source))
    if default is None:
        default = AgentSpec(
            "default",
            1,
            "default",
            tuple(source_refs),
            ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
            ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
        )
    else:
        default = replace(
            default,
            capabilities=_merge_default_capabilities(
                default.capabilities,
                source_refs,
            ),
        )
    specs["default"] = default
    roots: dict[str, AgentDefinition] = {}
    for agent_id in sorted(specs):
        roots[agent_id] = await compiler.compile(specs[agent_id])
    subagents = {
        agent_id: compiler.derive_subagent(definition)
        for agent_id, definition in roots.items()
    }
    _logger.info(
        "workspace Agent catalog frozen: agents=%s capabilities=%s",
        tuple(sorted(roots)),
        tuple(source.provider.provider for source in sources),
    )
    return AgentDefinitionCatalog(roots, subagents)


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


__all__ = ["open_workspace_runtime"]
