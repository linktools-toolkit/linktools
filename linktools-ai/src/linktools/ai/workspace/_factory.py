#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from linktools.core import environ

from ..adapter import RuntimeMemoryStore, StepExecutionHistoryReader
from ..agent import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AgentCompiler,
    AssistantTextOutput,
    OutputTypeRegistry,
)
from ..asset import (
    AssetPathAdapter,
    AssetRef,
    AssetRepository,
    AssetStore,
    AssetTypeBinding,
    AssetTypeRegistry,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from ..capability import CapabilityGrant, CapabilityProvider, SkillCapabilityProvider
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

_logger = environ.get_logger("ai.workspace.runtime")


async def _build_default_assets(
    workspace: Workspace,
    *,
    asset_bindings: Sequence[AssetTypeBinding[object]],
    asset_path_adapter: AssetPathAdapter | None,
) -> AssetRepository:
    adapter = asset_path_adapter or PrefixAssetPathAdapter(
        {
            "agent": "agents",
            "skill": "capabilities/skills",
            "mcp": "integrations/mcp",
        }
    )
    source = DirectoryAssetBackend(
        str(workspace.storage_root / "assets"),
        path_adapter=adapter,
    )
    writable = InMemoryAssetBackend()
    store = AssetStore(
        StorageOverlay(
            source,
            writer=writable,
            layers=(StorageLayer("workspace-defaults", writable),),
        )
    )
    await store.initialize()
    registry = AssetTypeRegistry()
    for binding in (*builtin_asset_bindings(), *tuple(asset_bindings)):
        registry.register(binding)
    snapshot = registry.freeze()
    validate_path_adapter(adapter, snapshot.kinds)
    return AssetRepository(store, snapshot)


def validate_path_adapter(adapter: AssetPathAdapter, kinds: Sequence[str]) -> None:
    try:
        adapter.validate(tuple(kinds))
    except (AIError, ValueError) as error:
        if isinstance(error, AIError) and error.code is ErrorCode.ASSET_LAYOUT_CONFLICT:
            raise
        raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT) from error


async def _list_asset_ids(assets: AssetRepository, kind: str) -> tuple[str, ...]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    values: set[str] = set()
    while True:
        page = await assets.list(kind=kind, cursor=cursor, limit=200)
        values.update(item.ref.id for item in page.items)
        if page.next_cursor is None:
            return tuple(sorted(values))
        if page.next_cursor in seen_cursors:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor


async def _list_asset_refs(
    assets: AssetRepository,
    kind: str,
    *,
    required: bool,
) -> tuple[AgentCapabilityRef, ...]:
    provider = "skill" if kind == "skill" else "mcp"
    return tuple(
        AgentCapabilityRef(provider, item, required=required)
        for item in await _list_asset_ids(assets, kind)
    )


async def _put_default_asset(repository: AssetRepository, ref: AssetRef, value: object) -> None:
    try:
        await repository.resolve(ref)
    except AIError as error:
        if error.code is not ErrorCode.STORAGE_NOT_FOUND:
            raise
        await repository.put(ref, value)


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
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(
            workspace.storage_root / "runtime" / "conversation"
        )
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
    assets: "AssetRepository | None" = None,
    asset_bindings: "Sequence[AssetTypeBinding[object]]" = (),
    asset_path_adapter: "AssetPathAdapter | None" = None,
    runtime: "RuntimeState | None" = None,
    models: "ModelRegistry | None" = None,
    output_types: "OutputTypeRegistry | None" = None,
    capabilities: "Sequence[CapabilityGrant]" = (),
    capability_providers: "Sequence[CapabilityProvider]" = (),
) -> AsyncIterator[Runtime]:
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace is required")
    effective_tenant_id = workspace.workspace_id if tenant_id is None else validate_tenant_id(tenant_id)
    if assets is not None and (asset_bindings or asset_path_adapter is not None):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    workspace_owned_assets = assets is None
    selected_assets = assets
    if selected_assets is None:
        selected_assets = await _build_default_assets(
            workspace,
            asset_bindings=asset_bindings,
            asset_path_adapter=asset_path_adapter,
        )
    elif not selected_assets.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "AssetRepository must be ready")
    providers = (SkillCapabilityProvider(), *tuple(capability_providers))
    _validate_providers(providers)
    if workspace_owned_assets:
        skill_refs = await _list_asset_refs(selected_assets, "skill", required=False)
        mcp_refs = (
            await _list_asset_refs(selected_assets, "mcp", required=False)
            if any(provider.provider == "mcp" for provider in capability_providers)
            else ()
        )
        await _put_default_asset(
            selected_assets,
            AssetRef("agent", "default"),
            AgentSpec(
                "default",
                1,
                "default",
                (*skill_refs, *mcp_refs),
                ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
                ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
            ),
        )
    selected_runtime = runtime or _default_runtime_state(workspace)
    try:
        await selected_runtime.initialize(
            namespace=workspace.workspace_id,
            tenant_id=effective_tenant_id,
        )
        runtime_value = await _compose_runtime(
            workspace,
            selected_assets,
            tenant_id=effective_tenant_id,
            providers=providers,
            capabilities=capabilities,
            models=models,
            output_types=output_types,
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
    assets: AssetRepository,
    *,
    tenant_id: str,
    providers: Sequence[CapabilityProvider],
    capabilities: Sequence[CapabilityGrant],
    models: ModelRegistry | None,
    output_types: OutputTypeRegistry | None,
    state: RuntimeState,
) -> Runtime:
    selected_models = models or _build_default_models(workspace)
    output_registry = _build_output_types(output_types)
    grants = (*_build_workspace_capability_grants(workspace), *tuple(capabilities))
    _validate_grants(grants)
    profile = canonical_sha256(
        {
            "version": 4,
            "workspace_id": workspace.workspace_id,
            "platform_capabilities": {
                "filesystem": {"version": 1},
                "shell": {"version": 1},
                "memory": {"version": 1},
                "subagent": {"version": 1, "tool": "delegate_task", "nested": False},
            },
            "direct_grants": [
                {
                    "id": grant.id,
                    "fingerprint": grant.fingerprint,
                    "inherit_to_subagents": grant.inherit_to_subagents,
                }
                for grant in sorted(grants, key=lambda value: value.id)
            ],
        }
    )
    compiler = AgentCompiler(
        assets,
        model_resolver=selected_models.snapshot(),
        output_types=output_registry,
        capability_providers=providers,
        capability_grants=grants,
        execution_profile_fingerprint=profile,
    )

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
    history = StepExecutionHistoryReader(
        namespace=workspace.workspace_id,
        executions=state.execution.executions,
        store=state.steps.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("execution-history", grant_key),
    )
    return await build_local_runtime(
        state=state,
        compiler=compiler,
        authorization=TenantAuthorizationPolicy(tenant_id),
        tenant_id=tenant_id,
        namespace=workspace.workspace_id,
        execution_root=workspace.root,
        history_reader=history,
        memory_store_factory=memory_store_factory,
        grant_key=grant_key,
    )


def _validate_providers(providers: Sequence[CapabilityProvider]) -> None:
    names = [provider.provider for provider in providers]
    if len(names) != len(set(names)) or "application" in names:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _validate_grants(grants: Sequence[CapabilityGrant]) -> None:
    identities = [(grant.provider, grant.id) for grant in grants]
    if len(identities) != len(set(identities)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _build_workspace_capability_grants(workspace: Workspace) -> tuple[CapabilityGrant, ...]:
    from ._tools import build_workspace_capability_grants

    return build_workspace_capability_grants(workspace.root)


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


__all__ = ["open_workspace_runtime", "validate_path_adapter"]
