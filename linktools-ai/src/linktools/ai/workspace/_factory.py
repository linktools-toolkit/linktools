#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from linktools.core import environ

from ..adapter import (
    RuntimeMemoryStore,
    RuntimePersistence,
    RuntimeStepPersistence,
    StepExecutionHistoryReader,
    open_runtime_persistence,
    runtime_durable_domains,
    runtime_storage_engine,
    runtime_storage_kind,
    runtime_storage_path,
)
from ..agent import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AgentCompiler,
    AgentDefinition,
    AgentExecutor,
    AssistantTextOutput,
    OutputTypeRegistry,
)
from ..asset import (
    AssetCacheAdapter,
    AssetInfo,
    AssetKey,
    AssetRef,
    AssetRepository,
    AssetStore,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetTypeRegistrySnapshot,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from ..capability import (
    CapabilityBinding,
    CapabilityProvider,
    MCPRuntimeProvider,
    build_builtin_capability_providers,
    canonical_bootstrap_refs,
)
from ..core import (
    HmacCursorSigner,
    TenantAuthorizationPolicy,
    canonical_sha256,
)
from ..errors import AIError, ErrorCode
from ..model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelMaterializer,
    ModelRegistry,
    ModelResolver,
    ModelRoute,
    OpenAIModelMaterializer,
    SnapshotModelResolver,
    StaticModelCredentialProvider,
)
from ..runtime import (
    ConversationCursor,
    DefaultApprovalService,
    DefaultArtifactService,
    DefaultEvaluationService,
    DefaultEventService,
    DefaultExecutionService,
    DefaultSessionService,
    DefaultTaskService,
    LocalExecutionBackend,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    Runtime,
    RuntimeDomain,
    RuntimeRetention,
    RuntimeStorage,
    RuntimeStores,
    RuntimeTaskNodeRunner,
)
from ..spec import (
    AgentCapabilityRef,
    AgentSpec,
    PromptSpec,
    builtin_asset_bindings,
)
from ..storage import (
    ContentCache,
    StorageLayer,
    StorageOverlay,
    StorageWriter,
)
from ..task import LocalTaskGraphLauncher, TaskNodeRunner
from ._root import Workspace
from ._tools import build_workspace_capability_grants

_logger = environ.get_logger("ai.workspace.runtime")


class _RetentionController(Protocol):
    async def release_execution_handoff(self, execution_id: str, *, tenant_id: str) -> None: ...
    async def release_session(self, session_id: str, *, tenant_id: str, continuation: "ConversationCursor | None") -> None: ...
    async def release_task_graph(self, graph_id: str, *, tenant_id: str) -> None: ...
    async def release_evaluation(self, evaluation_id: str, *, tenant_id: str) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _RuntimeResources:
    namespace: str
    tenant_id: str
    persistence: RuntimePersistence

    @property
    def domain(self) -> RuntimeStores:
        return self.persistence.domain

    @property
    def steps(self) -> RuntimeStepPersistence:
        return self.persistence.steps

    @property
    def retention(self) -> _RetentionController:
        return self.persistence.retention


@asynccontextmanager
async def _open_resources(
    storage: RuntimeStorage,
    *,
    namespace: str,
    tenant_id: str,
) -> "AsyncIterator[_RuntimeResources]":
    async with open_runtime_persistence(storage, namespace=namespace, tenant_id=tenant_id) as persistence:
        yield _RuntimeResources(namespace, tenant_id, persistence)


async def build_asset_store(
    root: str | Path,
    *,
    writer: "StorageWriter[AssetKey, bytes, AssetInfo] | None" = None,
    layers: "Sequence[StorageLayer[AssetKey, bytes, AssetInfo]]" = (),
    cache: "ContentCache | None" = None,
) -> AssetStore:
    """Build a Workspace AssetStore with source-first precedence."""
    workspace_root = Path(root).expanduser().resolve()
    local = LocalDirectoryAssetBackend(
        str(workspace_root),
        writable=False,
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
    )
    selected_writer = writer or InMemoryAssetBackend()
    layer_values = tuple(layers)
    layer_ids = {layer.id for layer in layer_values}
    if writer is None:
        if "workspace-defaults" in layer_ids:
            raise ValueError("workspace-defaults conflicts with an Asset layer")
        layer_values = (*layer_values, StorageLayer("workspace-defaults", selected_writer))
    elif all(layer.backend is not selected_writer for layer in layer_values):
        if "workspace-writer" in layer_ids:
            raise ValueError("workspace-writer conflicts with an Asset layer")
        layer_values = (*layer_values, StorageLayer("workspace-writer", selected_writer))
    store = AssetStore(
        StorageOverlay(
            local,
            writer=selected_writer,
            layers=layer_values,
            cache=cache,
            cache_adapter=AssetCacheAdapter() if cache is not None else None,
        )
    )
    await store.initialize()
    repository = build_workspace_asset_repository(store)
    await _put_default_asset(repository, AssetRef("prompt", "default"), PromptSpec("default", 1, "", ()))
    return store


def build_workspace_asset_repository(
    store: AssetStore,
    *,
    extra_bindings: "Sequence[AssetTypeBinding[object]]" = (),
) -> AssetRepository:
    registry = AssetTypeRegistry()
    for binding in builtin_asset_bindings():
        registry.register(binding)
    for binding in extra_bindings:
        registry.register(binding)
    return AssetRepository(store, registry.freeze())


async def _compile_recovery_definitions(compiler: AgentCompiler, definitions: dict[str, AgentDefinition], stores: RuntimeStores, *, tenant_id: str) -> None:
    checkpoints = await stores.recovery.checkpoints.list(tenant_id=tenant_id)
    for checkpoint in checkpoints:
        if checkpoint.state is RecoveryCheckpointState.COMPLETED or checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE:
            continue
        try:
            definition = await compiler.compile(agent_id=checkpoint.input.agent_id, prompt_id=checkpoint.input.prompt_id)
        except AIError as error:
            if error.code in {ErrorCode.STORAGE_UNAVAILABLE, ErrorCode.STORAGE_INTEGRITY_ERROR, ErrorCode.STORAGE_DEPENDENCY_NOT_READY, ErrorCode.STORAGE_CLOSED}:
                raise
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE, safe_details={"execution_id": checkpoint.execution_id}) from error
        if definition.digest != checkpoint.input.binding_digest:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE, safe_details={"execution_id": checkpoint.execution_id})
        definitions[checkpoint.input.binding_digest] = definition
        _logger.debug(
            "recovery definition compiled: execution=%s agent=%s prompt=%s",
            checkpoint.execution_id,
            checkpoint.input.agent_id,
            checkpoint.input.prompt_id,
        )


@asynccontextmanager
async def open_workspace_runtime(
    workspace: Workspace,
    *,
    runtime_storage: "RuntimeStorage | None" = None,
    asset_store: "AssetStore | None" = None,
    model: "str | None" = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    mcp_runtime: "MCPRuntimeProvider | None" = None,
    extra_asset_bindings: "Sequence[AssetTypeBinding[object]]" = (),
    capability_provider_factories: "Sequence[Callable[[AssetRepository], CapabilityProvider]]" = (),
    capability_grants: "Sequence[CapabilityBinding]" = (),
    task_node_runner: "TaskNodeRunner | None" = None,
) -> "AsyncIterator[Runtime]":
    selected_storage = runtime_storage or RuntimeStorage.filesystem(workspace.storage_root / "runtime")
    namespace = workspace.workspace_id
    tenant_id = workspace.workspace_id
    target_kind = runtime_storage_kind(selected_storage)
    durable_domains = runtime_durable_domains(selected_storage)
    registry = _build_asset_registry(extra_asset_bindings)
    selected_assets = await build_asset_store(workspace.storage_root) if asset_store is None else asset_store
    if asset_store is not None:
        await selected_assets.initialize()
    phase_a_assets = AssetRepository(selected_assets, registry)
    phase_a_providers = _build_capability_providers(phase_a_assets, mcp_runtime, capability_provider_factories)
    phase_a_refs = await _bootstrap_refs(phase_a_providers)
    _logger.debug("workspace capability bootstrap phase A complete: workspace=%s refs=%s", workspace.workspace_id, len(phase_a_refs))
    await _put_default_asset(phase_a_assets, AssetRef("agent", "default"), AgentSpec("default", 1, "default", phase_a_refs, "assistant-text", 1))
    final_providers = _build_capability_providers(phase_a_assets, mcp_runtime, capability_provider_factories)
    phase_b_refs = await _bootstrap_refs(final_providers)
    if phase_a_refs != phase_b_refs:
        _logger.error("workspace capability bootstrap changed between phases: workspace=%s", workspace.workspace_id)
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    _logger.debug("workspace capability bootstrap phase B complete: workspace=%s refs=%s", workspace.workspace_id, len(phase_b_refs))
    output_types = OutputTypeRegistry()
    output_types.register(ASSISTANT_TEXT_OUTPUT_SCHEMA_ID, ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION, AssistantTextOutput)
    output_types.freeze()
    route_model = model or _configured_model(workspace.config)
    if route_model is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "a model route is required")
    connection_id = "local-openai"
    credential_id = "local-openai-api-key" if api_key else None
    connections = ModelConnectionRegistry((ModelConnectionConfig(connection_id, base_url, credential_id=credential_id),))
    model_registry = ModelRegistry()
    snapshot = model_registry.prime({"default": ModelRoute("default", "openai", route_model, connection_id)})
    resolver: ModelResolver = SnapshotModelResolver(snapshot)
    materializer: ModelMaterializer = OpenAIModelMaterializer(StaticModelCredentialProvider({} if api_key is None else {credential_id: api_key}))
    profile = canonical_sha256({"workspace": workspace.root.as_posix(), "grants": 2, "version": 2})
    grants = (*build_workspace_capability_grants(workspace.root), *capability_grants)
    compiler = AgentCompiler(
        phase_a_assets,
        model_resolver=resolver,
        model_connections=connections,
        output_types=output_types,
        capability_providers=final_providers,
        capability_grants=grants,
        execution_profile_fingerprint=profile,
    )
    async with _open_resources(selected_storage, namespace=namespace, tenant_id=tenant_id) as resources:
        definitions: dict[str, AgentDefinition] = {}
        if RuntimeDomain.RECOVERY in durable_domains:
            await _compile_recovery_definitions(compiler, definitions, resources.domain, tenant_id=resources.tenant_id)
        executor = AgentExecutor(materializer, execution_root=workspace.root)
        backend = LocalExecutionBackend(
            resources.domain,
            resources.steps,
            executor,
            definitions,
            tenant_id=tenant_id,
            execution_root=workspace.root,
            step_reads={domain: resources.steps.read_store(domain) for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)},
            step_lifecycle=resources.steps,
            memory_store_factory=lambda memory_tenant, execution_id, memory_scope: RuntimeMemoryStore(resources.domain, tenant_id=memory_tenant, execution_id=execution_id, memory_scope=memory_scope, transient=selected_storage.plan.route(RuntimeDomain.MEMORY).retention is RuntimeRetention.TRANSIENT),
            recovery_enabled=RuntimeDomain.RECOVERY in durable_domains,
            conversation_durable=selected_storage.plan.route(RuntimeDomain.CONVERSATION).retention is RuntimeRetention.DURABLE,
            handoff_contract_digest=_handoff_contract_digest(selected_storage),
        )
        history = StepExecutionHistoryReader(resources.namespace, resources.domain, resources.steps.read_store(RuntimeDomain.EXECUTION), HmacCursorSigner("execution-history", _grant_key(workspace)))
        authorization = TenantAuthorizationPolicy(tenant_id)
        execution = DefaultExecutionService(
            resources.domain,
            authorization,
            backend=backend,
            history_reader=history,
            release_terminal=resources.retention.release_execution_handoff,
            terminal_verifier=backend.verify_terminal_projection,
        )
        session = DefaultSessionService(
            resources.domain,
            authorization,
            execution,
            HmacCursorSigner("session", _grant_key(workspace)),
            release_terminal=resources.retention.release_session,
        )
        task_runner = task_node_runner or RuntimeTaskNodeRunner(execution, definitions)
        task_launcher = LocalTaskGraphLauncher(
            resources.domain.task.tasks,
            task_runner,
            owner=f"workspace:{workspace.workspace_id}",
        )
        task = DefaultTaskService(resources.domain, authorization, task_launcher, release_terminal=resources.retention.release_task_graph)
        evaluation = DefaultEvaluationService(
            resources.domain,
            authorization,
            execution,
            release_terminal=resources.retention.release_evaluation,
            acquire_execution_hold=execution._acquire_dependency_hold,
            release_execution_hold=execution._release_dependency_hold,
            request_execution_handoff=execution._request_terminal_handoff,
        )
        approval = DefaultApprovalService(resources.domain, authorization)
        event = DefaultEventService(resources.domain, authorization)
        artifact = DefaultArtifactService(resources.domain, authorization, grant_key=_grant_key(workspace), cursor_signer=HmacCursorSigner("artifact", _grant_key(workspace)))

        async def close_runtime() -> None:
            async def await_quiesced(coroutine: "Awaitable[None]", *, name: str) -> bool:
                cleanup_task = asyncio.create_task(coroutine, name=name)
                cancelled = False
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        cancelled = True
                await cleanup_task
                return cancelled

            launcher_cancelled = await await_quiesced(
                task_launcher.shutdown(),
                name=f"workspace-task-quiesce-{workspace.workspace_id}",
            )
            backend_cancelled = await await_quiesced(
                backend.close(),
                name=f"workspace-execution-quiesce-{workspace.workspace_id}",
            )
            if launcher_cancelled or backend_cancelled:
                raise asyncio.CancelledError

        runtime = Runtime(compiler, execution, session, task, evaluation, approval, event, artifact, definitions=definitions, close_callback=close_runtime)
        if RuntimeDomain.RECOVERY in durable_domains:
            await backend.reconcile()
        sql_dialect = None
        engine_ownership = (
            "linktools"
            if target_kind == "sqlite"
            else "caller"
            if target_kind == "sql"
            else "none"
        )
        _logger.info(
            "workspace runtime opened: namespace=%s target_kind=%s storage_root=%s location=%s selected_domains=%s recovery_enabled=%s sql_dialect=%s engine_ownership=%s",
            namespace,
            target_kind,
            workspace.storage_root,
            _target_location(selected_storage),
            sorted(domain.value for domain in durable_domains),
            RuntimeDomain.RECOVERY in durable_domains,
            sql_dialect,
            engine_ownership,
        )
        try:
            yield runtime
        finally:
            await runtime.close()

def _build_asset_registry(extra_bindings: Sequence[AssetTypeBinding[object]]) -> AssetTypeRegistrySnapshot:
    registry = AssetTypeRegistry()
    for binding in builtin_asset_bindings():
        registry.register(binding)
    for binding in extra_bindings:
        registry.register(binding)
    return registry.freeze()


async def _put_default_asset(repository: AssetRepository, ref: AssetRef, value: object) -> None:
    try:
        await repository.resolve(ref)
    except AIError as error:
        if error.code is not ErrorCode.STORAGE_NOT_FOUND:
            raise
        await repository.put(ref, value)


def _build_capability_providers(
    assets: AssetRepository,
    mcp_runtime: "MCPRuntimeProvider | None",
    factories: Sequence[Callable[[AssetRepository], CapabilityProvider]],
) -> "tuple[CapabilityProvider, ...]":
    providers = list(build_builtin_capability_providers(assets, mcp_runtime=mcp_runtime))
    providers.extend(factory(assets) for factory in factories)
    return tuple(providers)


async def _bootstrap_refs(providers: Sequence[CapabilityProvider]) -> "tuple[AgentCapabilityRef, ...]":
    refs: list[AgentCapabilityRef] = []
    for provider in providers:
        refs.extend(canonical_bootstrap_refs(provider.provider, await provider.bootstrap_refs()))
    grouped: dict[tuple[str, str], AgentCapabilityRef] = {}
    for ref in refs:
        key = ref.provider, ref.id
        previous = grouped.get(key)
        if previous is not None and previous != ref:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        grouped[key] = ref
    return tuple(sorted(grouped.values(), key=lambda ref: (ref.provider, ref.id, 0 if ref.revision is None else ref.revision)))


def _configured_model(config: dict[str, object]) -> str | None:
    value = config.get("model")
    return value if isinstance(value, str) and value.strip() else None


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


def _handoff_contract_digest(storage: RuntimeStorage) -> str:
    routes: dict[str, dict[str, str]] = {}
    for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
        route = storage.plan.route(domain)
        if route.object_store is not None:
            store_id = route.object_store.store_id
        elif route.retention is RuntimeRetention.DURABLE:
            store_id = "builtin"
        elif route.retention is RuntimeRetention.TRANSIENT:
            store_id = "transient"
        else:
            store_id = "memory"
        routes[domain.value] = {"retention": route.retention.value, "object_store_id": store_id}
    return canonical_sha256({"version": 1, **routes})


def _target_location(storage: RuntimeStorage) -> object:
    return runtime_storage_path(storage)


__all__ = ["RuntimeStorage", "RuntimeDomain", "build_asset_store", "build_workspace_asset_repository", "open_workspace_runtime"]
