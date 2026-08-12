#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from linktools.core import environ
from pydantic_ai_harness.step_persistence import (
    InMemoryStepStore,
    StepStore,
)

from ..adapter import (
    DurableFilesystemStepStore,
    RoutedStepStore,
    RuntimeMemoryStore,
    SqlStepStore,
    StepExecutionHistoryReader,
    build_filesystem_runtime,
    build_in_memory_runtime,
    open_sql_runtime,
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
    AssetRef,
    AssetInfo,
    AssetKey,
    AssetRepository,
    AssetStore,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetTypeRegistrySnapshot,
    FilesystemAssetBackend,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
    PrefixAssetPathAdapter,
    SqlAssetBackend,
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
    DefaultApprovalService,
    DefaultArtifactService,
    DefaultEvaluationService,
    DefaultEventService,
    DefaultExecutionService,
    DefaultSessionService,
    DefaultTaskService,
    LocalExecutionBackend,
    Runtime,
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
    RuntimeStorage,
    StorageDomain,
    StorageLayer,
    StorageOverlay,
    StorageWriter,
    create_sql_storage_context,
)
from ..task import LocalTaskGraphLauncher, TaskNodeRunner
from ._root import Workspace
from ._tools import build_workspace_capability_grants

_logger = environ.get_logger("ai.workspace.runtime")


@dataclass(frozen=True, slots=True)
class _RuntimeResources:
    storage: RuntimeStorage
    namespace: str
    domain: RuntimeStores
    steps: StepStore
    sql_context: "object | None" = None


@asynccontextmanager
async def _open_resources(
    storage: RuntimeStorage,
    *,
    namespace: str,
    sql_context: "object | None" = None,
) -> "AsyncIterator[_RuntimeResources]":
    if storage.target_kind == "memory":
        runtime = build_in_memory_runtime(namespace=namespace)
        steps = InMemoryStepStore()
        await runtime.initialize()
        try:
            yield _RuntimeResources(storage, namespace, runtime.persistence, steps)
        finally:
            await runtime.close()
        return
    if storage.target_kind == "filesystem":
        runtime = build_filesystem_runtime(str(storage.location), namespace=namespace, persist=storage.persist)
        durable_steps = DurableFilesystemStepStore(runtime.runtime_root, namespace, writer_lock=runtime.writer_lock)
        steps = RoutedStepStore(InMemoryStepStore(), durable_steps, storage.persist)
        await runtime.initialize()
        await steps.initialize()
        try:
            yield _RuntimeResources(storage, namespace, runtime.persistence, steps)
        finally:
            await steps.close()
            await runtime.close()
        return
    owns_context = sql_context is None
    if sql_context is None:
        sql_context = await _prepare_sql_context(storage, namespace)
    domain = await open_sql_runtime(sql_context.engine, namespace=namespace, persist=storage.persist)
    durable_steps = SqlStepStore(sql_context.engine, namespace=namespace)
    steps = RoutedStepStore(InMemoryStepStore(), durable_steps, storage.persist)
    await steps.initialize()
    try:
        await domain.initialize()
        yield _RuntimeResources(storage, namespace, domain, steps, sql_context)
    finally:
        await steps.close()
        await domain.close()
        if owns_context:
            await sql_context.close()


async def build_asset_store(root: str | Path) -> AssetStore:
    """Build the read-only workspace asset layer."""
    workspace_root = Path(root).expanduser().resolve()
    local = LocalDirectoryAssetBackend(
        str(workspace_root),
        writable=False,
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
    )
    defaults = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(local, writer=defaults, layers=(StorageLayer("defaults", defaults),)))
    await store.initialize()
    repository = build_workspace_asset_repository(store)
    await _put_default_asset(repository, AssetRef("prompt", "default"), PromptSpec("default", 1, "", ()))
    return store


async def _prepare_sql_context(storage: RuntimeStorage, namespace: str) -> object:
    from sqlalchemy.ext.asyncio import create_async_engine

    if storage.target_kind == "sqlite":
        from ..migrate import build_schema_metadata, provision_database

        engine = create_async_engine(f"sqlite+aiosqlite:///{storage.location}")
        context = create_sql_storage_context(engine, namespace, owns_engine=True)
        try:
            await provision_database(engine)
            metadata, digest = build_schema_metadata()
            await context.initialize(metadata=metadata, schema_manifest_digest=digest)
        except Exception:
            await context.close()
            raise
        return context
    if storage.engine is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "SQL runtime engine is required")
    from ..migrate import build_schema_metadata

    context = create_sql_storage_context(storage.engine, namespace)
    metadata, digest = build_schema_metadata()
    await context.initialize(metadata=metadata, schema_manifest_digest=digest)
    return context


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


async def _compile_recovery_definitions(compiler: AgentCompiler, definitions: dict[str, AgentDefinition], stores: RuntimeStores) -> None:
    checkpoints = await stores.recovery_checkpoint.list(tenant_id=stores.namespace)
    for checkpoint in checkpoints:
        if checkpoint.phase in {"completed", "failed", "cancelled"}:
            continue
        agent_id = checkpoint.payload.get("agent_id")
        prompt_id = checkpoint.payload.get("prompt_id")
        if not isinstance(agent_id, str) or not isinstance(prompt_id, str) or not agent_id.strip() or not prompt_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        definition = await compiler.compile(agent_id=agent_id, prompt_id=prompt_id)
        definitions[definition.digest] = definition
        _logger.debug(
            "recovery definition compiled: checkpoint=%s agent=%s prompt=%s",
            checkpoint.checkpoint_id,
            agent_id,
            prompt_id,
        )


@asynccontextmanager
async def open_workspace_runtime(
    workspace: Workspace,
    *,
    storage: "RuntimeStorage | None" = None,
    model: "str | None" = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    mcp_runtime: "MCPRuntimeProvider | None" = None,
    extra_asset_bindings: "Sequence[AssetTypeBinding[object]]" = (),
    capability_provider_factories: "Sequence[Callable[[AssetRepository], CapabilityProvider]]" = (),
    capability_grants: "Sequence[CapabilityBinding]" = (),
    task_node_runner: "TaskNodeRunner | None" = None,
) -> "AsyncIterator[Runtime]":
    selected_storage = storage or RuntimeStorage.filesystem(workspace.storage_root / "runtime")
    namespace = workspace.workspace_id
    registry = _build_asset_registry(extra_asset_bindings)
    sql_context = None
    if selected_storage.target_kind in {"sqlite", "sql"}:
        sql_context = await _prepare_sql_context(selected_storage, namespace)
    try:
        writer = _asset_writer(selected_storage, namespace, sql_context)
        phase_a_store, phase_a_assets = await _create_workspace_repository(workspace.storage_root / "assets", registry, writer=writer)
        phase_a_providers = _build_capability_providers(phase_a_assets, mcp_runtime, capability_provider_factories)
        phase_a_refs = await _bootstrap_refs(phase_a_providers)
        _logger.debug("workspace capability bootstrap phase A complete: workspace=%s refs=%s", workspace.workspace_id, len(phase_a_refs))
        del phase_a_store
        final_store, assets = await _create_workspace_repository(workspace.storage_root / "assets", registry, generated_refs=phase_a_refs, writer=writer)
        final_providers = _build_capability_providers(assets, mcp_runtime, capability_provider_factories)
        phase_b_refs = await _bootstrap_refs(final_providers)
        if phase_a_refs != phase_b_refs:
            _logger.error("workspace capability bootstrap changed between phases: workspace=%s", workspace.workspace_id)
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.debug("workspace capability bootstrap phase B complete: workspace=%s refs=%s", workspace.workspace_id, len(phase_b_refs))
        del final_store
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
            assets,
            model_resolver=resolver,
            model_connections=connections,
            output_types=output_types,
            capability_providers=final_providers,
            capability_grants=grants,
            execution_profile_fingerprint=profile,
        )
        async with _open_resources(selected_storage, namespace=namespace, sql_context=sql_context) as resources:
            definitions: dict[str, AgentDefinition] = {}
            if StorageDomain.RECOVERY in selected_storage.persist:
                await _compile_recovery_definitions(compiler, definitions, resources.domain)
            executor = AgentExecutor(materializer, execution_root=workspace.root)
            backend = LocalExecutionBackend(
                resources.domain,
                resources.steps,
                executor,
                definitions,
                tenant_id=workspace.workspace_id,
                execution_root=workspace.root,
                memory_store_factory=lambda tenant_id, namespace: RuntimeMemoryStore(resources.domain, tenant_id=tenant_id, namespace=namespace),
            )
            history = StepExecutionHistoryReader(resources.namespace, resources.domain, resources.steps, HmacCursorSigner("execution-history", _grant_key(workspace)))
            authorization = TenantAuthorizationPolicy(workspace.workspace_id)
            execution = DefaultExecutionService(resources.domain, authorization, backend=backend, history_reader=history)
            session = DefaultSessionService(resources.domain, authorization, execution, HmacCursorSigner("session", _grant_key(workspace)))
            task_runner = task_node_runner or RuntimeTaskNodeRunner(execution, definitions)
            task_launcher = LocalTaskGraphLauncher(
                resources.domain.task,
                task_runner,
                owner=f"workspace:{workspace.workspace_id}",
            )
            task = DefaultTaskService(resources.domain, authorization, task_launcher)
            evaluation = DefaultEvaluationService(resources.domain, authorization, execution)
            approval = DefaultApprovalService(resources.domain, authorization)
            event = DefaultEventService(resources.domain, authorization)
            artifact = DefaultArtifactService(resources.domain, authorization, grant_key=_grant_key(workspace), cursor_signer=HmacCursorSigner("artifact", _grant_key(workspace)))

            async def close_runtime() -> None:
                await task_launcher.shutdown()
                await backend.close()

            runtime = Runtime(compiler, execution, session, task, evaluation, approval, event, artifact, definitions=definitions, close_callback=close_runtime)
            if StorageDomain.RECOVERY in selected_storage.persist:
                await backend.reconcile()
            sql_dialect = None if sql_context is None else sql_context.dialect.name
            engine_ownership = (
                "linktools"
                if selected_storage.owns_engine
                else "caller"
                if selected_storage.target_kind == "sql"
                else "none"
            )
            _logger.info(
                "workspace runtime opened: namespace=%s target_kind=%s storage_root=%s location=%s selected_domains=%s recovery_enabled=%s sql_dialect=%s engine_ownership=%s",
                namespace,
                selected_storage.target_kind,
                workspace.storage_root,
                selected_storage.location,
                sorted(domain.value for domain in selected_storage.persist),
                StorageDomain.RECOVERY in selected_storage.persist,
                sql_dialect,
                engine_ownership,
            )
            try:
                yield runtime
            finally:
                await runtime.close()
    finally:
        if sql_context is not None:
            await sql_context.close()

def _build_asset_registry(extra_bindings: Sequence[AssetTypeBinding[object]]) -> AssetTypeRegistrySnapshot:
    registry = AssetTypeRegistry()
    for binding in builtin_asset_bindings():
        registry.register(binding)
    for binding in extra_bindings:
        registry.register(binding)
    return registry.freeze()


async def _create_workspace_repository(
    root: Path,
    registry: AssetTypeRegistrySnapshot,
    *,
    generated_refs: "tuple[AgentCapabilityRef, ...] | None" = None,
    writer: "StorageWriter[AssetKey, bytes, AssetInfo] | None" = None,
) -> "tuple[AssetStore, AssetRepository]":
    local = LocalDirectoryAssetBackend(
        str(root),
        writable=False,
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
    )
    writer_backend: StorageWriter[AssetKey, bytes, AssetInfo] = InMemoryAssetBackend()
    if writer is not None:
        writer_backend = writer
    store = AssetStore(StorageOverlay(local, writer=writer_backend, layers=(StorageLayer("defaults", writer_backend),)))
    await store.initialize()
    repository = AssetRepository(store, registry)
    await _put_default_asset(repository, AssetRef("prompt", "default"), PromptSpec("default", 1, "", ()))
    if generated_refs is not None:
        await _put_default_asset(repository, AssetRef("agent", "default"), AgentSpec("default", 1, "default", generated_refs, "assistant-text", 1))
    return store, repository


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


def _asset_writer(
    storage: RuntimeStorage,
    namespace: str,
    sql_context: "object | None",
) -> "StorageWriter[AssetKey, bytes, AssetInfo]":
    if StorageDomain.ASSET not in storage.persist:
        return InMemoryAssetBackend()
    if storage.target_kind == "filesystem" and storage.location is not None:
        namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        return FilesystemAssetBackend(str(storage.location / namespace_key / "asset"))
    if storage.target_kind in {"sqlite", "sql"} and sql_context is not None:
        return SqlAssetBackend(sql_context.engine, namespace=namespace)
    return InMemoryAssetBackend()


__all__ = ["RuntimeStorage", "StorageDomain", "build_asset_store", "build_workspace_asset_repository", "open_workspace_runtime"]
