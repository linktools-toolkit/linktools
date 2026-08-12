#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ
from pydantic_ai_harness.step_persistence import (
    InMemoryStepStore,
    SqliteStepStore,
    StepStore,
)

from ..adapter import (
    DurableFilesystemStepStore,
    RuntimeMemoryStore,
    SqlRuntimeSchema,
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
from ..core import HmacCursorSigner, TenantAuthorizationPolicy, canonical_sha256
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
    RuntimeBackend,
    RuntimePersistence,
    RuntimeTaskNodeRunner,
)
from ..spec import (
    AgentCapabilityRef,
    AgentSpec,
    PromptSpec,
    builtin_asset_bindings,
)
from ..storage import StorageLayer, StorageOverlay
from ..task import LocalTaskGraphLauncher, TaskNodeRunner
from ._root import Workspace
from ._tools import build_workspace_capability_grants

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.workspace.runtime")


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


@dataclass(frozen=True, slots=True)
class RuntimePersistenceConfig:
    """Select the runtime persistence implementation."""

    backend: RuntimeBackend
    namespace: str
    deployment_id: str = "workspace"
    location: str | None = field(default=None, repr=False)
    local_tenant_id: str | None = None

    @classmethod
    def in_memory(cls, *, namespace: str, deployment_id: str = "memory") -> "RuntimePersistenceConfig":
        return cls(RuntimeBackend.IN_MEMORY, namespace, deployment_id)

    @classmethod
    def filesystem(cls, root: str, *, workspace_id: str) -> "RuntimePersistenceConfig":
        if not root.strip() or not workspace_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls(RuntimeBackend.FILESYSTEM, workspace_id, "workspace", str(Path(root).expanduser().resolve()), workspace_id)

    @classmethod
    def sqlite(cls, path: str, *, namespace: str, deployment_id: str) -> "RuntimePersistenceConfig":
        if not path.strip() or path == ":memory:":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls(RuntimeBackend.SQLITE, namespace, deployment_id, str(Path(path).expanduser().resolve()))

    @classmethod
    def mysql(cls, *, namespace: str, deployment_id: str) -> "RuntimePersistenceConfig":
        return cls(RuntimeBackend.MYSQL, namespace, deployment_id)

    @classmethod
    def postgresql(cls, *, namespace: str, deployment_id: str) -> "RuntimePersistenceConfig":
        return cls(RuntimeBackend.POSTGRESQL, namespace, deployment_id)

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.deployment_id.strip() or _has_control(self.namespace) or _has_control(self.deployment_id):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.backend is RuntimeBackend.IN_MEMORY:
            if self.location is not None or self.local_tenant_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.FILESYSTEM:
            if self.location is None or not Path(self.location).is_absolute() or not self.local_tenant_id or _has_control(self.local_tenant_id):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.SQLITE:
            if self.location is None or self.location == ":memory:" or not Path(self.location).is_absolute() or self.local_tenant_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend in {RuntimeBackend.MYSQL, RuntimeBackend.POSTGRESQL} and (self.location is not None or self.local_tenant_id is not None):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class _RuntimeResources:
    backend: RuntimeBackend
    namespace: str
    domain: RuntimePersistence
    steps: StepStore


@asynccontextmanager
async def _open_resources(
    config: RuntimePersistenceConfig,
    *,
    session_factory: "async_sessionmaker[AsyncSession] | None" = None,
) -> AsyncIterator[_RuntimeResources]:
    if config.backend is RuntimeBackend.IN_MEMORY:
        runtime = build_in_memory_runtime(namespace=config.namespace)
        steps = InMemoryStepStore()
        await runtime.initialize()
        try:
            yield _RuntimeResources(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await runtime.close()
        return
    if config.backend is RuntimeBackend.FILESYSTEM:
        runtime = build_filesystem_runtime(str(config.location), workspace_id=config.namespace)
        steps = DurableFilesystemStepStore(runtime.runtime_root, config.namespace, writer_lock=runtime.writer_lock)
        await runtime.initialize()
        await steps.initialize()
        try:
            yield _RuntimeResources(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await steps.close()
            await runtime.close()
        return
    if session_factory is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "SQL runtime requires a session factory")
    from ..storage import (
        SqlSchemaRegistry,
        build_sqlite_storage,
        build_storage,
        initialize_storage,
    )

    registry = SqlSchemaRegistry()
    tables = SqlRuntimeSchema.register_schema(registry)
    manifest = registry.freeze()
    database = await (
        build_sqlite_storage(session_factory=session_factory, metadata=registry.metadata, schema_manifest_digest=manifest.digest)
        if config.backend is RuntimeBackend.SQLITE
        else build_storage(session_factory=session_factory, metadata=registry.metadata, schema_manifest_digest=manifest.digest)
    )
    await initialize_storage(database)
    domain = await open_sql_runtime(
        database,
        session_factory=session_factory,
        backend=config.backend,
        namespace=config.namespace,
        deployment_id=config.deployment_id,
        tables=tables,
    )
    if config.backend is RuntimeBackend.SQLITE:
        steps = SqliteStepStore(database=namespace_scoped_step_db_path(str(config.location), config.namespace))
    else:
        steps = SqlStepStore(database, session_factory, config.namespace)
        await steps.initialize()
    try:
        await domain.sessions.initialize()
        yield _RuntimeResources(config.backend, config.namespace, domain, steps)
    finally:
        if config.backend is not RuntimeBackend.SQLITE:
            await steps.close()
        await domain.sessions.close()


def namespace_scoped_step_db_path(runtime_path: str, namespace: str) -> Path:
    path = Path(runtime_path).expanduser().resolve()
    return path.with_name(f"{path.name}.steps.{hashlib.sha256(namespace.encode('utf-8')).hexdigest()}.db")


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


@asynccontextmanager
async def open_workspace_runtime(
    workspace: Workspace,
    *,
    config: RuntimePersistenceConfig | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    session_factory: "async_sessionmaker[AsyncSession] | None" = None,
    mcp_runtime: "MCPRuntimeProvider | None" = None,
    extra_asset_bindings: "Sequence[AssetTypeBinding[object]]" = (),
    capability_provider_factories: "Sequence[Callable[[AssetRepository], CapabilityProvider]]" = (),
    capability_grants: "Sequence[CapabilityBinding]" = (),
    task_node_runner: "TaskNodeRunner | None" = None,
) -> AsyncIterator[Runtime]:
    persistence_config = config or RuntimePersistenceConfig.filesystem(str(workspace.root), workspace_id=workspace.workspace_id)
    registry = _build_asset_registry(extra_asset_bindings)
    phase_a_store, phase_a_assets = await _create_workspace_repository(workspace.root / ".linktools", registry)
    phase_a_providers = _build_capability_providers(phase_a_assets, mcp_runtime, capability_provider_factories)
    phase_a_refs = await _bootstrap_refs(phase_a_providers)
    _logger.debug("workspace capability bootstrap phase A complete: workspace=%s refs=%s", workspace.workspace_id, len(phase_a_refs))
    del phase_a_store
    final_store, assets = await _create_workspace_repository(workspace.root / ".linktools", registry, generated_refs=phase_a_refs)
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
    async with _open_resources(persistence_config, session_factory=session_factory) as resources:
        definitions: dict[str, AgentDefinition] = {}
        executor = AgentExecutor(materializer, execution_root=workspace.root)
        backend = LocalExecutionBackend(
            resources.domain,
            resources.steps,
            executor,
            definitions,
            execution_root=workspace.root,
            memory_store_factory=lambda tenant_id, namespace: RuntimeMemoryStore(resources.domain, tenant_id=tenant_id, namespace=namespace),
        )
        history = StepExecutionHistoryReader(resources.namespace, resources.domain, resources.steps, HmacCursorSigner("execution-history", _grant_key(workspace)))
        authorization = TenantAuthorizationPolicy()
        execution = DefaultExecutionService(resources.domain, authorization, backend=backend, history_reader=history)
        session = DefaultSessionService(resources.domain, authorization, execution, HmacCursorSigner("session", _grant_key(workspace)))
        task_runner = task_node_runner or RuntimeTaskNodeRunner(execution, definitions)
        task_launcher = LocalTaskGraphLauncher(
            resources.domain.tasks,
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
        await backend.reconcile()
        _logger.debug("workspace runtime opened: workspace=%s backend=%s", workspace.workspace_id, persistence_config.backend)
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


async def _create_workspace_repository(
    root: Path,
    registry: AssetTypeRegistrySnapshot,
    *,
    generated_refs: "tuple[AgentCapabilityRef, ...] | None" = None,
) -> "tuple[AssetStore, AssetRepository]":
    local = LocalDirectoryAssetBackend(
        str(root),
        writable=False,
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
    )
    defaults = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(local, writer=defaults, layers=(StorageLayer("defaults", defaults),)))
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


__all__ = ["RuntimePersistenceConfig", "build_asset_store", "build_workspace_asset_repository", "namespace_scoped_step_db_path", "open_workspace_runtime"]
