#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from linktools.core import environ
from pydantic_ai_harness.step_persistence import InMemoryStepStore, SqliteStepStore, StepStore

from ..adapter import (
    DurableFilesystemStepStore,
    SqlRuntimeSchema,
    SqlStepStore,
    StepExecutionHistoryReader,
    build_filesystem_runtime,
    build_in_memory_runtime,
    open_sql_runtime,
)
from ..adapter import RuntimeMemoryStore
from ..agent import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AgentCompiler,
    AgentDefinition,
    AgentExecutor,
    AssistantTextOutput,
    OutputTypeRegistry,
    build_asset_capability_providers,
)
from ..asset import (
    AssetKey,
    AssetRef,
    AssetRepository,
    AssetStore,
    AssetTypeBinding,
    AssetTypeRegistry,
    AssetVariantBinding,
    DirectoryLayout,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
    PrefixAssetPathAdapter,
    SingleFileLayout,
)
from ..capability import MCPRuntimeProvider, PydanticMCPRuntimeProvider
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
    Runtime,
    RuntimeBackend,
    RuntimePersistence,
    RuntimeTaskNodeRunner,
)
from ..runtime import LocalExecutionBackend
from ..spec import (
    AgentCapabilityRef,
    AgentSpec,
    AgentSpecCodec,
    MCPServerSpec,
    MCPServerSpecCodec,
    PromptSpec,
    PromptSpecCodec,
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)
from ..storage import StorageLayer, StorageOverlay
from ..task import LocalTaskGraphLauncher, RuntimeTaskNodeResultVerifier
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
    from ..storage import SqlSchemaRegistry, build_sqlite_storage, build_storage, initialize_storage

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
    """Build the read-only workspace asset layer plus built-in defaults."""
    workspace_root = Path(root).expanduser().resolve()
    local = LocalDirectoryAssetBackend(
        str(workspace_root),
        writable=False,
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
    )
    defaults = InMemoryAssetBackend()
    await _put_default_assets(defaults, _discover_workspace_capability_refs(workspace_root))
    return AssetStore(StorageOverlay(local, layers=(StorageLayer("defaults", defaults),)))


def build_asset_repository(store: AssetStore) -> AssetRepository:
    registry = AssetTypeRegistry()
    for binding in _asset_bindings():
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
) -> AsyncIterator[Runtime]:
    persistence_config = config or RuntimePersistenceConfig.filesystem(str(workspace.root), workspace_id=workspace.workspace_id)
    asset_store = await build_asset_store(workspace.root / ".linktools")
    await asset_store.initialize()
    assets = build_asset_repository(asset_store)
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
    grants = build_workspace_capability_grants(workspace.root)
    compiler = AgentCompiler(
        assets,
        model_resolver=resolver,
        model_connections=connections,
        output_types=output_types,
        capability_providers=build_asset_capability_providers(assets, mcp_runtime=mcp_runtime or PydanticMCPRuntimeProvider(execution_root=workspace.root)),
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
        task_runner = RuntimeTaskNodeRunner(execution, definitions)
        task_launcher = LocalTaskGraphLauncher(
            resources.domain.tasks,
            task_runner,
            RuntimeTaskNodeResultVerifier(resources.domain.executions),
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


async def _put_default_assets(
    backend: InMemoryAssetBackend,
    capabilities: tuple[AgentCapabilityRef, ...],
) -> None:
    await backend.put(
        AssetKey("agent", "default"),
        AgentSpecCodec().encode(AgentSpec("default", 1, "default", capabilities, "assistant-text", 1)),
    )
    await backend.put(AssetKey("prompt", "default"), PromptSpecCodec().encode(PromptSpec("default", 1, "", ())))


def _discover_workspace_capability_refs(root: Path) -> tuple[AgentCapabilityRef, ...]:
    refs: list[AgentCapabilityRef] = []
    refs.extend(AgentCapabilityRef("mcp", server_id, required=False) for server_id in _discover_mcp_ids(root / "mcp"))
    return tuple(refs)


def _discover_mcp_ids(mcp_root: Path) -> tuple[str, ...]:
    if not mcp_root.is_dir():
        return ()
    codec = MCPServerSpecCodec()
    server_ids: list[str] = []
    for path in sorted(item for item in mcp_root.rglob("*") if item.is_file()):
        server_id = path.relative_to(mcp_root).as_posix()
        try:
            AssetRef("mcp", server_id)
            specification = codec.decode(path.read_bytes())
        except (AIError, OSError, ValueError) as error:
            _logger.debug("workspace mcp skipped: path=%s reason=%s", path, type(error).__name__)
            continue
        if specification.id != server_id:
            _logger.debug("workspace mcp skipped: path=%s reason=declaration_id_mismatch", path)
            continue
        server_ids.append(server_id)
    return tuple(server_ids)


def _configured_model(config: dict[str, object]) -> str | None:
    value = config.get("model")
    return value if isinstance(value, str) and value.strip() else None


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


def _asset_bindings() -> tuple[AssetTypeBinding[object], ...]:
    def identity(ref: AssetRef, value: object) -> bool:
        return isinstance(value, (AgentSpec, PromptSpec, MCPServerSpec, SkillSpec)) and value.id == ref.id

    return (
        cast("AssetTypeBinding[object]", AssetTypeBinding("agent", AgentSpec, (AssetVariantBinding("json", SingleFileLayout(""), AgentSpecCodec(), "agent-spec", 1),), "json", identity, "id-v1", True)),
        cast("AssetTypeBinding[object]", AssetTypeBinding("prompt", PromptSpec, (AssetVariantBinding("json", SingleFileLayout(""), PromptSpecCodec(), "prompt-spec", 1),), "json", identity, "id-v1", True)),
        cast("AssetTypeBinding[object]", AssetTypeBinding("mcp", MCPServerSpec, (AssetVariantBinding("json", SingleFileLayout(""), MCPServerSpecCodec(), "mcp-spec", 1),), "json", identity, "id-v1", True)),
        cast("AssetTypeBinding[object]", AssetTypeBinding("skill", SkillSpec, (
            AssetVariantBinding("json", SingleFileLayout(""), SkillSpecCodec(), "skill-spec", 1),
            AssetVariantBinding("directory", DirectoryLayout("SKILL.md"), SkillMarkdownSpecCodec(), "skill-markdown", 1, SkillMarkdownSpecAdapter(), "skill-name-v1"),
        ), "directory", identity, "id-v1", True)),
    )


__all__ = ["RuntimePersistenceConfig", "build_asset_repository", "build_asset_store", "namespace_scoped_step_db_path", "open_workspace_runtime"]
