#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唯一 Composition Root for process-local Runtime services."""

import hashlib
from collections.abc import AsyncIterator
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
    SqlRuntimeSchema,
    SqlStepStore,
    StepExecutionHistoryReader,
    build_filesystem_runtime,
    build_in_memory_runtime,
    open_sql_runtime,
)
from ..agent import AgentBindingRegistry, AgentCatalogView, OutputTypeRegistry
from ..asset import AssetInfo, AssetKey, AssetStore
from ..capability import MCPToolProvider, SkillProvider
from ..core import AuthorizationPolicy, HmacCursorSigner, PrincipalProvider
from ..errors import AIError, ErrorCode
from ..model import ModelMaterializer, ModelRegistry, ModelResolver
from ..observe import MiddlewarePipeline
from ..runtime import (
    CancelEffectOutcome,
    DefaultApprovalService,
    DefaultArtifactService,
    DefaultEvaluationService,
    DefaultEventService,
    DefaultExecutionService,
    DefaultSessionService,
    DefaultTaskService,
    ExecutionHistoryReader,
    ExecutionLauncher,
    ExecutionRecord,
    ExecutionRequest,
    RuntimeBackend,
    RuntimePersistence,
    RuntimeServices,
    ToolPolicy,
    ToolStateStore,
    WorkflowGateway,
    WorkflowTaskGraphLauncher,
    new_runtime_service_identity,
)
from ..storage import StorageComposition
from ..task import (
    LocalTaskGraphLauncher,
    RuntimeTaskNodeResultVerifier,
    TaskGraphLauncher,
    TaskNodeResultVerifier,
    TaskNodeRunner,
)
from ..workspace import Sandbox
from ._binding_launcher import BindingExecutionLauncher
from ._facade import LocalRuntimeServices, RuntimeAccess, build_runtime_access

_logger = environ.get_logger("ai.app.assembly")

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from ..storage import StorageDatabase


@dataclass(frozen=True, slots=True)
class RuntimePersistenceConfig:
    backend: RuntimeBackend
    namespace: str
    deployment_id: str
    location: str | None = field(default=None, repr=False)
    local_tenant_id: str | None = None

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.deployment_id.strip() or _has_control(self.namespace) or _has_control(self.deployment_id):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.backend is RuntimeBackend.IN_MEMORY:
            if self.location or self.local_tenant_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.FILESYSTEM:
            if self.location is None or not Path(self.location).is_absolute() or not self.local_tenant_id or not self.local_tenant_id.strip() or _has_control(self.local_tenant_id):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.SQLITE:
            if self.location is None or self.location == ":memory:" or not Path(self.location).is_absolute():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.MYSQL:
            if self.location is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.POSTGRESQL:
            if self.location is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    @classmethod
    def in_memory(cls, *, namespace: str, deployment_id: str = "memory") -> "RuntimePersistenceConfig":
        return cls(backend=RuntimeBackend.IN_MEMORY, namespace=namespace, deployment_id=deployment_id)

    @classmethod
    def filesystem(cls, root: str, *, workspace_id: str) -> "RuntimePersistenceConfig":
        if not root.strip() or not workspace_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls(backend=RuntimeBackend.FILESYSTEM, namespace=workspace_id, deployment_id="workspace", location=str(Path(root).expanduser().resolve()), local_tenant_id=workspace_id)

    @classmethod
    def sqlite(cls, path: str, *, namespace: str, deployment_id: str) -> "RuntimePersistenceConfig":
        if not path.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls(backend=RuntimeBackend.SQLITE, namespace=namespace, deployment_id=deployment_id, location=str(Path(path).expanduser().resolve()))

    @classmethod
    def mysql(cls, *, namespace: str, deployment_id: str) -> "RuntimePersistenceConfig":
        return cls(backend=RuntimeBackend.MYSQL, namespace=namespace, deployment_id=deployment_id)

    @classmethod
    def postgresql(cls, *, namespace: str, deployment_id: str) -> "RuntimePersistenceConfig":
        return cls(backend=RuntimeBackend.POSTGRESQL, namespace=namespace, deployment_id=deployment_id)


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def namespace_scoped_step_db_path(runtime_path: str | Path, namespace: str) -> Path:
    path = Path(runtime_path).expanduser().resolve()
    return path.with_name(f"{path.name}.steps.{hashlib.sha256(namespace.encode('utf-8')).hexdigest()}.db")


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    backend: RuntimeBackend
    namespace: str
    domain: RuntimePersistence
    steps: StepStore


async def _open_sql_step_store(
    database: "StorageDatabase",
    session_factory: "async_sessionmaker[AsyncSession]",
    config: RuntimePersistenceConfig,
) -> StepStore:
    return SqlStepStore(database, session_factory, config.namespace)


def _validate_persistence_config(config: RuntimePersistenceConfig) -> None:
    config.__post_init__()


@asynccontextmanager
async def open_runtime_resources(
    config: RuntimePersistenceConfig,
    *,
    session_factory: "async_sessionmaker[AsyncSession] | None" = None,
) -> AsyncIterator[RuntimeResources]:
    _validate_persistence_config(config)
    if config.backend is RuntimeBackend.IN_MEMORY:
        runtime = build_in_memory_runtime(namespace=config.namespace)
        steps = InMemoryStepStore()
        await runtime.initialize()
        try:
            yield RuntimeResources(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await runtime.close()
        return
    if config.backend is RuntimeBackend.FILESYSTEM:
        runtime = build_filesystem_runtime(str(config.location), workspace_id=config.namespace)
        steps = DurableFilesystemStepStore(runtime.runtime_root, config.namespace, writer_lock=runtime.writer_lock)
        await runtime.initialize()
        await steps.initialize()
        try:
            yield RuntimeResources(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await steps.close()
            await runtime.close()
        return
    from ..storage import (
        SqlSchemaRegistry,
        build_sqlite_storage,
        build_storage,
        initialize_storage,
    )
    if session_factory is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "SQL runtime requires an injected session factory")
    _logger.debug("SQL runtime dependencies injected: backend=%s session_factory=%s", config.backend, type(session_factory).__name__)
    registry = SqlSchemaRegistry()
    tables = SqlRuntimeSchema.register_schema(registry)
    manifest = registry.freeze()
    database = (
        await build_sqlite_storage(
            session_factory=session_factory,
            metadata=registry.metadata,
            schema_manifest_digest=manifest.digest,
        )
        if config.backend is RuntimeBackend.SQLITE
        else build_storage(
            session_factory=session_factory,
            metadata=registry.metadata,
            schema_manifest_digest=manifest.digest,
        )
    )
    try:
        await initialize_storage(database)
        persistence = await open_sql_runtime(database, session_factory=session_factory, backend=config.backend, namespace=config.namespace, deployment_id=config.deployment_id, tables=tables)
        steps: StepStore = SqliteStepStore(database=namespace_scoped_step_db_path(str(config.location), config.namespace)) if config.backend is RuntimeBackend.SQLITE else await _open_sql_step_store(database, session_factory, config)
        if config.backend is not RuntimeBackend.SQLITE:
            await steps.initialize()
        await persistence.sessions.initialize()
        _logger.info("runtime resources opened: backend=%s namespace=%s atomic_domain_id=%s", config.backend, config.namespace, persistence.atomic_domain_id)
        try:
            yield RuntimeResources(config.backend, config.namespace, persistence, steps)
        finally:
            if config.backend is not RuntimeBackend.SQLITE:
                await steps.close()
            await persistence.sessions.close()
    finally:
        _logger.info("runtime resources released: backend=%s namespace=%s external SQL engine retained", config.backend, config.namespace)


@asynccontextmanager
async def open_runtime_services(
    config: RuntimePersistenceConfig,
    authorization: "AuthorizationPolicy",
    *,
    grant_key: bytes,
    workflow_gateway: "WorkflowGateway | None" = None,
    execution_launcher: "ExecutionLauncher | None" = None,
    task_launcher: "TaskGraphLauncher | None" = None,
    session_factory: "async_sessionmaker[AsyncSession] | None" = None,
) -> AsyncIterator[RuntimeServices]:
    async with open_runtime_resources(config, session_factory=session_factory) as resources:
        history_reader = StepExecutionHistoryReader(config.namespace, resources.domain, resources.steps, HmacCursorSigner("execution-history", grant_key))
        yield build_runtime_services(
            resources.domain,
            authorization,
            grant_key=grant_key,
            workflow_gateway=workflow_gateway,
            execution_launcher=execution_launcher,
            task_launcher=task_launcher,
            history_reader=history_reader,
            schema_digest=resources.domain.atomic_domain_id,
        )

@dataclass(frozen=True, slots=True)
class AppServices:
    runtime_services: RuntimeServices
    access: RuntimeAccess
    principal_provider: "PrincipalProvider | None" = None


class _WorkflowExecutionLauncher:
    def __init__(self, gateway: WorkflowGateway) -> None:
        self._gateway = gateway

    async def start(self, request: ExecutionRequest, execution: "ExecutionRecord") -> None:
        await self._gateway.start_execution(execution.execution_id, request)

    async def cancel(self, execution: "ExecutionRecord") -> "CancelEffectOutcome":
        result = await self._gateway.cancel_execution(execution.execution_id)
        return CancelEffectOutcome.CONFIRMED if result.cancelled else CancelEffectOutcome.UNKNOWN


def build_asset_store(
    storage: "StorageComposition[AssetKey, bytes, AssetInfo]",
) -> AssetStore:
    return AssetStore(storage)


def build_app_services(
    asset_store: AssetStore,
    model_registry: ModelRegistry,
    model_resolver: ModelResolver,
    skill_provider: SkillProvider,
    mcp_provider: MCPToolProvider,
    middleware: MiddlewarePipeline,
    sandbox: Sandbox,
    tool_policy: ToolPolicy,
    output_types: OutputTypeRegistry,
    tool_state: ToolStateStore,
    principal_provider: PrincipalProvider,
    runtime_services: RuntimeServices,
) -> AppServices:
    if any(
        value is None
        for value in (
            asset_store,
            model_registry,
            model_resolver,
            skill_provider,
            mcp_provider,
            middleware,
            sandbox,
            tool_policy,
            output_types,
            tool_state,
            principal_provider,
            runtime_services,
        )
    ):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    registry_snapshot = model_registry.snapshot()
    resolver_snapshot = model_resolver.snapshot()
    if (
        resolver_snapshot.revision != registry_snapshot.revision
        or resolver_snapshot.digest != registry_snapshot.digest
    ):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not output_types.frozen:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not asset_store.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not skill_provider.manifest() or not mcp_provider.manifest():
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    services = AppServices(runtime_services, build_runtime_access(runtime_services), principal_provider)
    _logger.info("agent services composed: model_revision=%s", model_registry.snapshot().revision)
    return services


def build_runtime_services(
    persistence: RuntimePersistence,
    authorization: "AuthorizationPolicy",
    *,
    grant_key: bytes,
    history_reader: ExecutionHistoryReader,
    schema_digest: str,
    workflow_gateway: "WorkflowGateway | None" = None,
    execution_launcher: "ExecutionLauncher | None" = None,
    task_launcher: "TaskGraphLauncher | None" = None,
) -> RuntimeServices:
    """Compose all default services from one persistence and authorization root."""
    identity = new_runtime_service_identity(
        backend=persistence.backend,
        namespace=persistence.namespace,
        atomic_domain_id=persistence.atomic_domain_id,
        schema_digest=schema_digest,
    )
    if execution_launcher is not None:
        launcher = execution_launcher
        execution_gateway = None
    elif workflow_gateway is not None:
        launcher = _WorkflowExecutionLauncher(workflow_gateway)
        execution_gateway = workflow_gateway
    else:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    execution = DefaultExecutionService(persistence, authorization, launcher=launcher, history_reader=history_reader)
    graph_launcher = task_launcher if task_launcher is not None else WorkflowTaskGraphLauncher(workflow_gateway) if workflow_gateway is not None else None
    services = RuntimeServices(
        identity,
        execution,
        DefaultSessionService(persistence, authorization, execution, HmacCursorSigner("session", grant_key)),
        DefaultTaskService(persistence, authorization, graph_launcher),
        DefaultEvaluationService(persistence, authorization, execution),
        DefaultApprovalService(persistence, authorization, execution_gateway),
        DefaultEventService(persistence, authorization),
        DefaultArtifactService(persistence, authorization, grant_key=grant_key, cursor_signer=HmacCursorSigner("artifact", grant_key)),
    )
    _logger.info("runtime services composed: backend=%s namespace=%s launcher=%s gateway=%s", persistence.backend, persistence.namespace, type(launcher).__name__, workflow_gateway is not None)
    return services


def build_local_runtime_services(
    resources: RuntimeResources,
    authorization: "AuthorizationPolicy",
    *,
    grant_key: bytes,
    materializer: ModelMaterializer,
    agent_catalog: "AgentCatalogView | None" = None,
    task_launcher: "TaskGraphLauncher | None" = None,
    task_runner: "TaskNodeRunner | None" = None,
    task_result_verifier: "TaskNodeResultVerifier | None" = None,
    task_owner: str = "local",
) -> LocalRuntimeServices:
    """Create local services and their one shared binding registry."""
    registry = AgentBindingRegistry()
    launcher = BindingExecutionLauncher(
        registry,
        materializer,
        resources,
        agent_catalog=agent_catalog,
    )
    history_reader = StepExecutionHistoryReader(
        resources.namespace,
        resources.domain,
        resources.steps,
        HmacCursorSigner("execution-history", grant_key),
    )
    if task_launcher is None and task_runner is None and task_result_verifier is not None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "task result verifier requires a task runner")
    if task_launcher is None and task_runner is not None:
        task_launcher = LocalTaskGraphLauncher(
            resources.domain.tasks,
            task_runner,
            task_result_verifier or RuntimeTaskNodeResultVerifier(resources.domain.executions),
            owner=task_owner,
        )
    services = build_runtime_services(
        resources.domain,
        authorization,
        grant_key=grant_key,
        history_reader=history_reader,
        schema_digest=resources.domain.atomic_domain_id,
        execution_launcher=launcher,
        task_launcher=task_launcher,
    )
    _logger.info(
        "local runtime composition ready: namespace=%s registry=%s launcher=%s",
        resources.namespace,
        id(registry),
        type(launcher).__name__,
    )
    return LocalRuntimeServices.compose(services, registry)


__all__ = ["AppServices", "RuntimePersistenceConfig", "RuntimeResources", "build_app_services", "build_asset_store", "build_local_runtime_services", "build_runtime_services", "namespace_scoped_step_db_path", "open_runtime_resources", "open_runtime_services"]
