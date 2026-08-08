#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唯一 Composition Root for process-local Runtime services."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from typing import Protocol

from linktools.core import environ

from ..asset import AssetCodecRegistry, AssetStore
from ..adapter import DurableFileStepStore, SqlRuntimeSchema, SqlStepStore, StepExecutionHistoryReader, build_file_runtime, build_memory_runtime, open_sql_runtime
from ..capability import MCPToolProvider, Sandbox, SkillProvider, SubagentProvider, ToolPolicy, ToolStateStore
from ..model import ModelRegistry, ModelResolver
from ..observe import MiddlewarePipeline
from ..runtime import (
    DefaultApprovalService,
    DefaultArtifactService,
    DefaultEventService,
    DefaultEvaluationService,
    DefaultExecutionService,
    DefaultSessionService,
    DefaultTaskService,
    RuntimeServices,
)
from .facade import Runtime, RuntimeAccess, build_runtime_access
from ..runtime.execution import CancelEffectOutcome, ExecutionLauncher
from pydantic_ai_harness.step_persistence import InMemoryStepStore, SqliteStepStore, StepStore

from ..runtime.persistence import RuntimeBackend, RuntimePersistence
from ..runtime.persistence import ExecutionRecord
from ..runtime.services import ExecutionHistoryReader, ExecutionRequest, WorkflowGateway, new_runtime_service_identity
from ..core import AuthorizationPolicy, HmacCursorSigner, PrincipalProvider
from ..core.errors import ErrorCode, AIError
from ..spec import AgentSpecCodec, PromptSpecCodec
from ..capability import MCPServerSpecCodec, SkillSpecCodec
from ..spec.output import OutputTypeRegistry

_logger = environ.get_logger("ai.app.assembly")


@dataclass(frozen=True, slots=True)
class RuntimeStoreConfig:
    backend: RuntimeBackend
    namespace: str
    deployment_id: str
    location: str | None = field(default=None, repr=False)
    local_tenant_id: str | None = None

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.deployment_id.strip() or _has_control(self.namespace) or _has_control(self.deployment_id):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.backend is RuntimeBackend.MEMORY:
            if self.location or self.local_tenant_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.FILE:
            if self.location is None or not Path(self.location).is_absolute() or not self.local_tenant_id or not self.local_tenant_id.strip() or _has_control(self.local_tenant_id):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.SQLITE:
            if self.location is None or self.location == ":memory:" or not Path(self.location).is_absolute():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.backend is RuntimeBackend.MYSQL:
            _validate_url(self.location or "", "mysql+asyncmy")
        elif self.backend is RuntimeBackend.POSTGRESQL:
            _validate_url(self.location or "", "postgresql+asyncpg")

    @classmethod
    def memory(cls, *, namespace: str, deployment_id: str = "memory") -> "RuntimeStoreConfig":
        return cls(backend=RuntimeBackend.MEMORY, namespace=namespace, deployment_id=deployment_id)

    @classmethod
    def file(cls, root: str, *, workspace_id: str) -> "RuntimeStoreConfig":
        if not root.strip() or not workspace_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls(backend=RuntimeBackend.FILE, namespace=workspace_id, deployment_id="workspace", location=str(Path(root).expanduser().resolve()), local_tenant_id=workspace_id)

    @classmethod
    def sqlite(cls, path: str, *, namespace: str, deployment_id: str) -> "RuntimeStoreConfig":
        if not path.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return cls(backend=RuntimeBackend.SQLITE, namespace=namespace, deployment_id=deployment_id, location=str(Path(path).expanduser().resolve()))

    @classmethod
    def mysql(cls, url: str, *, namespace: str, deployment_id: str) -> "RuntimeStoreConfig":
        _validate_url(url, "mysql+asyncmy")
        return cls(backend=RuntimeBackend.MYSQL, namespace=namespace, deployment_id=deployment_id, location=url)

    @classmethod
    def postgresql(cls, url: str, *, namespace: str, deployment_id: str) -> "RuntimeStoreConfig":
        _validate_url(url, "postgresql+asyncpg")
        return cls(backend=RuntimeBackend.POSTGRESQL, namespace=namespace, deployment_id=deployment_id, location=url)


def _validate_url(url: str, scheme: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != scheme or not parsed.hostname:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def namespace_scoped_step_db_path(runtime_path: str | Path, namespace: str) -> Path:
    path = Path(runtime_path).expanduser().resolve()
    return path.with_name(f"{path.name}.steps.{hashlib.sha256(namespace.encode('utf-8')).hexdigest()}.db")


@dataclass(frozen=True, slots=True)
class RuntimeStores:
    backend: RuntimeBackend
    namespace: str
    domain: RuntimePersistence
    steps: StepStore


async def _open_sql_step_store(database: object, config: RuntimeStoreConfig) -> StepStore:
    return SqlStepStore(database, config.namespace)


def _validate_store_config(config: RuntimeStoreConfig) -> None:
    config.__post_init__()


@asynccontextmanager
async def open_runtime_store(config: RuntimeStoreConfig) -> AsyncIterator[RuntimeStores]:
    _validate_store_config(config)
    if config.backend is RuntimeBackend.MEMORY:
        runtime = build_memory_runtime(namespace=config.namespace)
        steps = InMemoryStepStore()
        await runtime.initialize()
        try:
            yield RuntimeStores(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await runtime.close()
        return
    if config.backend is RuntimeBackend.FILE:
        runtime = build_file_runtime(str(config.location), workspace_id=config.namespace)
        steps = DurableFileStepStore(str(config.location), config.namespace, writer_lock=runtime.writer_lock)
        await runtime.initialize()
        await steps.initialize()
        try:
            yield RuntimeStores(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await steps.close()
            await runtime.close()
        return
    from ..storage import SqlSchemaRegistry, build_sqlite_storage, build_storage, close_storage, initialize_storage
    registry = SqlSchemaRegistry()
    tables = SqlRuntimeSchema.register_schema(registry)
    manifest = registry.freeze()
    database = build_sqlite_storage(Path(str(config.location)), metadata=registry.metadata, schema_manifest_digest=manifest.digest) if config.backend is RuntimeBackend.SQLITE else build_storage(str(config.location), metadata=registry.metadata, schema_manifest_digest=manifest.digest)
    try:
        await initialize_storage(database)
        persistence = await open_sql_runtime(database, backend=config.backend, namespace=config.namespace, deployment_id=config.deployment_id, tables=tables)
        steps: StepStore = SqliteStepStore(database=namespace_scoped_step_db_path(str(config.location), config.namespace)) if config.backend is RuntimeBackend.SQLITE else await _open_sql_step_store(database, config)
        if config.backend is not RuntimeBackend.SQLITE:
            await steps.initialize()
        await persistence.sessions.initialize()
        _logger.info("runtime store opened: backend=%s namespace=%s atomic_domain_id=%s", config.backend, config.namespace, persistence.atomic_domain_id)
        try:
            yield RuntimeStores(config.backend, config.namespace, persistence, steps)
        finally:
            if config.backend is not RuntimeBackend.SQLITE:
                await steps.close()
            await persistence.sessions.close()
    finally:
        await close_storage(database)


@asynccontextmanager
async def open_runtime_services(
    config: RuntimeStoreConfig,
    authorization: "AuthorizationPolicy",
    *,
    grant_key: bytes,
    workflow_gateway: "WorkflowGateway | None" = None,
    execution_launcher: "ExecutionLauncher | None" = None,
) -> AsyncIterator[RuntimeServices]:
    async with open_runtime_store(config) as stores:
        history_reader = StepExecutionHistoryReader(config.namespace, stores.domain, stores.steps)
        yield build_runtime_services(
            stores.domain,
            authorization,
            grant_key=grant_key,
            workflow_gateway=workflow_gateway,
            execution_launcher=execution_launcher,
            history_reader=history_reader,
            schema_digest=stores.domain.atomic_domain_id,
        )

@dataclass(frozen=True, slots=True)
class AppServices:
    runtime_services: RuntimeServices
    access: RuntimeAccess
    runtime_factory: "RuntimeFactory"
    principal_provider: "PrincipalProvider | None" = None


class RuntimeFactory(Protocol):
    async def build_for_request(self, request: ExecutionRequest) -> Runtime: ...


class _WorkflowExecutionLauncher:
    def __init__(self, gateway: WorkflowGateway) -> None:
        self._gateway = gateway

    async def start(self, request: ExecutionRequest, execution: "ExecutionRecord") -> None:
        await self._gateway.start_execution(execution.execution_id, request)

    async def cancel(self, execution: "ExecutionRecord") -> "CancelEffectOutcome":
        result = await self._gateway.cancel_execution(execution.execution_id)
        return CancelEffectOutcome.CONFIRMED if result.cancelled else CancelEffectOutcome.UNKNOWN


def build_asset_codecs() -> AssetCodecRegistry:
    registry = AssetCodecRegistry()
    registry.register(AgentSpecCodec())
    registry.register(PromptSpecCodec())
    registry.register(SkillSpecCodec())
    registry.register(MCPServerSpecCodec())
    manifest = registry.freeze()
    _logger.info("asset codecs frozen: entries=%s digest=%s", len(manifest.entries), manifest.digest)
    return registry


def build_app_services(
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
) -> AppServices:
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
    if not asset_store.codec_manifest.entries or not asset_store.codec_manifest.digest:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not skill_provider.manifest() or not mcp_provider.manifest() or not subagent_provider.manifest():
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    services = AppServices(runtime_services, build_runtime_access(runtime_services), _MissingRuntimeFactory(), principal_provider)
    _logger.info("agent services composed: model_revision=%s", model_registry.snapshot().revision)
    return services


class _MissingRuntimeFactory:
    async def build_for_request(self, request: ExecutionRequest) -> Runtime:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


def build_runtime_services(
    persistence: RuntimePersistence,
    authorization: "AuthorizationPolicy",
    *,
    grant_key: bytes,
    history_reader: ExecutionHistoryReader,
    schema_digest: str,
    workflow_gateway: "WorkflowGateway | None" = None,
    execution_launcher: "ExecutionLauncher | None" = None,
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
    services = RuntimeServices(
        identity,
        execution,
        DefaultSessionService(persistence, authorization, execution, HmacCursorSigner("session", grant_key)),
        DefaultTaskService(persistence, authorization, workflow_gateway),
        DefaultEvaluationService(persistence, authorization, execution),
        DefaultApprovalService(persistence, authorization, execution_gateway),
        DefaultEventService(persistence, authorization),
        DefaultArtifactService(persistence, authorization, grant_key=grant_key, cursor_signer=HmacCursorSigner("artifact", grant_key)),
    )
    _logger.info("runtime services composed: backend=%s namespace=%s launcher=%s gateway=%s", persistence.backend, persistence.namespace, type(launcher).__name__, workflow_gateway is not None)
    return services


__all__ = ["AppServices", "RuntimeFactory", "RuntimeStoreConfig", "RuntimeStores", "build_app_services", "build_asset_codecs", "build_runtime_services", "namespace_scoped_step_db_path", "open_runtime_services", "open_runtime_store"]
