#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""唯一 Composition Root for process-local Runtime services."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit
from typing import Protocol

from linktools.core import environ

from ..asset import AssetCodecRegistry, AssetStore
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
from .runtime import Runtime, RuntimeAccess, build_runtime_access
from ..runtime.execution import CancelEffectOutcome, ExecutionLauncher
from pydantic_ai.messages import ModelRequest, ModelResponse, TextContent, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import InMemoryStepStore, SqliteStepStore, RunRecord, StepEvent, StepStore

from ..runtime.persistence import RuntimeBackend, RuntimePersistence
from ..runtime.persistence import validate_runtime_profile
from ..runtime.persistence import ExecutionRecord
from ..runtime.services import ExecutionHistoryReader, ExecutionRequest, TraceItem, TranscriptItem, WorkflowGateway, new_runtime_service_identity
from ..core import AuthorizationPolicy, ExecutionProfile, HmacCursorSigner, Page, PrincipalProvider
from ..core.ids import step_conversation_id, step_run_id
from ..core.errors import ErrorCode, AIError
from ..storage.lock import FileWriterLock
from ..spec import AgentSpecCodec, PromptSpecCodec
from ..capability.codec import MCPServerSpecCodec, SkillSpecCodec
from ..spec.output import OutputTypeRegistry

_logger = environ.get_logger("ai.app.services")


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


class StepExecutionHistoryReader:
    def __init__(self, namespace: str, persistence: RuntimePersistence, store: StepStore) -> None:
        self._namespace = namespace
        self._persistence = persistence
        self._store = store

    async def trace(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[TraceItem]:
        record = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        entries = await self._history_tree(record, tenant_id)
        projected: list[tuple[tuple[object, ...], TraceItem]] = []
        for item, depth in entries:
            for segment_sequence, events in await self._segment_events(item, tenant_id):
                for ordinal, event in enumerate(events):
                    mapped = _trace_item(item, segment_sequence, depth, ordinal, event)
                    if mapped is not None:
                        projected.append(((_event_timestamp(event), depth, item.execution_id, segment_sequence, ordinal, mapped.payload.get("kind", "")), mapped))
        projected.sort(key=lambda value: value[0])
        values = [item for _, item in projected]
        start = _cursor_offset(cursor, len(values))
        selected = tuple(TraceItem(item.execution_id, start + index + 1, item.payload) for index, item in enumerate(values[start:start + limit]))
        next_offset = start + len(selected)
        return Page(selected, str(next_offset) if next_offset < len(values) else None)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[TranscriptItem]:
        if not 1 <= limit <= 200:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        record = await self._persistence.executions.get(execution_id, tenant_id=tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if record.agent_run_sequence == 0:
            return Page((), None)
        await self._history_tree(record, tenant_id)
        final_run_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id, segment_sequence=record.agent_run_sequence)
        snapshot = await self._store.latest_snapshot(run_id=final_run_id)
        if snapshot is None:
            if record.status.value == "SUCCEEDED":
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            return Page((), None)
        conversation_id = step_conversation_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id)
        values: list[str] = []
        for message in snapshot.messages:
            if isinstance(message, (ModelRequest, ModelResponse)):
                if message.conversation_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if message.conversation_id != conversation_id:
                    continue
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        values.extend(_user_text(part))
            elif isinstance(message, ModelResponse):
                values.extend(part.content for part in message.parts if isinstance(part, TextPart) and part.content)
        start = _cursor_offset(cursor, len(values))
        selected = tuple(TranscriptItem(execution_id, start + index + 1, value) for index, value in enumerate(values[start:start + limit]))
        next_offset = start + len(selected)
        return Page(selected, str(next_offset) if next_offset < len(values) else None)

    async def _history_tree(self, root: ExecutionRecord, tenant_id: str) -> list[tuple[ExecutionRecord, int]]:
        result: list[tuple[ExecutionRecord, int]] = []
        visited: set[str] = set()

        async def visit(record: ExecutionRecord, depth: int) -> None:
            if record.execution_id in visited or depth > 8:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.tenant_id != tenant_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if depth == 0:
                if record.parent_execution_id is not None or record.lineage_kind.value not in {"RUN", "RETRY", "FORK", "SESSION_RESUME"}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                if record.lineage_kind != "SUBAGENT" or record.root_execution_id != root.root_execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(record.execution_id)
            result.append((record, depth))
            children = await self._persistence.executions.list_children(record.execution_id, tenant_id=tenant_id)
            for child in children:
                await visit(child, depth + 1)

        await visit(root, 0)
        return result

    async def _segment_events(self, record: ExecutionRecord, tenant_id: str) -> list[tuple[int, list[StepEvent]]]:
        if record.agent_run_sequence < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        conversation_id = step_conversation_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=record.execution_id)
        terminal = record.status.value in {"SUCCEEDED", "FAILED", "CANCELLED"}
        result: list[tuple[int, list[StepEvent]]] = []
        for sequence in range(1, record.agent_run_sequence + 1):
            deterministic_id = step_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=record.execution_id, segment_sequence=sequence)
            run = await self._store.get_run(run_id=deterministic_id)
            if run is None:
                if not terminal and sequence == record.agent_run_sequence:
                    continue
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _validate_run(run, deterministic_id, conversation_id, sequence)
            events = await self._store.list_events(run_id=deterministic_id)
            for event in events:
                if event.run_id != deterministic_id or event.conversation_id not in {None, conversation_id}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result.append((sequence, events))
        return result


def _trace_item(record: ExecutionRecord, segment_sequence: int, depth: int, ordinal: int, event: StepEvent) -> TraceItem | None:
    mapping = {
        "model_request_started": ("MODEL_REQUEST", "STARTED"),
        "model_request_completed": ("MODEL_RESPONSE", "SUCCEEDED"),
        "model_request_failed": ("MODEL_RESPONSE", "FAILED"),
        "tool_call_started": ("TOOL_CALL", "STARTED"),
        "tool_call_completed": ("TOOL_RESULT", "SUCCEEDED"),
        "tool_call_failed": ("TOOL_ERROR", "FAILED"),
    }
    value = mapping.get(event.kind)
    if value is None:
        return None
    kind, status = value
    payload = {"kind": kind, "status": status, "step_index": event.step_index, "segment_sequence": segment_sequence, "scope": "root" if depth == 0 else "subagent", "depth": depth}
    if event.agent_name is not None:
        payload["agent_name"] = event.agent_name
    if event.tool_call_id is not None:
        payload["tool_call_id"] = event.tool_call_id
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if depth > 0:
        payload["child_execution_id"] = record.execution_id
    return TraceItem(record.execution_id, ordinal, payload)


def _validate_run(run: RunRecord, expected_id: str, conversation_id: str, sequence: int) -> None:
    if run.run_id != expected_id or run.conversation_id != conversation_id or run.metadata.get("segment_sequence") != str(sequence):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    agent_name = run.metadata.get("agent_name")
    if agent_name is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", agent_name) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _event_timestamp(event: StepEvent) -> datetime:
    return event.timestamp.astimezone(timezone.utc)


def _cursor_offset(cursor: str | None, size: int) -> int:
    if cursor is None:
        return 0
    try:
        offset = int(cursor)
    except ValueError as error:
        raise AIError(ErrorCode.PAGE_CURSOR_INVALID) from error
    if offset < 0 or offset > size:
        raise AIError(ErrorCode.PAGE_CURSOR_INVALID)
    return offset


def _user_text(part: UserPromptPart) -> list[str]:
    if isinstance(part.content, str):
        return [part.content] if part.content else []
    values: list[str] = []
    for item in part.content:
        if isinstance(item, str) and item:
            values.append(item)
        elif isinstance(item, TextContent) and item.content:
            values.append(item.content)
    return values


async def _open_sql_step_store(database: object, config: RuntimeStoreConfig) -> StepStore:
    from ..adapter.step import SqlStepStore
    return SqlStepStore(database, config.namespace)


def _validate_store_config(config: RuntimeStoreConfig) -> None:
    config.__post_init__()


@asynccontextmanager
async def open_runtime_store(config: RuntimeStoreConfig) -> AsyncIterator[RuntimeStores]:
    _validate_store_config(config)
    if config.backend is RuntimeBackend.MEMORY:
        from ..adapter.memory import build_memory_runtime
        runtime = build_memory_runtime(namespace=config.namespace)
        steps = InMemoryStepStore()
        await runtime.initialize()
        try:
            yield RuntimeStores(config.backend, config.namespace, runtime.persistence, steps)
        finally:
            await runtime.close()
        return
    if config.backend is RuntimeBackend.FILE:
        from ..adapter.memory import build_file_runtime
        from ..adapter.step import DurableFileStepStore
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
    from ..adapter.schema import SqlRuntimeSchema
    from ..adapter.sql import open_sql_runtime
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
    profile: "ExecutionProfile",
    temporal_enabled: bool,
    grant_key: bytes,
    workflow_gateway: "WorkflowGateway | None" = None,
    execution_launcher: "ExecutionLauncher | None" = None,
) -> AsyncIterator[RuntimeServices]:
    validate_runtime_profile(config.backend, profile, temporal_enabled=temporal_enabled)
    local_lock = None
    if config.backend is RuntimeBackend.SQLITE and profile is ExecutionProfile.LOCAL_CODING:
        path = Path(str(config.location))
        local_lock = FileWriterLock(path.with_name(f"{path.name}.local.lock"))
        await local_lock.acquire()
    try:
        async with open_runtime_store(config) as stores:
            history_reader = StepExecutionHistoryReader(config.namespace, stores.domain, stores.steps)
            yield build_runtime_services(stores.domain, authorization, profile=profile, temporal_enabled=temporal_enabled, grant_key=grant_key, workflow_gateway=workflow_gateway, execution_launcher=execution_launcher, history_reader=history_reader, schema_digest=stores.domain.atomic_domain_id)
    finally:
        if local_lock is not None:
            await local_lock.release()

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
    profile: "ExecutionProfile",
    temporal_enabled: bool,
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
        profile=profile,
        temporal_enabled=temporal_enabled,
    )
    if profile is ExecutionProfile.PRODUCTION_SANDBOXED:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if workflow_gateway is not None and execution_launcher is not None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if profile is ExecutionProfile.LOCAL_CODING:
        if workflow_gateway is not None or execution_launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        launcher = execution_launcher
    elif profile is ExecutionProfile.PRODUCTION_SERVICE:
        if workflow_gateway is None or execution_launcher is not None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        launcher = _WorkflowExecutionLauncher(workflow_gateway)
    else:
        raise AIError(ErrorCode.PROFILE_NOT_ALLOWED)
    execution = DefaultExecutionService(persistence, authorization, launcher=launcher, service_profile=profile, history_reader=history_reader)
    services = RuntimeServices(
        identity,
        execution,
        DefaultSessionService(persistence, authorization, execution, HmacCursorSigner("session", grant_key), service_profile=profile),
        DefaultTaskService(persistence, authorization, workflow_gateway),
        DefaultEvaluationService(persistence, authorization, execution),
        DefaultApprovalService(persistence, authorization, workflow_gateway),
        DefaultEventService(persistence, authorization),
        DefaultArtifactService(persistence, authorization, grant_key=grant_key, cursor_signer=HmacCursorSigner("artifact", grant_key)),
    )
    _logger.info("runtime services composed: profile=%s backend=%s namespace=%s", profile, persistence.backend, persistence.namespace)
    return services


__all__ = ["AppServices", "RuntimeFactory", "RuntimeStoreConfig", "RuntimeStores", "StepExecutionHistoryReader", "build_app_services", "build_asset_codecs", "build_runtime_services", "namespace_scoped_step_db_path", "open_runtime_services", "open_runtime_store"]
