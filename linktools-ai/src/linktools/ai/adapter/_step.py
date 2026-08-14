#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime Step staging and explicit owner archive implementations."""

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaContext
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
)

from ..core import validate_persistence_namespace, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..runtime import RuntimeDomain, RuntimeRetention
from ..storage import (
    FilesystemMutationLock,
    FilesystemObjectStore,
    FilesystemWriterLock,
    ObjectStore,
    SqlObjectStore,
    SqlStorageContext,
    create_sql_storage_context,
    dialect_for_name,
    provision_sql,
    read_object,
    validate_sql,
    write_json_atomic,
)
from ._schema import build_step_sql_metadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine



_logger = environ.get_logger("ai.adapter.step")
_ARCHIVE_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY})
_ARCHIVE_ORDER = (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)


@dataclass(frozen=True, slots=True)
class _ExecutionProjection:
    run: RunRecord
    events: tuple[StepEvent, ...]
    snapshots: tuple[ContinuableSnapshot, ...]


@dataclass(frozen=True, slots=True)
class _RecoveryEffectProjection:
    run: RunRecord
    effects: tuple[ToolEffectRecord, ...]


class StagingStepStore(StepStore):
    """Process-local Step facts used before explicit promotion."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[StepEvent]] = {}
        self._snapshots: dict[str, list[ContinuableSnapshot]] = {}
        self._effects: dict[str, list[ToolEffectRecord]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def register_run(self, record: RunRecord) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        async with self._lock:
            if record.run_id in self._runs and self._runs[record.run_id] != record:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[record.run_id] = record

    async def get_run(self, *, run_id: str) -> "RunRecord | None":
        return self._runs.get(run_id)

    async def list_runs(self, *, parent_run_id: "str | None" = None, conversation_id: "str | None" = None) -> "list[RunRecord]":
        values = [item for item in self._runs.values() if (parent_run_id is None or item.parent_run_id == parent_run_id) and (conversation_id is None or item.conversation_id == conversation_id)]
        return sorted(values, key=lambda item: (item.started_at, item.run_id))

    async def append_event(self, event: StepEvent) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if event.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            values = self._events.setdefault(event.run_id, [])
            if event in values:
                return
            values.append(event)

    async def list_events(self, *, run_id: str) -> "list[StepEvent]":
        return list(self._events.get(run_id, ()))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if snapshot.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            values = self._snapshots.setdefault(snapshot.run_id, [])
            if snapshot in values:
                return
            if values and values[-1] == snapshot:
                return
            values.append(snapshot)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> "ContinuableSnapshot | None":
        values = self._snapshots.get(run_id, ())
        if not values:
            return None
        snapshot = values[-1]
        if not include_interrupted and snapshot.state != "complete":
            return None
        return snapshot

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)
        if record.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            values = self._effects.setdefault(record.run_id, [])
            if record in values:
                return
            if values and values[-1] == record:
                return
            values.append(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> "ToolEffectRecord | None":
        values = [item for item in self._effects.get(run_id, ()) if item.tool_call_id == tool_call_id]
        return values[-1] if values else None

    async def list_unresolved_tool_effects(self, *, run_id: str) -> "list[ToolEffectRecord]":
        latest: dict[str, ToolEffectRecord] = {}
        for item in self._effects.get(run_id, ()):
            latest[item.tool_call_id] = item
        return [item for item in latest.values() if item.status == "started"]

    async def fact_run_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._runs) | set(self._events) | set(self._snapshots) | set(self._effects)))

    async def _fact_projection(self, run_id: str) -> "tuple[RunRecord | None, tuple[StepEvent, ...], tuple[ContinuableSnapshot, ...], tuple[ToolEffectRecord, ...]]":
        async with self._lock:
            return (
                self._runs.get(run_id),
                tuple(self._events.get(run_id, ())),
                tuple(self._snapshots.get(run_id, ())),
                tuple(self._effects.get(run_id, ())),
            )

    async def _clear_facts(self) -> None:
        async with self._lock:
            self._runs.clear()
            self._events.clear()
            self._snapshots.clear()
            self._effects.clear()

    async def release_run(self, run_id: str) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)
            self._events.pop(run_id, None)
            self._snapshots.pop(run_id, None)
            self._effects.pop(run_id, None)


class InMemoryStepArchive(StepStore):
    """Hydrated owner archive for volatile and transient Conversation facts."""

    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        self._store = StagingStepStore()
        self._runtime_domain = runtime_domain

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    async def initialize(self) -> None:
        await self._store.initialize()

    async def close(self) -> None:
        await self._store.close()

    async def register_run(self, record: RunRecord) -> None:
        await self._store.register_run(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await self._store.get_run(run_id=run_id)

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        return await self._store.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def append_event(self, event: StepEvent) -> None:
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.append_event(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await self._store.list_events(run_id=run_id)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self._runtime_domain not in {RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.save_snapshot(snapshot)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        return await self._store.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.record_tool_effect(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await self._store.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await self._store.list_unresolved_tool_effects(run_id=run_id)

    async def _fact_projection(self, run_id: str) -> tuple[RunRecord | None, tuple[StepEvent, ...], tuple[ContinuableSnapshot, ...], tuple[ToolEffectRecord, ...]]:
        return await self._store._fact_projection(run_id)

    async def _clear_facts(self) -> None:
        await self._store._clear_facts()

    async def release_run(self, run_id: str) -> None:
        await self._store.release_run(run_id)


class FilesystemStepArchive(StagingStepStore):
    """Generation-two append-only Step archive with fixed owner domain."""

    def __init__(self, root: str | Path, *, namespace: str, tenant_id: str, runtime_domain: RuntimeDomain, object_store: ObjectStore | None = None) -> None:
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        super().__init__()
        self._root = Path(root).expanduser().resolve() / _namespace_key(namespace) / "steps" / runtime_domain.value / _scope_key(tenant_id)
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        physical_root = Path(root).expanduser().resolve()
        self._object_store = object_store if object_store is not None else FilesystemObjectStore(physical_root / _namespace_key(namespace) / "objects")
        self._mutation_lock = FilesystemMutationLock(self._root / "mutation.lock")
        self._lifetime_lock = FilesystemWriterLock(physical_root / _namespace_key(namespace) / "steps" / "step.lock")
        self._manage_writer_lock = True
        self._counters: dict[str, dict[str, int]] = {}
        self._ready = False

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @classmethod
    def _runtime(
        cls,
        root: str | Path,
        *,
        namespace: str,
        tenant_id: str,
        runtime_domain: RuntimeDomain,
        object_store: ObjectStore,
        writer_lock: FilesystemWriterLock,
    ) -> "FilesystemStepArchive":
        archive = cls(
            root,
            namespace=namespace,
            tenant_id=tenant_id,
            runtime_domain=runtime_domain,
            object_store=object_store,
        )
        archive._lifetime_lock = writer_lock
        archive._manage_writer_lock = False
        return archive

    async def initialize(self) -> None:
        if self._manage_writer_lock:
            await self._lifetime_lock.acquire()
        try:
            async with self._mutation_lock:
                self._root.mkdir(parents=True, exist_ok=True)
                for run_path in self._root.glob("runs/*"):
                    await self._load_run(run_path)
            self._ready = True
        except BaseException:
            if self._manage_writer_lock:
                await self._lifetime_lock.release()
            raise
        _logger.debug("step archive initialized: domain=%s tenant_scope=%s", self._runtime_domain.value, _scope_key(self._tenant_id))

    async def close(self) -> None:
        self._ready = False
        if self._manage_writer_lock and self._lifetime_lock.acquired:
            await self._lifetime_lock.release()

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_ready()
        async with self._mutation_lock:
            previous = self._runs.get(record.run_id)
            await super().register_run(record)
            directory = self._run_dir(record.run_id)
            try:
                directory.mkdir(parents=True, exist_ok=True)
                counters = self._counters.setdefault(record.run_id, {"last_event_index": 0, "last_snapshot_index": 0, "last_effect_index": 0})
                await asyncio.to_thread(write_json_atomic, directory / "run.json", _run_json(record, counters), fsync=True)
            except BaseException as error:
                if previous is None:
                    self._runs.pop(record.run_id, None)
                    self._counters.pop(record.run_id, None)
                else:
                    self._runs[record.run_id] = previous
                _raise_filesystem_storage_error(error)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_ready()
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if event.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._mutation_lock:
            before = len(self._events.get(event.run_id, ()))
            previous_counter = self._counters[event.run_id]["last_event_index"]
            await super().append_event(event)
            if len(self._events.get(event.run_id, ())) == before:
                return
            counters = self._counters[event.run_id]
            index = counters["last_event_index"] + 1
            try:
                await asyncio.to_thread(write_json_atomic, self._run_dir(event.run_id) / "events" / f"event-{index:020d}.json", _event_json(event), fsync=True)
                counters["last_event_index"] = index
                await asyncio.to_thread(write_json_atomic, self._run_dir(event.run_id) / "run.json", _run_json(self._runs[event.run_id], counters), fsync=True)
            except BaseException as error:
                self._events[event.run_id] = self._events[event.run_id][:before]
                counters["last_event_index"] = previous_counter
                _raise_filesystem_storage_error(error)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_ready()
        if self._runtime_domain not in {RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if snapshot.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._mutation_lock:
            before = len(self._snapshots.get(snapshot.run_id, ()))
            previous_counter = self._counters[snapshot.run_id]["last_snapshot_index"]
            await super().save_snapshot(snapshot)
            if len(self._snapshots.get(snapshot.run_id, ())) == before:
                return
            counters = self._counters[snapshot.run_id]
            index = counters["last_snapshot_index"] + 1
            try:
                payload = await _snapshot_json(snapshot, self._object_store, self._media_prefix())
                await asyncio.to_thread(write_json_atomic, self._run_dir(snapshot.run_id) / "snapshots" / f"snapshot-{index:020d}.json", payload, fsync=True)
                counters["last_snapshot_index"] = index
                await asyncio.to_thread(write_json_atomic, self._run_dir(snapshot.run_id) / "run.json", _run_json(self._runs[snapshot.run_id], counters), fsync=True)
            except BaseException as error:
                self._snapshots[snapshot.run_id] = self._snapshots[snapshot.run_id][:before]
                counters["last_snapshot_index"] = previous_counter
                _raise_filesystem_storage_error(error)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_ready()
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._mutation_lock:
            before = len(self._effects.get(record.run_id, ()))
            previous_counter = self._counters[record.run_id]["last_effect_index"]
            await super().record_tool_effect(record)
            if len(self._effects.get(record.run_id, ())) == before:
                return
            counters = self._counters[record.run_id]
            index = counters["last_effect_index"] + 1
            try:
                await asyncio.to_thread(write_json_atomic, self._run_dir(record.run_id) / "effects" / f"effect-{index:020d}.json", _effect_json(record), fsync=True)
                counters["last_effect_index"] = index
                await asyncio.to_thread(write_json_atomic, self._run_dir(record.run_id) / "run.json", _run_json(self._runs[record.run_id], counters), fsync=True)
            except BaseException as error:
                self._effects[record.run_id] = self._effects[record.run_id][:before]
                counters["last_effect_index"] = previous_counter
                _raise_filesystem_storage_error(error)

    async def _load_run(self, directory: Path) -> None:
        run_path = directory / "run.json"
        if not run_path.is_file():
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        value = await _read_json(run_path)
        try:
            record = _run_from_json(value)
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        if directory.name != _digest(record.run_id):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        try:
            counters = {name: int(value.get(name, 0)) for name in ("last_event_index", "last_snapshot_index", "last_effect_index")}
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        if any(number < 0 for number in counters.values()):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        original_counters = dict(counters)
        self._counters[record.run_id] = counters
        await super().register_run(record)
        event_paths = sorted((directory / "events").glob("event-*.json"))
        snapshot_paths = sorted((directory / "snapshots").glob("snapshot-*.json"))
        effect_paths = sorted((directory / "effects").glob("effect-*.json"))
        if (event_paths and self._runtime_domain is not RuntimeDomain.EXECUTION) or (effect_paths and self._runtime_domain is not RuntimeDomain.RECOVERY):
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        for paths, counter_name in ((event_paths, "last_event_index"), (snapshot_paths, "last_snapshot_index"), (effect_paths, "last_effect_index")):
            indexes = [_fact_index(path) for path in paths]
            if indexes != list(range(1, len(indexes) + 1)) or counters[counter_name] not in {len(indexes), len(indexes) - 1 if indexes else 0}:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            counters[counter_name] = len(indexes)
        for path in event_paths:
            try:
                event = _event_from_json(await _read_json(path))
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
            if event.run_id != record.run_id:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await super().append_event(event)
        for path in snapshot_paths:
            try:
                snapshot = await _snapshot_from_json(await _read_json(path), self._object_store)
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
            if snapshot.run_id != record.run_id:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await super().save_snapshot(snapshot)
        for path in effect_paths:
            try:
                effect = _effect_from_json(await _read_json(path))
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
            if effect.run_id != record.run_id:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await super().record_tool_effect(effect)
        if original_counters != counters:
            await asyncio.to_thread(write_json_atomic, run_path, _run_json(record, counters), fsync=True)

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(character in run_id for character in "/\\\x00"):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return self._root / "runs" / _digest(run_id)

    def _media_prefix(self) -> str:
        return f"v1/step/{self._runtime_domain.value}/{_namespace_key(self._namespace)}/{_scope_key(self._tenant_id)}"

    async def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


class SqlStepArchive(StepStore):
    """Explicit SQL Step archive boundary; schema is owned by the Runtime adapter."""

    def __init__(self, engine: "AsyncEngine", *, namespace: str, tenant_id: str, runtime_domain: RuntimeDomain, object_store: ObjectStore | None = None) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        dialect_for_name(engine.dialect.name)
        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        self.engine = engine
        self.namespace = namespace
        self.tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self.object_store = object_store or SqlObjectStore(engine)
        self._provision = True
        self._context = create_sql_storage_context(engine)
        self._owns_context = True
        self._metadata = build_step_sql_metadata(
            runtime_domain,
            object_store=None if self.object_store.store_id == "builtin" else self.object_store,
        )
        self._ready = False

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @classmethod
    def _runtime(
        cls,
        engine: "AsyncEngine",
        *,
        namespace: str,
        tenant_id: str,
        runtime_domain: RuntimeDomain,
        object_store: ObjectStore,
        context: "SqlStorageContext",
    ) -> "SqlStepArchive":
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        archive = cls.__new__(cls)
        archive.engine = engine
        archive.namespace = namespace
        archive.tenant_id = tenant_id
        archive._runtime_domain = runtime_domain
        archive.object_store = object_store
        archive._provision = False
        archive._context = context
        archive._owns_context = False
        archive._metadata = build_step_sql_metadata(
            runtime_domain,
            object_store=None if object_store.store_id == "builtin" else object_store,
        )
        archive._ready = False
        return archive

    async def initialize(self) -> None:
        try:
            if self._provision:
                await provision_sql(self.engine, self._metadata)
                await self._context.initialize()
            else:
                await validate_sql(self.engine, self._metadata)
            await self._validate_current_state()
            self._ready = True
            _logger.debug("SQL step archive initialized: domain=%s", self._runtime_domain.value)
        except BaseException:
            if self._owns_context:
                try:
                    await self._context.close()
                except BaseException:
                    _logger.error("SQL step context cleanup failed after initialize error", exc_info=environ.debug)
            raise

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_ready()
        from sqlalchemy import select

        table = self._metadata.tables["step_runs"]
        key = _namespace_key(self.namespace)
        async with self._begin() as connection:
            row = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.runtime_domain == self._runtime_domain.value, table.c.step_run_id == run_id).with_for_update())).mappings().first()
            result = None if row is None else _run_from_sql(row)
            if row is not None:
                await self._validate_run_indexes(connection, run_id, row)
        return result

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        await self._ensure_ready()
        from sqlalchemy import select

        table = self._metadata.tables["step_runs"]
        key = _namespace_key(self.namespace)
        query = select(table).where(
            table.c.namespace_key == key,
            table.c.tenant_id == self.tenant_id,
            table.c.runtime_domain == self._runtime_domain.value,
        )
        if parent_run_id is not None:
            query = query.where(table.c.parent_step_run_id == parent_run_id)
        if conversation_id is not None:
            query = query.where(table.c.harness_conversation_id == conversation_id)
        query = query.order_by(table.c.started_at, table.c.step_run_id)
        async with self._begin() as connection:
            rows = (await connection.execute(query.with_for_update())).mappings().all()
            for row in rows:
                await self._validate_run_indexes(connection, str(row["step_run_id"]), row)
            result = [_run_from_sql(row) for row in rows]
        return result

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_ready()
        if "step_events" not in self._metadata.tables:
            return []
        from sqlalchemy import select

        table = self._metadata.tables["step_events"]
        key = _namespace_key(self.namespace)
        async with self._begin() as connection:
            runs = self._metadata.tables["step_runs"]
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == run_id).with_for_update())).mappings().first()
            rows = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.step_run_id == run_id).order_by(table.c.event_index))).mappings().all()
            if run_row is None:
                if rows:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return []
            await self._validate_run_indexes(connection, run_id, run_row)
            result = [_event_from_sql(row) for row in rows]
        return result

    async def _execution_projection(self, *, run_id: str) -> _ExecutionProjection | None:
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._ensure_ready()
        from sqlalchemy import select

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        events = self._metadata.tables["step_events"]
        snapshots = self._metadata.tables["step_snapshots"]
        async with self._begin() as connection:
            run_row = (
                await connection.execute(
                    select(runs).where(
                        runs.c.namespace_key == key,
                        runs.c.tenant_id == self.tenant_id,
                        runs.c.runtime_domain == self._runtime_domain.value,
                        runs.c.step_run_id == run_id,
                    ).with_for_update()
                )
            ).mappings().first()
            if run_row is None:
                return None
            event_rows = (
                await connection.execute(
                    select(events)
                    .where(events.c.namespace_key == key, events.c.tenant_id == self.tenant_id, events.c.step_run_id == run_id)
                    .order_by(events.c.event_index)
                )
            ).mappings().all()
            snapshot_rows = (
                await connection.execute(
                    select(snapshots)
                    .where(
                        snapshots.c.namespace_key == key,
                        snapshots.c.tenant_id == self.tenant_id,
                        snapshots.c.runtime_domain == self._runtime_domain.value,
                        snapshots.c.step_run_id == run_id,
                    )
                    .order_by(snapshots.c.snapshot_index)
                )
                ).mappings().all()
            _validate_step_indexes(int(run_row["last_event_index"]), event_rows, "event_index")
            _validate_step_indexes(int(run_row["last_snapshot_index"]), snapshot_rows, "snapshot_index")
            result_run = _run_from_sql(run_row)
            result_events = tuple(_event_from_sql(row) for row in event_rows)
            result_snapshots = tuple(dict(row) for row in snapshot_rows)
        return _ExecutionProjection(
            run=result_run,
            events=result_events,
            snapshots=tuple([await _snapshot_from_sql(row, self.object_store) for row in result_snapshots]),
        )

    async def _recovery_effect_projection(self, *, run_id: str) -> _RecoveryEffectProjection | None:
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._ensure_ready()
        from sqlalchemy import select

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        effects = self._metadata.tables["step_effects"]
        async with self._begin() as connection:
            run_row = (
                await connection.execute(
                    select(runs).where(
                        runs.c.namespace_key == key,
                        runs.c.tenant_id == self.tenant_id,
                        runs.c.runtime_domain == self._runtime_domain.value,
                        runs.c.step_run_id == run_id,
                    ).with_for_update()
                )
            ).mappings().first()
            if run_row is None:
                return None
            effect_rows = (
                await connection.execute(
                    select(effects)
                    .where(effects.c.namespace_key == key, effects.c.tenant_id == self.tenant_id, effects.c.step_run_id == run_id)
                    .order_by(effects.c.effect_index)
                )
            ).mappings().all()
            _validate_step_indexes(int(run_row["last_effect_index"]), effect_rows, "effect_index")
            result = _RecoveryEffectProjection(run=_run_from_sql(run_row), effects=tuple(_effect_from_sql(row) for row in effect_rows))
        return result

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_ready()
        from sqlalchemy import select

        table = self._metadata.tables["step_snapshots"]
        key = _namespace_key(self.namespace)
        async with self._begin() as connection:
            runs = self._metadata.tables["step_runs"]
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == run_id).with_for_update())).mappings().first()
            if run_row is None:
                return None
            rows = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.runtime_domain == self._runtime_domain.value, table.c.step_run_id == run_id).order_by(table.c.snapshot_index))).mappings().all()
            _validate_step_indexes(int(run_row["last_snapshot_index"]), rows, "snapshot_index")
            row = dict(rows[-1]) if rows else None
        if row is None or (not include_interrupted and str(row["state"]) != "complete"):
            return None
        return await _snapshot_from_sql(row, self.object_store)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        await self._ensure_ready()
        if "step_effects" not in self._metadata.tables:
            return None
        from sqlalchemy import select

        table = self._metadata.tables["step_effects"]
        key = _namespace_key(self.namespace)
        async with self._begin() as connection:
            runs = self._metadata.tables["step_runs"]
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == run_id).with_for_update())).mappings().first()
            query = select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.step_run_id == run_id, table.c.tool_call_id == tool_call_id).order_by(table.c.effect_index.desc()).limit(1)
            row = (await connection.execute(query)).mappings().first()
            if run_row is None:
                if row is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return None
            await self._validate_run_indexes(connection, run_id, run_row)
        return None if row is None else _effect_from_sql(row)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        await self._ensure_ready()
        if "step_effects" not in self._metadata.tables:
            return []
        from sqlalchemy import select

        table = self._metadata.tables["step_effects"]
        key = _namespace_key(self.namespace)
        async with self._begin() as connection:
            runs = self._metadata.tables["step_runs"]
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == run_id).with_for_update())).mappings().first()
            rows = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.step_run_id == run_id).order_by(table.c.effect_index))).mappings().all()
            if run_row is None:
                if rows:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return []
            await self._validate_run_indexes(connection, run_id, run_row)
        latest: dict[str, ToolEffectRecord] = {}
        for row in rows:
            effect = _effect_from_sql(row)
            latest[effect.tool_call_id] = effect
        return [effect for effect in latest.values() if effect.status == "started"]

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_ready()
        from sqlalchemy import insert, select
        from sqlalchemy.exc import IntegrityError

        table = self._metadata.tables["step_runs"]
        key = _namespace_key(self.namespace)
        try:
            async with self._begin() as connection:
                current = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.runtime_domain == self._runtime_domain.value, table.c.step_run_id == record.run_id).with_for_update())).mappings().first()
                if current is not None:
                    await self._validate_run_indexes(connection, record.run_id, current)
                    if _run_from_sql(current) != record:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                else:
                    await connection.execute(insert(table).values(namespace_key=key, tenant_id=self.tenant_id, runtime_domain=self._runtime_domain.value, step_run_id=record.run_id, harness_conversation_id=record.conversation_id, parent_step_run_id=record.parent_run_id, agent_name=record.agent_name, metadata_json=dict(record.metadata), last_event_index=0, last_snapshot_index=0, last_effect_index=0, started_at=record.started_at))
            return
        except IntegrityError:
            async with self._begin() as connection:
                current = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.runtime_domain == self._runtime_domain.value, table.c.step_run_id == record.run_id).with_for_update())).mappings().first()
                if current is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await self._validate_run_indexes(connection, record.run_id, current)
                if _run_from_sql(current) != record:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_ready()
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from sqlalchemy import insert, select, update

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        events = self._metadata.tables["step_events"]
        async with self._begin() as connection:
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == event.run_id).with_for_update())).mappings().first()
            if run_row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            await self._validate_run_indexes(connection, event.run_id, run_row)
            previous_rows = (
                await connection.execute(
                    select(events)
                    .where(
                        events.c.namespace_key == key,
                        events.c.tenant_id == self.tenant_id,
                        events.c.step_run_id == event.run_id,
                    )
                    .order_by(events.c.event_index)
                )
            ).mappings().all()
            if any(_event_from_sql(previous) == event for previous in previous_rows):
                return
            index = int(run_row["last_event_index"]) + 1
            await connection.execute(insert(events).values(namespace_key=key, tenant_id=self.tenant_id, step_run_id=event.run_id, event_index=index, event_kind=event.kind, step_index=event.step_index, timestamp=event.timestamp, harness_conversation_id=event.conversation_id, parent_step_run_id=event.parent_run_id, agent_name=event.agent_name, tool_call_id=event.tool_call_id, tool_name=event.tool_name, error=event.error, metadata_json=dict(event.metadata)))
            await connection.execute(update(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == event.run_id).values(last_event_index=index))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_ready()
        if self._runtime_domain not in {RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from sqlalchemy import insert, select, update

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        snapshots = self._metadata.tables["step_snapshots"]
        async with self._begin() as preflight_connection:
            preflight_row = (
                await preflight_connection.execute(
                select(runs).where(
                    runs.c.namespace_key == key,
                    runs.c.tenant_id == self.tenant_id,
                    runs.c.runtime_domain == self._runtime_domain.value,
                    runs.c.step_run_id == snapshot.run_id,
                )
                )
            ).mappings().first()
            if preflight_row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            await self._validate_run_indexes(preflight_connection, snapshot.run_id, preflight_row)
        payload = await _snapshot_json(snapshot, self.object_store, self._media_prefix())
        async with self._begin() as connection:
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == snapshot.run_id).with_for_update())).mappings().first()
            if run_row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            await self._validate_run_indexes(connection, snapshot.run_id, run_row)
            previous_rows = (await connection.execute(select(snapshots).where(snapshots.c.namespace_key == key, snapshots.c.tenant_id == self.tenant_id, snapshots.c.runtime_domain == self._runtime_domain.value, snapshots.c.step_run_id == snapshot.run_id).order_by(snapshots.c.snapshot_index))).mappings().all()
            if any(_snapshot_row_matches(previous_row, payload, snapshot) for previous_row in previous_rows):
                return
            index = int(run_row["last_snapshot_index"]) + 1
            await connection.execute(insert(snapshots).values(namespace_key=key, tenant_id=self.tenant_id, runtime_domain=self._runtime_domain.value, step_run_id=snapshot.run_id, snapshot_index=index, step_index=snapshot.step_index, state=snapshot.state, harness_conversation_id=snapshot.conversation_id, parent_step_run_id=snapshot.parent_run_id, agent_name=snapshot.agent_name, timestamp=snapshot.timestamp, object_store_id=self.object_store.store_id, messages_json=payload["messages"]))
            await connection.execute(update(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == snapshot.run_id).values(last_snapshot_index=index))

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_ready()
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from sqlalchemy import insert, select, update

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        effects = self._metadata.tables["step_effects"]
        async with self._begin() as connection:
            run_row = (await connection.execute(select(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == record.run_id).with_for_update())).mappings().first()
            if run_row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            await self._validate_run_indexes(connection, record.run_id, run_row)
            previous_rows = (await connection.execute(select(effects).where(effects.c.namespace_key == key, effects.c.tenant_id == self.tenant_id, effects.c.step_run_id == record.run_id).order_by(effects.c.effect_index))).mappings().all()
            if any(_effect_from_sql(previous_row) == record for previous_row in previous_rows):
                return
            index = int(run_row["last_effect_index"]) + 1
            await connection.execute(insert(effects).values(namespace_key=key, tenant_id=self.tenant_id, step_run_id=record.run_id, effect_index=index, tool_call_id=record.tool_call_id, tool_name=record.tool_name, status=record.status, started_at=record.started_at, ended_at=record.ended_at, idempotency_key=record.idempotency_key, effect_summary=json.dumps(record.effect_summary, sort_keys=True, separators=(",", ":"))))
            await connection.execute(update(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self._runtime_domain.value, runs.c.step_run_id == record.run_id).values(last_effect_index=index))

    async def close(self) -> None:
        self._ready = False
        if self._owns_context:
            await self._context.close()

    async def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def _validate_current_state(self) -> None:
        from sqlalchemy import select

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        async with self._begin() as connection:
            run_rows = (
                await connection.execute(
                    select(runs).where(
                        runs.c.namespace_key == key,
                        runs.c.tenant_id == self.tenant_id,
                        runs.c.runtime_domain == self._runtime_domain.value,
                    )
                )
            ).mappings().all()
            run_ids = {str(row["step_run_id"]) for row in run_rows}
            event_rows: Sequence[Mapping[str, object]] = ()
            if "step_events" in self._metadata.tables:
                events = self._metadata.tables["step_events"]
                event_rows = (
                    await connection.execute(
                        select(events).where(
                            events.c.namespace_key == key,
                            events.c.tenant_id == self.tenant_id,
                        ).order_by(events.c.step_run_id, events.c.event_index)
                    )
                ).mappings().all()
            snapshot_rows: Sequence[Mapping[str, object]] = ()
            if "step_snapshots" in self._metadata.tables:
                snapshots = self._metadata.tables["step_snapshots"]
                snapshot_rows = (
                    await connection.execute(
                        select(snapshots).where(
                            snapshots.c.namespace_key == key,
                            snapshots.c.tenant_id == self.tenant_id,
                            snapshots.c.runtime_domain == self._runtime_domain.value,
                        ).order_by(snapshots.c.step_run_id, snapshots.c.snapshot_index)
                    )
                ).mappings().all()
            effect_rows: Sequence[Mapping[str, object]] = ()
            if "step_effects" in self._metadata.tables:
                effects = self._metadata.tables["step_effects"]
                effect_rows = (
                    await connection.execute(
                        select(effects).where(
                            effects.c.namespace_key == key,
                            effects.c.tenant_id == self.tenant_id,
                        ).order_by(effects.c.step_run_id, effects.c.effect_index)
                    )
                ).mappings().all()
            for child_rows, index_name in ((event_rows, "event_index"), (snapshot_rows, "snapshot_index"), (effect_rows, "effect_index")):
                if any(str(row["step_run_id"]) not in run_ids for row in child_rows):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            grouped_events = _group_step_rows(event_rows)
            grouped_snapshots = _group_step_rows(snapshot_rows)
            grouped_effects = _group_step_rows(effect_rows)
            for run_row in run_rows:
                run_id = str(run_row["step_run_id"])
                _validate_step_indexes(int(run_row["last_event_index"]), grouped_events.get(run_id, ()), "event_index")
                _validate_step_indexes(int(run_row["last_snapshot_index"]), grouped_snapshots.get(run_id, ()), "snapshot_index")
                _validate_step_indexes(int(run_row["last_effect_index"]), grouped_effects.get(run_id, ()), "effect_index")
            for row in event_rows:
                _event_from_sql(row)
            for row in effect_rows:
                _effect_from_sql(row)
            snapshot_values = tuple(dict(row) for row in snapshot_rows)
        for row in snapshot_values:
            await _snapshot_from_sql(row, self.object_store)

    async def _validate_run_indexes(self, connection: object, run_id: str, run_row: Mapping[str, object]) -> None:
        from sqlalchemy import select

        key = _namespace_key(self.namespace)
        if "step_events" in self._metadata.tables:
            events = self._metadata.tables["step_events"]
            event_rows = (
                await connection.execute(
                    select(events)
                    .where(events.c.namespace_key == key, events.c.tenant_id == self.tenant_id, events.c.step_run_id == run_id)
                    .order_by(events.c.event_index)
                )
            ).mappings().all()
            _validate_step_indexes(int(run_row["last_event_index"]), event_rows, "event_index")
        if "step_snapshots" in self._metadata.tables:
            snapshots = self._metadata.tables["step_snapshots"]
            snapshot_rows = (
                await connection.execute(
                    select(snapshots)
                    .where(snapshots.c.namespace_key == key, snapshots.c.tenant_id == self.tenant_id, snapshots.c.runtime_domain == self._runtime_domain.value, snapshots.c.step_run_id == run_id)
                    .order_by(snapshots.c.snapshot_index)
                )
            ).mappings().all()
            _validate_step_indexes(int(run_row["last_snapshot_index"]), snapshot_rows, "snapshot_index")
        if "step_effects" in self._metadata.tables:
            effects = self._metadata.tables["step_effects"]
            effect_rows = (
                await connection.execute(
                    select(effects)
                    .where(effects.c.namespace_key == key, effects.c.tenant_id == self.tenant_id, effects.c.step_run_id == run_id)
                    .order_by(effects.c.effect_index)
                )
            ).mappings().all()
            _validate_step_indexes(int(run_row["last_effect_index"]), effect_rows, "effect_index")

    def _media_prefix(self) -> str:
        return f"v1/step/{self._runtime_domain.value}/{_namespace_key(self.namespace)}/{_scope_key(self.tenant_id)}"

    @asynccontextmanager
    async def _begin(self) -> AsyncIterator[object]:
        async with self._context.sessions.begin() as session:
            yield session


class RuntimeStepPersistence(StepStore):
    """Own the staging facts and the fixed Step archive routing matrix."""

    def __init__(
        self,
        staging: StagingStepStore,
        *,
        conversation_archive: StepStore,
        execution_archive: "StepStore | None",
        recovery_archive: "StepStore | None",
        conversation_retention: RuntimeRetention,
        execution_retention: RuntimeRetention,
        recovery_retention: RuntimeRetention,
    ) -> None:
        self._staging = staging
        self._validate_archive(RuntimeDomain.CONVERSATION, conversation_retention, conversation_archive)
        self._validate_archive(RuntimeDomain.EXECUTION, execution_retention, execution_archive)
        self._validate_archive(RuntimeDomain.RECOVERY, recovery_retention, recovery_archive)
        self._archives = {
            RuntimeDomain.CONVERSATION: conversation_archive,
            **({RuntimeDomain.EXECUTION: execution_archive} if execution_archive is not None else {}),
            **({RuntimeDomain.RECOVERY: recovery_archive} if recovery_archive is not None else {}),
        }
        self._step_reads = {
            RuntimeDomain.CONVERSATION: conversation_archive,
            RuntimeDomain.EXECUTION: execution_archive if execution_archive is not None else staging,
            RuntimeDomain.RECOVERY: recovery_archive if recovery_archive is not None else staging,
        }
        self._business_unavailable = False
        self._close_preflight_ok = False
        self._close_clear_done = False
        self._close_child_cursor = 0
        self._close_lock = asyncio.Lock()
        self._initialized: list[StepStore] = []

    @staticmethod
    def _validate_archive(runtime_domain: RuntimeDomain, retention: RuntimeRetention, archive: StepStore | None) -> None:
        if not isinstance(retention, RuntimeRetention):
            raise ValueError("Step archive retention is invalid")
        if runtime_domain is RuntimeDomain.CONVERSATION and archive is None:
            raise ValueError("Conversation Step archive is required")
        if retention is RuntimeRetention.TRANSIENT:
            if runtime_domain is RuntimeDomain.CONVERSATION and not isinstance(archive, InMemoryStepArchive):
                raise ValueError("transient Conversation archive must be in-memory")
            if runtime_domain is RuntimeDomain.CONVERSATION and archive.runtime_domain is not runtime_domain:
                raise ValueError("Step archive owner is invalid")
            if runtime_domain is not RuntimeDomain.CONVERSATION and archive is not None:
                raise ValueError("transient Step archive must be absent")
            return
        if retention is RuntimeRetention.VOLATILE:
            if not isinstance(archive, InMemoryStepArchive):
                raise ValueError("volatile Step archive must be in-memory")
            if archive.runtime_domain is not runtime_domain:
                raise ValueError("Step archive owner is invalid")
            return
        if not isinstance(archive, (FilesystemStepArchive, SqlStepArchive)):
            raise ValueError("durable Step archive must be filesystem or SQL")
        if archive.runtime_domain is not runtime_domain:
            raise ValueError("Step archive owner is invalid")

    async def initialize(self) -> None:
        children = self._children()
        initialized: list[StepStore] = []
        try:
            for child in children:
                await child.initialize()
                initialized.append(child)
        except BaseException:
            for child in reversed(initialized):
                await child.close()
            raise
        self._initialized = initialized
        self._business_unavailable = False

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_business()
        await self._staging.register_run(record)
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if archive is not None:
            await archive.register_run(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_business()
        return await self._staging.get_run(run_id=run_id)

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        await self._ensure_business()
        return await self._staging.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_business()
        await self._staging.append_event(event)
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if archive is not None:
            await _materialize_run(self._staging, archive, event.run_id)
            await archive.append_event(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_business()
        return await self._staging.list_events(run_id=run_id)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_business()
        await self._staging.save_snapshot(snapshot)
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if archive is not None:
            await _materialize_run(self._staging, archive, snapshot.run_id)
            await archive.save_snapshot(snapshot)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_business()
        return await self._staging.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_business()
        await self._staging.record_tool_effect(record)
        archive = self._archives.get(RuntimeDomain.RECOVERY)
        if archive is not None:
            await _materialize_run(self._staging, archive, record.run_id)
            await archive.record_tool_effect(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        await self._ensure_business()
        return await self._staging.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        await self._ensure_business()
        return await self._staging.list_unresolved_tool_effects(run_id=run_id)

    def read_store(self, runtime_domain: RuntimeDomain) -> StepStore:
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        return self._step_reads[runtime_domain]

    async def materialize_recovery_snapshot(self, *, step_run_id: str, require_complete: bool) -> None:
        run, _, _, _ = await self._staging._fact_projection(step_run_id)
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id, include_interrupted=True)
        if run is None:
            _, events, snapshots, effects = await self._staging._fact_projection(step_run_id)
            if events or snapshots or effects:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if require_complete:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if snapshot is None:
            if require_complete:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if require_complete and snapshot.state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive = self._archives.get(RuntimeDomain.RECOVERY)
        if archive is not None:
            await _materialize_snapshot(self._staging, archive, run, snapshot)

    async def materialize_conversation(self, *, step_run_id: str) -> None:
        run = await self._staging.get_run(run_id=step_run_id)
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id)
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if run is None or snapshot is None or snapshot.state != "complete" or archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await _materialize_snapshot(self._staging, archive, run, snapshot)

    async def materialize_from_recovery(self, *, target: RuntimeDomain, step_run_id: str) -> None:
        if target not in {RuntimeDomain.EXECUTION, RuntimeDomain.CONVERSATION}:
            raise ValueError("Step materialization target is invalid")
        run, events, snapshots, effects = await self._staging._fact_projection(step_run_id)
        child_exists = bool(events or snapshots or effects)
        if run is None and child_exists:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run_exists = run is not None
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        execution = self._archives.get(RuntimeDomain.EXECUTION)
        destination = self._archives.get(target)
        if target is RuntimeDomain.CONVERSATION:
            source = recovery if recovery is not None else self._staging if run_exists else None
            if source is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            source_run, source_snapshot = await _read_complete_current_snapshot(source, step_run_id)
            if destination is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            await _materialize_snapshot(source, destination, source_run, source_snapshot)
            return
        if run_exists:
            source = recovery if recovery is not None else self._staging
            source_run, source_snapshot = await _read_complete_current_snapshot(source, step_run_id)
            if execution is not None:
                await _materialize_snapshot(source, execution, source_run, source_snapshot)
            return
        if execution is None and recovery is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if execution is not None and recovery is not None:
            recovery_run, recovery_snapshot = await _read_complete_current_snapshot(recovery, step_run_id)
            recovery_projection = await _read_recovery_projection(recovery, step_run_id)
            if recovery_projection is None or recovery_projection.run != recovery_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await _materialize_snapshot(recovery, execution, recovery_run, recovery_snapshot)
            execution_projection = await _read_execution_projection(execution, step_run_id)
            if execution_projection is None or execution_projection.run != recovery_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_run = execution_projection.run
            source_events = execution_projection.events
            source_snapshots = execution_projection.snapshots
            source_effects = recovery_projection.effects
        elif execution is not None:
            execution_projection = await _read_execution_projection(execution, step_run_id)
            if execution_projection is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_run = execution_projection.run
            source_events = execution_projection.events
            source_snapshots = execution_projection.snapshots
            source_effects = ()
        else:
            recovery_run, recovery_snapshot = await _read_complete_current_snapshot(recovery, step_run_id)
            recovery_projection = await _read_recovery_projection(recovery, step_run_id)
            if recovery_projection is None or recovery_projection.run != recovery_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_run = recovery_run
            source_events = ()
            source_snapshots = (recovery_snapshot,)
            source_effects = recovery_projection.effects
        if not source_snapshots or source_snapshots[-1].state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._staging.register_run(source_run)
        for event in source_events:
            await self._staging.append_event(event)
        for snapshot in source_snapshots:
            await self._staging.save_snapshot(snapshot)
        for effect in source_effects:
            await self._staging.record_tool_effect(effect)

    async def verify_terminal_attempts(self, *, candidate_step_run_ids: tuple[str, ...], required_step_run_id: str | None) -> None:
        candidates = tuple(dict.fromkeys(candidate_step_run_ids))
        if required_step_run_id is not None and required_step_run_id not in candidates:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for run_id in candidates:
            await self._verify_staging_run(run_id, required=required_step_run_id == run_id)

    async def release_staging_many(self, *, candidate_step_run_ids: tuple[str, ...]) -> None:
        first: BaseException | None = None
        for run_id in tuple(dict.fromkeys(candidate_step_run_ids)):
            try:
                await self.verify_terminal_attempts(candidate_step_run_ids=(run_id,), required_step_run_id=None)
                await self._staging.release_run(run_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if first is None:
                    first = error
        if first is not None:
            raise first

    async def release_archive(self, runtime_domain: RuntimeDomain, step_run_id: str) -> None:
        if runtime_domain is not RuntimeDomain.CONVERSATION or runtime_domain not in self._archives:
            raise ValueError("only a conversation archive can be released")
        archive = self._archives[runtime_domain]
        if type(archive) is not InMemoryStepArchive:
            raise ValueError("conversation archive is not transient")
        await archive.release_run(step_run_id)

    async def preflight_close(self) -> None:
        async with self._close_lock:
            if self._close_preflight_ok:
                return
            for run_id in await self._staging.fact_run_ids():
                await self._verify_staging_run(run_id, required=False)
            self._close_preflight_ok = True

    async def close(self) -> None:
        async with self._close_lock:
            if not self._close_preflight_ok:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if not self._close_clear_done:
                await self._staging._clear_facts()
                for archive in self._archives.values():
                    if isinstance(archive, InMemoryStepArchive):
                        await archive._clear_facts()
                self._close_clear_done = True
            children = self._children()
            while self._close_child_cursor < len(children):
                await children[self._close_child_cursor].close()
                self._close_child_cursor += 1

    async def _ensure_business(self) -> None:
        if self._business_unavailable:
            raise AIError(ErrorCode.STORAGE_CLOSED)

    def _children(self) -> tuple[StepStore, ...]:
        values: list[StepStore] = []
        seen: set[int] = set()
        for child in (self._staging, *(self._archives.get(domain) for domain in _ARCHIVE_ORDER)):
            if child is not None and id(child) not in seen:
                values.append(child)
                seen.add(id(child))
        return tuple(values)

    async def _verify_staging_run(self, run_id: str, *, required: bool) -> None:
        staged_run, staged_events, staged_snapshots, staged_effects = await self._staging._fact_projection(run_id)
        if staged_run is None and not staged_events and not staged_snapshots and not staged_effects:
            if required:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if staged_run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if required and (not staged_snapshots or staged_snapshots[-1].state != "complete"):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = self._archives.get(RuntimeDomain.EXECUTION)
        if execution is not None:
            projection = await _read_execution_projection(execution, run_id)
            staged_projection = _ExecutionProjection(staged_run, staged_events, staged_snapshots)
            if projection != staged_projection:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        if recovery is not None:
            projection = await _read_recovery_projection(recovery, run_id)
            if projection is None:
                if staged_effects:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif projection != _RecoveryEffectProjection(staged_run, staged_effects):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _read_complete_current_snapshot(store: StepStore, run_id: str) -> tuple[RunRecord, ContinuableSnapshot]:
    run = await store.get_run(run_id=run_id)
    snapshot = await store.latest_snapshot(run_id=run_id, include_interrupted=True)
    if run is None or snapshot is None or snapshot.state != "complete":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return run, snapshot


async def _read_execution_projection(store: StepStore, run_id: str) -> _ExecutionProjection | None:
    if isinstance(store, SqlStepArchive):
        return await store._execution_projection(run_id=run_id)
    if isinstance(store, (StagingStepStore, InMemoryStepArchive)):
        run, events, snapshots, _ = await store._fact_projection(run_id)
        return None if run is None else _ExecutionProjection(run, events, snapshots)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _read_recovery_projection(store: StepStore, run_id: str) -> _RecoveryEffectProjection | None:
    if isinstance(store, SqlStepArchive):
        return await store._recovery_effect_projection(run_id=run_id)
    if isinstance(store, (StagingStepStore, InMemoryStepArchive)):
        run, _, _, effects = await store._fact_projection(run_id)
        return None if run is None else _RecoveryEffectProjection(run, effects)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _materialize_snapshot(source: StepStore, target: StepStore, run: RunRecord, snapshot: ContinuableSnapshot) -> None:
    await _materialize_run(source, target, run.run_id)
    await target.save_snapshot(snapshot)
    current = await target.latest_snapshot(run_id=run.run_id, include_interrupted=True)
    if current != snapshot or current is None or current.state != "complete":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _materialize_run(source: StepStore, target: StepStore, run_id: str) -> RunRecord:
    source_run = await source.get_run(run_id=run_id)
    if source_run is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    target_run = await target.get_run(run_id=run_id)
    if target_run is None:
        try:
            await target.register_run(source_run)
        except AIError:
            target_run = await target.get_run(run_id=run_id)
            if target_run == source_run:
                return source_run
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        target_run = await target.get_run(run_id=run_id)
    if target_run != source_run:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return source_run


class ObjectMediaAdapter:
    """Harness media protocol backed by the generic ObjectStore."""

    def __init__(self, object_store: ObjectStore, *, prefix: str = "v1/runtime/media") -> None:
        self._store = object_store
        self._prefix = prefix.rstrip("/")

    async def put(self, data: bytes, *, context: MediaContext = MediaContext()) -> str:
        del context
        digest = hashlib.sha256(data).hexdigest()
        key = f"{self._prefix}/{digest}"
        await self._store.put(key, _bytes(data), expected_size=len(data), expected_digest=digest)
        return f"object://{self._store.store_id}/{key}"

    async def get(self, uri: str, *, context: MediaContext = MediaContext()) -> bytes:
        del context
        store_id, key = _parse_uri(uri)
        if store_id != self._store.store_id:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        stat = await self._store.stat(key)
        if stat is None:
            raise FileNotFoundError(uri)
        return await read_object(self._store, key, expected_digest=stat.digest, expected_size=stat.size)

    async def exists(self, uri: str, *, context: MediaContext = MediaContext()) -> bool:
        del context
        store_id, key = _parse_uri(uri)
        if store_id != self._store.store_id:
            return False
        return await self._store.stat(key) is not None

    async def public_url(self, uri: str, *, context: MediaContext = MediaContext()) -> str | None:
        del uri, context
        return None

    async def get_metadata(self, uri: str, *, context: MediaContext = MediaContext()) -> Mapping[str, str]:
        del uri, context
        return {}


async def _bytes(value: bytes):
    yield value


async def _read_json(path: Path) -> dict[str, object]:
    try:
        value = await asyncio.to_thread(path.read_text, encoding="utf-8")
        parsed = json.loads(value)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if not isinstance(parsed, dict):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return parsed


def _raise_filesystem_storage_error(error: BaseException) -> None:
    if isinstance(error, AIError):
        raise error
    if isinstance(error, OSError):
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    raise error


def _namespace_key(namespace: str) -> str:
    return _digest(namespace)


def _scope_key(tenant_id: str) -> str:
    return _digest("tenant:" + tenant_id)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_json(record: RunRecord, counters: Mapping[str, int] | None = None) -> dict[str, object]:
    value = {"run_id": record.run_id, "conversation_id": record.conversation_id, "parent_run_id": record.parent_run_id, "agent_name": record.agent_name, "metadata": dict(record.metadata), "started_at": record.started_at.astimezone(timezone.utc).isoformat()}
    if counters is not None:
        value.update(counters)
    return value


def _group_step_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, tuple[Mapping[str, object], ...]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["step_run_id"]), []).append(row)
    return {run_id: tuple(values) for run_id, values in grouped.items()}


def _validate_step_indexes(counter: int, rows: Sequence[Mapping[str, object]], index_name: str) -> None:
    indexes = [int(row[index_name]) for row in rows]
    if counter < 0 or indexes != list(range(1, len(indexes) + 1)) or counter != len(indexes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _fact_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error


def _run_from_json(value: dict[str, object]) -> RunRecord:
    return RunRecord(run_id=str(value["run_id"]), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), metadata=dict(value.get("metadata", {})), started_at=_datetime(value["started_at"]))


def _run_from_sql(value: Mapping[str, object]) -> RunRecord:
    return RunRecord(run_id=str(value["step_run_id"]), conversation_id=value.get("harness_conversation_id"), parent_run_id=value.get("parent_step_run_id"), agent_name=value.get("agent_name"), metadata=dict(value.get("metadata_json") or {}), started_at=_datetime(value["started_at"]))


def _event_json(event: StepEvent) -> dict[str, object]:
    return {"run_id": event.run_id, "kind": event.kind, "step_index": event.step_index, "timestamp": event.timestamp.astimezone(timezone.utc).isoformat(), "conversation_id": event.conversation_id, "parent_run_id": event.parent_run_id, "agent_name": event.agent_name, "tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "error": event.error, "metadata": dict(event.metadata)}


def _event_from_json(value: dict[str, object]) -> StepEvent:
    return StepEvent(run_id=str(value["run_id"]), kind=value["kind"], step_index=int(value["step_index"]), timestamp=_datetime(value["timestamp"]), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), tool_call_id=value.get("tool_call_id"), tool_name=value.get("tool_name"), error=value.get("error"), metadata=dict(value.get("metadata", {})))


def _event_from_sql(value: Mapping[str, object]) -> StepEvent:
    return StepEvent(run_id=str(value["step_run_id"]), kind=str(value["event_kind"]), step_index=int(value["step_index"]), timestamp=_datetime(value["timestamp"]), conversation_id=value.get("harness_conversation_id"), parent_run_id=value.get("parent_step_run_id"), agent_name=value.get("agent_name"), tool_call_id=value.get("tool_call_id"), tool_name=value.get("tool_name"), error=value.get("error"), metadata=dict(value.get("metadata_json") or {}))


async def _snapshot_json(snapshot: ContinuableSnapshot, object_store: ObjectStore, media_prefix: str) -> dict[str, object]:
    messages = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages))
    await _externalize_media(messages, object_store, media_prefix)
    return {"run_id": snapshot.run_id, "step_index": snapshot.step_index, "messages": messages, "conversation_id": snapshot.conversation_id, "parent_run_id": snapshot.parent_run_id, "agent_name": snapshot.agent_name, "timestamp": snapshot.timestamp.astimezone(timezone.utc).isoformat(), "state": snapshot.state, "object_store_id": object_store.store_id}


async def _snapshot_from_json(value: dict[str, object], object_store: ObjectStore) -> ContinuableSnapshot:
    object_store_id = value.get("object_store_id")
    if not isinstance(object_store_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if object_store_id != object_store.store_id:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    messages = value.get("messages", [])
    await _materialize_media(messages, object_store)
    return ContinuableSnapshot(run_id=str(value["run_id"]), step_index=int(value["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(messages), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), timestamp=_datetime(value["timestamp"]), state=value.get("state", "complete"))


async def _snapshot_from_sql(value: Mapping[str, object], object_store: ObjectStore) -> ContinuableSnapshot:
    object_store_id = value.get("object_store_id")
    if not isinstance(object_store_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if object_store_id != object_store.store_id:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    messages = value.get("messages_json") or []
    await _materialize_media(messages, object_store)
    return ContinuableSnapshot(run_id=str(value["step_run_id"]), step_index=int(value["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(messages), conversation_id=value.get("harness_conversation_id"), parent_run_id=value.get("parent_step_run_id"), agent_name=value.get("agent_name"), timestamp=_datetime(value["timestamp"]), state=value.get("state", "complete"))


def _snapshot_row_matches(row: Mapping[str, object], payload: Mapping[str, object], snapshot: ContinuableSnapshot) -> bool:
    return (
        str(row["step_run_id"]) == snapshot.run_id
        and int(row["step_index"]) == snapshot.step_index
        and row.get("messages_json") == payload.get("messages")
        and row.get("harness_conversation_id") == snapshot.conversation_id
        and row.get("parent_step_run_id") == snapshot.parent_run_id
        and row.get("agent_name") == snapshot.agent_name
        and _datetime(row["timestamp"]) == snapshot.timestamp.astimezone(timezone.utc)
        and str(row["state"]) == snapshot.state
        and row.get("object_store_id") == payload.get("object_store_id")
    )


async def _externalize_media(value: object, object_store: ObjectStore, media_prefix: str) -> None:
    if isinstance(value, list):
        for item in value:
            await _externalize_media(item, object_store, media_prefix)
        return
    if not isinstance(value, dict):
        return
    if value.get("kind") == "binary":
        encoded = value.get("data")
        if not isinstance(encoded, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        digest = hashlib.sha256(data).hexdigest()
        key = f"{media_prefix}/{digest}"

        async def chunks() -> AsyncIterator[bytes]:
            yield data

        await object_store.put(key, chunks(), expected_size=len(data), expected_digest=digest)
        value.pop("data", None)
        value["object_store_id"] = object_store.store_id
        value["object_key"] = key
        value["object_digest"] = digest
        value["object_size"] = len(data)
        return
    for item in value.values():
        await _externalize_media(item, object_store, media_prefix)


async def _materialize_media(value: object, object_store: ObjectStore) -> None:
    if isinstance(value, list):
        for item in value:
            await _materialize_media(item, object_store)
        return
    if not isinstance(value, dict):
        return
    if value.get("kind") == "binary":
        store_id = value.get("object_store_id")
        key = value.get("object_key")
        digest = value.get("object_digest")
        size = value.get("object_size")
        if not isinstance(store_id, str) or not isinstance(key, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if store_id != object_store.store_id:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        try:
            data = await read_object(object_store, key, expected_digest=digest, expected_size=size)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_NOT_FOUND:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise
        value["data"] = base64.b64encode(data).decode("ascii")
        for field in ("object_store_id", "object_key", "object_digest", "object_size"):
            value.pop(field, None)
        return
    for item in value.values():
        await _materialize_media(item, object_store)


def _effect_json(record: ToolEffectRecord) -> dict[str, object]:
    return {"tool_call_id": record.tool_call_id, "tool_name": record.tool_name, "run_id": record.run_id, "status": record.status, "started_at": record.started_at.astimezone(timezone.utc).isoformat(), "ended_at": None if record.ended_at is None else record.ended_at.astimezone(timezone.utc).isoformat(), "idempotency_key": record.idempotency_key, "effect_summary": record.effect_summary}


def _effect_from_json(value: dict[str, object]) -> ToolEffectRecord:
    return ToolEffectRecord(tool_call_id=str(value["tool_call_id"]), tool_name=str(value["tool_name"]), run_id=str(value["run_id"]), status=value["status"], started_at=_datetime(value["started_at"]), ended_at=None if value.get("ended_at") is None else _datetime(value["ended_at"]), idempotency_key=value.get("idempotency_key"), effect_summary=value.get("effect_summary"))


def _effect_from_sql(value: Mapping[str, object]) -> ToolEffectRecord:
    summary = value.get("effect_summary")
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except json.JSONDecodeError as error:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    return ToolEffectRecord(tool_call_id=str(value["tool_call_id"]), tool_name=str(value["tool_name"]), run_id=str(value["step_run_id"]), status=str(value["status"]), started_at=_datetime(value["started_at"]), ended_at=None if value.get("ended_at") is None else _datetime(value["ended_at"]), idempotency_key=value.get("idempotency_key"), effect_summary=summary)


def _datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_uri(uri: str) -> tuple[str, str]:
    prefix, separator, value = uri.partition("://")
    if separator != "://" or "/" not in value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    store_id, key = value.split("/", 1)
    return store_id, key


__all__ = [
    "FilesystemStepArchive",
    "InMemoryStepArchive",
    "ObjectMediaAdapter",
    "RuntimeStepPersistence",
    "SqlStepArchive",
    "StagingStepStore",
]
