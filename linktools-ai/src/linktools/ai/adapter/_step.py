#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime Step staging, promotion, and explicit archive implementations."""

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
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
from ..runtime import RuntimeDomain
from ..storage import (
    FilesystemMutationLock,
    FilesystemObjectStore,
    FilesystemWriterLock,
    ObjectStore,
    SqlObjectStore,
    create_sql_context,
    provision_sql,
    read_object,
    validate_sql,
    write_json_atomic,
)
from ._schema import build_step_sql_metadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..storage import SqlContext


_logger = environ.get_logger("ai.adapter.step")
_ARCHIVE_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY})


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
        async with self._lock:
            if record.run_id in self._runs and self._runs[record.run_id] != record:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[record.run_id] = record

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        values = [item for item in self._runs.values() if (parent_run_id is None or item.parent_run_id == parent_run_id) and (conversation_id is None or item.conversation_id == conversation_id)]
        return sorted(values, key=lambda item: (item.started_at, item.run_id))

    async def append_event(self, event: StepEvent) -> None:
        async with self._lock:
            values = self._events.setdefault(event.run_id, [])
            if event in values:
                return
            values.append(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return list(self._events.get(run_id, ()))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        async with self._lock:
            values = self._snapshots.setdefault(snapshot.run_id, [])
            if snapshot in values:
                return
            if values and values[-1] == snapshot:
                return
            values.append(snapshot)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        values = [item for item in self._snapshots.get(run_id, ()) if include_interrupted or item.state == "complete"]
        return values[-1] if values else None

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        async with self._lock:
            values = self._effects.setdefault(record.run_id, [])
            if record in values:
                return
            if values and values[-1] == record:
                return
            values.append(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        values = [item for item in self._effects.get(run_id, ()) if item.tool_call_id == tool_call_id]
        return values[-1] if values else None

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        latest: dict[str, ToolEffectRecord] = {}
        for item in self._effects.get(run_id, ()):
            latest[item.tool_call_id] = item
        return [item for item in latest.values() if item.status == "started"]


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


class SqlStepArchive(StagingStepStore):
    """Explicit SQL Step archive boundary; schema is owned by the Runtime adapter."""

    def __init__(self, engine: "AsyncEngine", *, namespace: str, tenant_id: str, runtime_domain: RuntimeDomain, object_store: ObjectStore | None = None) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        super().__init__()
        self.engine = engine
        self.namespace = namespace
        self.tenant_id = tenant_id
        self.runtime_domain = runtime_domain
        self.object_store = object_store or SqlObjectStore(engine)
        self._provision = True
        self._context = create_sql_context(engine)
        self._owns_context = True
        self._metadata = build_step_sql_metadata(
            runtime_domain,
            object_store=None if self.object_store.store_id == "builtin" else self.object_store,
        )
        self._ready = False

    @classmethod
    def _runtime(
        cls,
        engine: "AsyncEngine",
        *,
        namespace: str,
        tenant_id: str,
        runtime_domain: RuntimeDomain,
        object_store: ObjectStore,
        context: "SqlContext",
    ) -> "SqlStepArchive":
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        archive = cls.__new__(cls)
        StagingStepStore.__init__(archive)
        archive.engine = engine
        archive.namespace = namespace
        archive.tenant_id = tenant_id
        archive.runtime_domain = runtime_domain
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
        if self._provision:
            await provision_sql(self.engine, self._metadata)
        else:
            await validate_sql(self.engine, self._metadata)
        await self._context.initialize()
        await self._load()
        self._ready = True
        _logger.debug("SQL step archive initialized: domain=%s", self.runtime_domain.value)

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_ready()
        from sqlalchemy import insert, select

        table = self._metadata.tables["step_runs"]
        key = _namespace_key(self.namespace)
        async with self._begin() as connection:
            current = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.runtime_domain == self.runtime_domain.value, table.c.step_run_id == record.run_id))).mappings().first()
            if current is not None:
                if _run_from_sql(current) != record:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            else:
                await connection.execute(insert(table).values(namespace_key=key, tenant_id=self.tenant_id, runtime_domain=self.runtime_domain.value, step_run_id=record.run_id, harness_conversation_id=record.conversation_id, parent_step_run_id=record.parent_run_id, agent_name=record.agent_name, metadata_json=dict(record.metadata), last_event_index=0, last_snapshot_index=0, last_effect_index=0, started_at=record.started_at))
        await super().register_run(record)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_ready()
        if self.runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if event in self._events.get(event.run_id, ()):
            return
        from sqlalchemy import insert, select, update

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        events = self._metadata.tables["step_events"]
        async with self._begin() as connection:
            row = (await connection.execute(select(runs.c.last_event_index).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self.runtime_domain.value, runs.c.step_run_id == event.run_id).with_for_update())).first()
            if row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            index = int(row[0]) + 1
            await connection.execute(insert(events).values(namespace_key=key, tenant_id=self.tenant_id, step_run_id=event.run_id, event_index=index, event_kind=event.kind, step_index=event.step_index, timestamp=event.timestamp, harness_conversation_id=event.conversation_id, parent_step_run_id=event.parent_run_id, agent_name=event.agent_name, tool_call_id=event.tool_call_id, tool_name=event.tool_name, error=event.error, metadata_json=dict(event.metadata)))
            await connection.execute(update(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self.runtime_domain.value, runs.c.step_run_id == event.run_id).values(last_event_index=index))
        await super().append_event(event)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_ready()
        if snapshot in self._snapshots.get(snapshot.run_id, ()):
            return
        from sqlalchemy import insert, select, update

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        snapshots = self._metadata.tables["step_snapshots"]
        payload = await _snapshot_json(snapshot, self.object_store, self._media_prefix())
        async with self._begin() as connection:
            row = (await connection.execute(select(runs.c.last_snapshot_index).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self.runtime_domain.value, runs.c.step_run_id == snapshot.run_id).with_for_update())).first()
            if row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            index = int(row[0]) + 1
            previous_row = (await connection.execute(select(snapshots).where(snapshots.c.namespace_key == key, snapshots.c.tenant_id == self.tenant_id, snapshots.c.runtime_domain == self.runtime_domain.value, snapshots.c.step_run_id == snapshot.run_id).order_by(snapshots.c.snapshot_index.desc()).limit(1))).mappings().first()
            if previous_row is not None and await _snapshot_from_sql(previous_row, self.object_store) == snapshot:
                return
            await connection.execute(insert(snapshots).values(namespace_key=key, tenant_id=self.tenant_id, runtime_domain=self.runtime_domain.value, step_run_id=snapshot.run_id, snapshot_index=index, step_index=snapshot.step_index, state=snapshot.state, harness_conversation_id=snapshot.conversation_id, parent_step_run_id=snapshot.parent_run_id, agent_name=snapshot.agent_name, timestamp=snapshot.timestamp, object_store_id=self.object_store.store_id, messages_json=payload["messages"]))
            await connection.execute(update(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self.runtime_domain.value, runs.c.step_run_id == snapshot.run_id).values(last_snapshot_index=index))
        await super().save_snapshot(snapshot)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_ready()
        if self.runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record in self._effects.get(record.run_id, ()):
            return
        from sqlalchemy import insert, select, update

        key = _namespace_key(self.namespace)
        runs = self._metadata.tables["step_runs"]
        effects = self._metadata.tables["step_effects"]
        async with self._begin() as connection:
            row = (await connection.execute(select(runs.c.last_effect_index).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self.runtime_domain.value, runs.c.step_run_id == record.run_id).with_for_update())).first()
            if row is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            index = int(row[0]) + 1
            previous_row = (await connection.execute(select(effects).where(effects.c.namespace_key == key, effects.c.tenant_id == self.tenant_id, effects.c.step_run_id == record.run_id).order_by(effects.c.effect_index.desc()).limit(1))).mappings().first()
            if previous_row is not None and _effect_from_sql(previous_row) == record:
                return
            await connection.execute(insert(effects).values(namespace_key=key, tenant_id=self.tenant_id, step_run_id=record.run_id, effect_index=index, tool_call_id=record.tool_call_id, tool_name=record.tool_name, status=record.status, started_at=record.started_at, ended_at=record.ended_at, idempotency_key=record.idempotency_key, effect_summary=json.dumps(record.effect_summary, sort_keys=True, separators=(",", ":"))))
            await connection.execute(update(runs).where(runs.c.namespace_key == key, runs.c.tenant_id == self.tenant_id, runs.c.runtime_domain == self.runtime_domain.value, runs.c.step_run_id == record.run_id).values(last_effect_index=index))
        await super().record_tool_effect(record)

    async def _load(self) -> None:
        from sqlalchemy import select

        key = _namespace_key(self.namespace)
        async with self._connect() as connection:
            runs_table = self._metadata.tables["step_runs"]
            runs = (await connection.execute(select(runs_table).where(runs_table.c.namespace_key == key, runs_table.c.tenant_id == self.tenant_id, runs_table.c.runtime_domain == self.runtime_domain.value))).mappings().all()
            events = ()
            snapshots = ()
            effects = ()
            if "step_events" in self._metadata.tables:
                table = self._metadata.tables["step_events"]
                query = select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id)
                events = (await connection.execute(query.order_by(table.c.step_run_id, table.c.event_index))).mappings().all()
            if "step_snapshots" in self._metadata.tables:
                table = self._metadata.tables["step_snapshots"]
                snapshots = (await connection.execute(select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id, table.c.runtime_domain == self.runtime_domain.value).order_by(table.c.step_run_id, table.c.snapshot_index))).mappings().all()
            if "step_effects" in self._metadata.tables:
                table = self._metadata.tables["step_effects"]
                query = select(table).where(table.c.namespace_key == key, table.c.tenant_id == self.tenant_id)
                effects = (await connection.execute(query.order_by(table.c.step_run_id, table.c.effect_index))).mappings().all()
        run_by_id = {str(row["step_run_id"]): row for row in runs}
        if any(str(row["step_run_id"]) not in run_by_id for row in (*events, *snapshots, *effects)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.runtime_domain is not RuntimeDomain.EXECUTION and events:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.runtime_domain is not RuntimeDomain.RECOVERY and effects:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        event_groups = _group_step_rows(events)
        snapshot_groups = _group_step_rows(snapshots)
        effect_groups = _group_step_rows(effects)
        for row in runs:
            run_id = str(row["step_run_id"])
            try:
                record = _run_from_sql(row)
                _validate_step_indexes(
                    int(row["last_event_index"]),
                    event_groups.get(run_id, ()),
                    "event_index",
                )
                _validate_step_indexes(
                    int(row["last_snapshot_index"]),
                    snapshot_groups.get(run_id, ()),
                    "snapshot_index",
                )
                _validate_step_indexes(
                    int(row["last_effect_index"]),
                    effect_groups.get(run_id, ()),
                    "effect_index",
                )
            except AIError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            await super().register_run(record)
        for row in events:
            try:
                event = _event_from_sql(row)
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            await super().append_event(event)
        for row in snapshots:
            try:
                snapshot = await _snapshot_from_sql(row, self.object_store)
            except AIError:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            await super().save_snapshot(snapshot)
        for row in effects:
            try:
                effect = _effect_from_sql(row)
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            await super().record_tool_effect(effect)

    async def close(self) -> None:
        self._ready = False
        if self._owns_context:
            await self._context.close()

    async def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _media_prefix(self) -> str:
        return f"v1/step/{self.runtime_domain.value}/{_namespace_key(self.namespace)}/{_scope_key(self.tenant_id)}"

    @asynccontextmanager
    async def _begin(self) -> AsyncIterator[object]:
        async with self._context.sessions.begin() as session:
            yield session

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[object]:
        async with self._context.sessions() as session:
            yield session


class StepPromoter:
    """Promote staged facts once, preserving their original append order."""

    def __init__(self, archive: StepStore) -> None:
        self._archive = archive

    async def promote(self, staging: StagingStepStore, *, run_id: str) -> None:
        run = await staging.get_run(run_id=run_id)
        if run is not None:
            await self._archive.register_run(run)
        for event in await staging.list_events(run_id=run_id):
            await self._archive.append_event(event)
        for snapshot in staging._snapshots.get(run_id, ()):
            await self._archive.save_snapshot(snapshot)
        for effect in staging._effects.get(run_id, ()):
            await self._archive.record_tool_effect(effect)


class ArchiveStepPromoter:
    """Copy the terminal facts of one archive into another archive."""

    def __init__(self, source: StepStore, target: StepStore) -> None:
        self._source = source
        self._target = target

    async def promote(self, *, run_id: str) -> None:
        run = await self._source.get_run(run_id=run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await self._target.register_run(run)
        snapshot = await self._source.latest_snapshot(run_id=run_id)
        if snapshot is None or snapshot.state != "complete":
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await self._target.save_snapshot(snapshot)


class RuntimeStepPersistence(StagingStepStore):
    """Runtime-owned staging surface with explicit per-domain promotion."""

    def __init__(self, staging: StagingStepStore, archives: Mapping[RuntimeDomain, StepStore]) -> None:
        super().__init__()
        self._runs = staging._runs
        self._events = staging._events
        self._snapshots = staging._snapshots
        self._effects = staging._effects
        self._lock = staging._lock
        self.archives = dict(archives)
        self._event_domains: dict[str, set[RuntimeDomain]] = {}
        self._snapshot_domains: dict[str, list[set[RuntimeDomain]]] = {}
        self._effect_domains: dict[str, set[RuntimeDomain]] = {}

    async def initialize(self) -> None:
        await super().initialize()
        initialized: list[StepStore] = []
        try:
            for archive in self.archives.values():
                await archive.initialize()
                initialized.append(archive)
        except BaseException:
            for archive in reversed(initialized):
                await archive.close()
            await super().close()
            raise

    async def close(self) -> None:
        for archive in reversed(tuple(self.archives.values())):
            await archive.close()
        await super().close()

    async def release(self, run_id: str, runtime_domains: frozenset[RuntimeDomain]) -> None:
        async with self._lock:
            event_owners = self._event_domains.get(run_id, set())
            if not event_owners or event_owners <= runtime_domains:
                self._events.pop(run_id, None)
                self._event_domains.pop(run_id, None)
            effect_owners = self._effect_domains.get(run_id, set())
            if not effect_owners or effect_owners <= runtime_domains:
                self._effects.pop(run_id, None)
                self._effect_domains.pop(run_id, None)
            snapshots = self._snapshots.get(run_id, [])
            snapshot_domains = self._snapshot_domains.get(run_id, [])
            kept_snapshots: list[ContinuableSnapshot] = []
            kept_domains: list[set[RuntimeDomain]] = []
            for snapshot, owners in zip(snapshots, snapshot_domains, strict=False):
                if not owners or owners <= runtime_domains:
                    continue
                kept_snapshots.append(snapshot)
                kept_domains.append(owners)
            if kept_snapshots:
                self._snapshots[run_id] = kept_snapshots
                self._snapshot_domains[run_id] = kept_domains
            else:
                self._snapshots.pop(run_id, None)
                self._snapshot_domains.pop(run_id, None)
            if not self._events.get(run_id) and not self._snapshots.get(run_id) and not self._effects.get(run_id):
                self._runs.pop(run_id, None)
        _logger.debug("transient step staging released: run=%s", run_id)

    def mark_snapshot_owner(self, run_id: str, runtime_domain: RuntimeDomain) -> None:
        owners = self._snapshot_domains.setdefault(run_id, [])
        snapshots = self._snapshots.get(run_id, [])
        while len(owners) < len(snapshots):
            owners.append(set())
        if snapshots:
            owners[len(snapshots) - 1].add(runtime_domain)

    def mark_event_owner(self, run_id: str, runtime_domain: RuntimeDomain) -> None:
        self._event_domains.setdefault(run_id, set()).add(runtime_domain)

    def mark_effect_owner(self, run_id: str, runtime_domain: RuntimeDomain) -> None:
        self._effect_domains.setdefault(run_id, set()).add(runtime_domain)

    async def promote(self, runtime_domain: RuntimeDomain, run_id: str) -> None:
        archive = self.archives.get(runtime_domain)
        if archive is None:
            return
        await StepPromoter(archive).promote(self, run_id=run_id)
        _logger.debug("runtime step facts promoted: domain=%s run=%s", runtime_domain.value, run_id)

    async def promote_from(self, source_domain: RuntimeDomain, target_domain: RuntimeDomain, run_id: str) -> None:
        source = self.archives.get(source_domain)
        target = self.archives.get(target_domain)
        if source is None or target is None:
            return
        await ArchiveStepPromoter(source, target).promote(run_id=run_id)
        _logger.debug(
            "runtime archive facts promoted: source=%s target=%s run=%s",
            source_domain.value,
            target_domain.value,
            run_id,
        )

    def writer(self, runtime_domain: RuntimeDomain) -> StepStore:
        """Return the staging writer that promotes finalized facts for one owner."""
        return _PromotingStepStore(self, runtime_domain)

    def archive(self, runtime_domain: RuntimeDomain) -> StepStore:
        archive = self.archives.get(runtime_domain)
        if archive is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return archive


class _PromotingStepStore:
    def __init__(self, staging: RuntimeStepPersistence, runtime_domain: RuntimeDomain) -> None:
        self._staging = staging
        self._runtime_domain = runtime_domain

    async def register_run(self, record: RunRecord) -> None:
        await self._staging.register_run(record)
        await self._staging.promote(self._runtime_domain, run_id=record.run_id)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await self._staging.get_run(run_id=run_id)

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        return await self._staging.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def append_event(self, event: StepEvent) -> None:
        await self._staging.append_event(event)
        self._staging.mark_event_owner(event.run_id, self._runtime_domain)
        await self._staging.promote(self._runtime_domain, run_id=event.run_id)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await self._staging.list_events(run_id=run_id)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        before = len(self._staging._snapshots.get(snapshot.run_id, ()))
        await self._staging.save_snapshot(snapshot)
        if len(self._staging._snapshots.get(snapshot.run_id, ())) > before:
            self._staging.mark_snapshot_owner(snapshot.run_id, self._runtime_domain)
        await self._staging.promote(self._runtime_domain, run_id=snapshot.run_id)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        return await self._staging.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._staging.record_tool_effect(record)
        self._staging.mark_effect_owner(record.run_id, self._runtime_domain)
        await self._staging.promote(self._runtime_domain, run_id=record.run_id)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await self._staging.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await self._staging.list_unresolved_tool_effects(run_id=run_id)


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
        _, key = _parse_uri(uri)
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
    "ObjectMediaAdapter",
    "RuntimeStepPersistence",
    "SqlStepArchive",
    "StagingStepStore",
    "StepPromoter",
]
