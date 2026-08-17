#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime Step staging and explicit owner archive implementations."""

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
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

from ...core import (
    canonical_identity_digest,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode
from ...storage import (
    FilesystemMutationLock,
    FilesystemObjectStore,
    ObjectStore,
    SqlObjectStore,
    SqlStorageContext,
    build_object_sql_metadata,
    create_sql_storage_context,
    dialect_for_name,
    is_retryable_sql_transaction,
    read_object,
    write_json_atomic,
)
from ._filesystem import _filesystem_scope_root
from ._plan import RuntimeDomain, RuntimeRetentionMode
from ._schema import build_step_sql_metadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine



_logger = environ.get_logger("ai.runtime.state.steps")
_ARCHIVE_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY})
_ARCHIVE_ORDER = (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)
_SQL_RUN_BATCH_LIMIT = 256
_SQL_OPTIMISTIC_RETRY_LIMIT = 8


class _StepRetry(Exception):
    pass


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

    def _fact_run_ids(self) -> frozenset[str]:
        return frozenset(
            set(self._runs)
            | set(self._events)
            | set(self._snapshots)
            | set(self._effects)
        )

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
    """Hydrated owner archive for volatile and transient Step facts."""

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
        if self._runtime_domain not in _ARCHIVE_DOMAINS:
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

    def __init__(
        self,
        root: str | Path,
        *,
        namespace: str,
        tenant_id: str,
        runtime_domain: RuntimeDomain,
        object_store: ObjectStore | None = None,
    ) -> None:
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        super().__init__()
        physical_root = _filesystem_scope_root(
            Path(root),
            namespace=namespace,
            tenant_id=tenant_id,
        )
        self._configure_scope(
            physical_root,
            runtime_domain=runtime_domain,
            object_store=object_store,
        )

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @classmethod
    def _from_runtime_scope(
        cls,
        scope_root: Path,
        *,
        runtime_domain: RuntimeDomain,
        object_store: ObjectStore,
    ) -> "FilesystemStepArchive":
        archive = cls.__new__(cls)
        StagingStepStore.__init__(archive)
        archive._configure_scope(
            scope_root,
            runtime_domain=runtime_domain,
            object_store=object_store,
        )
        return archive

    def _configure_scope(
        self,
        scope_root: Path,
        *,
        runtime_domain: RuntimeDomain,
        object_store: ObjectStore | None,
    ) -> None:
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        physical_root = scope_root.expanduser().resolve(strict=False)
        self._root = physical_root / "steps"
        self._namespace_digest = physical_root.parent.name
        self._tenant_scope_key = physical_root.name
        self._runtime_domain = runtime_domain
        self._object_store = (
            object_store
            if object_store is not None
            else FilesystemObjectStore(physical_root / "objects")
        )
        self._mutation_lock = FilesystemMutationLock(self._root / "mutation.lock")
        self._counters: dict[str, dict[str, int]] = {}
        self._ready = False

    async def initialize(self) -> None:
        self._ready = False
        try:
            async with self._mutation_lock:
                self._root.mkdir(parents=True, exist_ok=True)
                await self._reload_all_runs_locked()
                self._ready = True
        except OSError as error:
            self._ready = False
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        except BaseException:
            self._ready = False
            raise

        _logger.debug(
            "step archive initialized: domain=%s tenant_scope=%s",
            self._runtime_domain.value,
            self._tenant_scope_key,
        )

    async def close(self) -> None:
        async with self._mutation_lock:
            self._ready = False
            await self._clear_facts()

    async def register_run(self, record: RunRecord) -> None:
        async with self._mutation_lock:
            try:
                self._require_ready_locked()
                await self._reload_run_locked(record.run_id, required=False)
                await super().register_run(record)
                directory = self._run_dir(record.run_id)
                directory.mkdir(parents=True, exist_ok=True)
                counters = self._counters.setdefault(
                    record.run_id,
                    {
                        "last_event_index": 0,
                        "last_snapshot_index": 0,
                        "last_effect_index": 0,
                    },
                )
                await asyncio.to_thread(write_json_atomic, directory / "run.json", _run_json(record, counters), fsync=True)
                _logger.debug(
                    "step archive run registered: domain=%s run=%s",
                    self._runtime_domain.value,
                    record.run_id,
                )
            except BaseException as error:
                unpublished = await self._cleanup_unpublished_run_locked(record.run_id)
                if unpublished:
                    self._runs.pop(record.run_id, None)
                    self._events.pop(record.run_id, None)
                    self._snapshots.pop(record.run_id, None)
                    self._effects.pop(record.run_id, None)
                    self._counters.pop(record.run_id, None)
                else:
                    await self._recover_run_locked(record.run_id)
                _raise_filesystem_storage_error(error)

    async def append_event(self, event: StepEvent) -> None:
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._mutation_lock:
            try:
                self._require_ready_locked()
                await self._reload_run_locked(event.run_id, required=True)
                before = len(self._events.get(event.run_id, ()))
                await super().append_event(event)
                if len(self._events.get(event.run_id, ())) == before:
                    return
                counters = self._counters[event.run_id]
                index = counters["last_event_index"] + 1
                await asyncio.to_thread(write_json_atomic, self._run_dir(event.run_id) / "events" / f"event-{index:020d}.json", _event_json(event), fsync=True)
                counters["last_event_index"] = index
                await asyncio.to_thread(write_json_atomic, self._run_dir(event.run_id) / "run.json", _run_json(self._runs[event.run_id], counters), fsync=True)
                _logger.debug(
                    "step archive event appended: domain=%s run=%s index=%s",
                    self._runtime_domain.value,
                    event.run_id,
                    index,
                )
            except BaseException as error:
                await self._recover_run_locked(event.run_id)
                _raise_filesystem_storage_error(error)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self._runtime_domain not in _ARCHIVE_DOMAINS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._mutation_lock:
            try:
                self._require_ready_locked()
                await self._reload_run_locked(snapshot.run_id, required=True)
                before = len(self._snapshots.get(snapshot.run_id, ()))
                await super().save_snapshot(snapshot)
                if len(self._snapshots.get(snapshot.run_id, ())) == before:
                    return
                counters = self._counters[snapshot.run_id]
                index = counters["last_snapshot_index"] + 1
                payload = await _snapshot_json(snapshot, self._object_store, self._media_prefix())
                await asyncio.to_thread(write_json_atomic, self._run_dir(snapshot.run_id) / "snapshots" / f"snapshot-{index:020d}.json", payload, fsync=True)
                counters["last_snapshot_index"] = index
                await asyncio.to_thread(write_json_atomic, self._run_dir(snapshot.run_id) / "run.json", _run_json(self._runs[snapshot.run_id], counters), fsync=True)
                _logger.debug(
                    "step archive snapshot saved: domain=%s run=%s index=%s",
                    self._runtime_domain.value,
                    snapshot.run_id,
                    index,
                )
            except BaseException as error:
                await self._recover_run_locked(snapshot.run_id)
                _raise_filesystem_storage_error(error)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with self._mutation_lock:
            try:
                self._require_ready_locked()
                await self._reload_run_locked(record.run_id, required=True)
                before = len(self._effects.get(record.run_id, ()))
                await super().record_tool_effect(record)
                if len(self._effects.get(record.run_id, ())) == before:
                    return
                counters = self._counters[record.run_id]
                index = counters["last_effect_index"] + 1
                await asyncio.to_thread(write_json_atomic, self._run_dir(record.run_id) / "effects" / f"effect-{index:020d}.json", _effect_json(record), fsync=True)
                counters["last_effect_index"] = index
                await asyncio.to_thread(write_json_atomic, self._run_dir(record.run_id) / "run.json", _run_json(self._runs[record.run_id], counters), fsync=True)
                _logger.debug(
                    "step archive tool effect recorded: domain=%s run=%s index=%s",
                    self._runtime_domain.value,
                    record.run_id,
                    index,
                )
            except BaseException as error:
                await self._recover_run_locked(record.run_id)
                _raise_filesystem_storage_error(error)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            record = self._runs.get(run_id)
            return None if record is None else deepcopy(record)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_all_runs_locked()
            return sorted(
                (
                    deepcopy(item)
                    for item in self._runs.values()
                    if (parent_run_id is None or item.parent_run_id == parent_run_id)
                    and (conversation_id is None or item.conversation_id == conversation_id)
                ),
                key=lambda item: (item.started_at, item.run_id),
            )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            return [deepcopy(item) for item in self._events.get(run_id, ())]

    async def latest_snapshot(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            values = self._snapshots.get(run_id, ())
            if not values:
                return None
            snapshot = values[-1]
            if not include_interrupted and snapshot.state != "complete":
                return None
            return deepcopy(snapshot)

    async def get_tool_effect(
        self,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> ToolEffectRecord | None:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            values = [
                item
                for item in self._effects.get(run_id, ())
                if item.tool_call_id == tool_call_id
            ]
            return None if not values else deepcopy(values[-1])

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            latest: dict[str, ToolEffectRecord] = {}
            for item in self._effects.get(run_id, ()):
                latest[item.tool_call_id] = item
            return [
                deepcopy(item)
                for item in latest.values()
                if item.status == "started"
            ]

    async def _fact_projection(
        self,
        run_id: str,
    ) -> tuple[
        RunRecord | None,
        tuple[StepEvent, ...],
        tuple[ContinuableSnapshot, ...],
        tuple[ToolEffectRecord, ...],
    ]:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            return (
                deepcopy(self._runs.get(run_id)),
                tuple(deepcopy(item) for item in self._events.get(run_id, ())),
                tuple(deepcopy(item) for item in self._snapshots.get(run_id, ())),
                tuple(deepcopy(item) for item in self._effects.get(run_id, ())),
            )

    async def release_run(self, run_id: str) -> None:
        async with self._mutation_lock:
            self._require_ready_locked()
            await self._reload_run_locked(run_id, required=False)
            self._runs.pop(run_id, None)
            self._events.pop(run_id, None)
            self._snapshots.pop(run_id, None)
            self._effects.pop(run_id, None)
            self._counters.pop(run_id, None)
            _logger.debug(
                "step archive run released from cache: domain=%s run=%s",
                self._runtime_domain.value,
                run_id,
            )

    async def _load_run(self, directory: Path) -> None:
        run_path = directory / "run.json"
        if not run_path.is_file():
            if not any(directory.iterdir()):
                directory.rmdir()
                return
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
            try:
                await asyncio.to_thread(
                    write_json_atomic,
                    run_path,
                    _run_json(record, counters),
                    fsync=True,
                )
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def _reload_run_locked(self, run_id: str, *, required: bool) -> None:
        try:
            directory = self._run_dir(run_id)
            self._runs.pop(run_id, None)
            self._events.pop(run_id, None)
            self._snapshots.pop(run_id, None)
            self._effects.pop(run_id, None)
            self._counters.pop(run_id, None)
            if not directory.exists():
                if required:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                return
            if not (directory / "run.json").is_file():
                if await self._cleanup_unpublished_run_locked(run_id):
                    if required:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    return
            await self._load_run(directory)
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def _reload_all_runs_locked(self) -> None:
        try:
            await self._clear_facts()
            runs_root = self._root / "runs"
            if not runs_root.exists():
                return
            for directory in sorted(runs_root.iterdir()):
                if directory.is_dir():
                    await self._load_run(directory)
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def _recover_run_locked(self, run_id: str) -> None:
        try:
            await self._reload_run_locked(run_id, required=False)
        except BaseException:
            self._runs.pop(run_id, None)
            self._events.pop(run_id, None)
            self._snapshots.pop(run_id, None)
            self._effects.pop(run_id, None)
            self._counters.pop(run_id, None)
            raise

    async def _cleanup_unpublished_run_locked(self, run_id: str) -> bool:
        directory = self._run_dir(run_id)
        if not directory.is_dir() or (directory / "run.json").exists():
            return False
        if any(directory.iterdir()):
            return False
        directory.rmdir()
        self._runs.pop(run_id, None)
        self._events.pop(run_id, None)
        self._snapshots.pop(run_id, None)
        self._effects.pop(run_id, None)
        self._counters.pop(run_id, None)
        _logger.info(
            "removed unpublished empty step run: domain=%s run=%s",
            self._runtime_domain.value,
            run_id,
        )
        return True

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(character in run_id for character in "/\\\x00"):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return self._root / "runs" / _digest(run_id)

    def _media_prefix(self) -> str:
        return f"v1/step/{self._runtime_domain.value}/{self._namespace_digest}/{self._tenant_scope_key}"

    def _require_ready_locked(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


class SqlStepArchive(StepStore):
    """Explicit SQL Step archive boundary owned by RuntimeState."""

    def __init__(self, engine: "AsyncEngine", *, namespace: str, tenant_id: str, runtime_domain: RuntimeDomain, object_store: ObjectStore | None = None) -> None:
        from sqlalchemy.ext.asyncio import AsyncEngine

        dialect_for_name(engine.dialect.name)
        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_persistence_namespace(namespace)
        validate_tenant_id(tenant_id)
        if runtime_domain not in _ARCHIVE_DOMAINS:
            raise ValueError("Step archive owner is invalid")
        builtin_object_store = object_store is None
        self.engine = engine
        self.namespace = namespace
        self.tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self.object_store = SqlObjectStore(engine) if object_store is None else object_store
        self._context = create_sql_storage_context(engine)
        self._owns_context = True
        from sqlalchemy import MetaData

        self._metadata = MetaData()
        build_step_sql_metadata(runtime_domain, metadata=self._metadata)
        if builtin_object_store:
            build_object_sql_metadata(metadata=self._metadata)
        self._ready = False

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @classmethod
    def from_runtime(
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
        archive._context = context
        archive._owns_context = False
        from sqlalchemy import MetaData

        archive._metadata = MetaData()
        build_step_sql_metadata(runtime_domain, metadata=archive._metadata)
        archive._ready = False
        return archive

    async def initialize(self) -> None:
        try:
            await self._context.initialize(metadata=self._metadata)
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

    def _run_epoch(self, row: Mapping[str, object]) -> tuple[int, int, int]:
        values = (
            row.get("last_event_index"),
            row.get("last_snapshot_index"),
            row.get("last_effect_index"),
        )
        try:
            if any(isinstance(value, bool) for value in values):
                raise ValueError
            epoch = tuple(int(value) for value in values)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if any(value < 0 for value in epoch):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return epoch

    async def _read_run_row(
        self,
        connection: object,
        run_id: str,
    ) -> Mapping[str, object] | None:
        from sqlalchemy import select

        table = self._metadata.tables["ai_step_runs"]
        return (
            await connection.execute(
                select(table).where(
                    table.c.namespace_digest == _namespace_digest(self.namespace),
                    table.c.tenant_id == self.tenant_id,
                    table.c.runtime_domain == self._runtime_domain.value,
                    table.c.run_id == run_id,
                )
            )
        ).mappings().first()

    async def _read_run_epochs(
        self,
        connection: object,
        run_ids: Sequence[str],
    ) -> dict[str, tuple[int, int, int]]:
        from sqlalchemy import select

        table = self._metadata.tables["ai_step_runs"]
        result: dict[str, tuple[int, int, int]] = {}
        unique = tuple(dict.fromkeys(run_ids))
        for offset in range(0, len(unique), _SQL_RUN_BATCH_LIMIT):
            batch = unique[offset : offset + _SQL_RUN_BATCH_LIMIT]
            if not batch:
                continue
            rows = (
                await connection.execute(
                    select(
                        table.c.run_id,
                        table.c.last_event_index,
                        table.c.last_snapshot_index,
                        table.c.last_effect_index,
                    ).where(
                        table.c.namespace_digest == _namespace_digest(self.namespace),
                        table.c.tenant_id == self.tenant_id,
                        table.c.runtime_domain == self._runtime_domain.value,
                        table.c.run_id.in_(batch),
                    )
                )
            ).mappings().all()
            for row in rows:
                result[str(row["run_id"])] = self._run_epoch(row)
        return result

    def _archive_families(self) -> tuple[tuple[str, str, object], ...]:
        families: list[tuple[str, str, object]] = []
        if "ai_step_events" in self._metadata.tables:
            families.append(("events", "event_index", self._metadata.tables["ai_step_events"]))
        if "ai_step_snapshots" in self._metadata.tables:
            families.append(("snapshots", "snapshot_index", self._metadata.tables["ai_step_snapshots"]))
        if "ai_step_effects" in self._metadata.tables:
            families.append(("effects", "effect_index", self._metadata.tables["ai_step_effects"]))
        return tuple(families)

    async def _load_index_metrics(
        self,
        connection: object,
        run_ids: Sequence[str],
    ) -> dict[str, dict[str, dict[str, object]]]:
        from sqlalchemy import String, func, literal, select, union_all

        unique = tuple(dict.fromkeys(run_ids))
        families = self._archive_families()
        result = {
            run_id: {
                family: {
                    "row_count": 0,
                    "distinct_count": 0,
                    "min_index": None,
                    "max_index": None,
                }
                for family, _, _ in families
            }
            for run_id in unique
        }
        for offset in range(0, len(unique), _SQL_RUN_BATCH_LIMIT):
            batch = unique[offset : offset + _SQL_RUN_BATCH_LIMIT]
            if not batch or not families:
                continue
            statements = []
            for family, index_name, table in families:
                conditions = [
                    table.c.namespace_digest == _namespace_digest(self.namespace),
                    table.c.tenant_id == self.tenant_id,
                    table.c.run_id.in_(batch),
                ]
                if family == "snapshots":
                    conditions.append(
                        table.c.runtime_domain == self._runtime_domain.value
                    )
                index_column = table.c[index_name]
                statements.append(
                    select(
                        literal(family, type_=String()).label("family"),
                        table.c.run_id,
                        func.count().label("row_count"),
                        func.count(func.distinct(index_column)).label("distinct_count"),
                        func.min(index_column).label("min_index"),
                        func.max(index_column).label("max_index"),
                    )
                    .where(*conditions)
                    .group_by(table.c.run_id)
                )
            rows = (await connection.execute(union_all(*statements))).mappings().all()
            for row in rows:
                run_id = str(row["run_id"])
                family = str(row["family"])
                if run_id not in result or family not in result[run_id]:
                    continue
                result[run_id][family] = {
                    "row_count": int(row["row_count"]),
                    "distinct_count": int(row["distinct_count"]),
                    "min_index": None if row["min_index"] is None else int(row["min_index"]),
                    "max_index": None if row["max_index"] is None else int(row["max_index"]),
                }
        return result

    def _validate_metrics(
        self,
        run_row: Mapping[str, object],
        metrics: Mapping[str, Mapping[str, object]],
    ) -> None:
        epoch = self._run_epoch(run_row)
        for last, family in zip(
            epoch,
            ("events", "snapshots", "effects"),
            strict=True,
        ):
            if family not in metrics:
                continue
            value = metrics[family]
            expected = {
                "row_count": last,
                "distinct_count": last,
                "min_index": None if last == 0 else 1,
                "max_index": None if last == 0 else last,
            }
            if any(value.get(name) != expected[name] for name in expected):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _stable_run(
        self,
        run_id: str,
    ) -> tuple[Mapping[str, object] | None, dict[str, dict[str, object]]]:
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                row = await self._read_run_row(connection, run_id)
                if row is None:
                    return None, {}
                initial = self._run_epoch(row)
                metrics = (await self._load_index_metrics(connection, (run_id,)))[run_id]
                final = (await self._read_run_epochs(connection, (run_id,))).get(run_id)
                if final is None or final != initial:
                    stable = False
                else:
                    self._validate_metrics(row, metrics)
                    stable = True
            if stable:
                return row, metrics
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_ready()
        row, _ = await self._stable_run(run_id)
        return None if row is None else _run_from_sql(row)

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        await self._ensure_ready()
        from sqlalchemy import select

        table = self._metadata.tables["ai_step_runs"]
        key = _namespace_digest(self.namespace)
        query = select(table).where(
            table.c.namespace_digest == key,
            table.c.tenant_id == self.tenant_id,
            table.c.runtime_domain == self._runtime_domain.value,
        )
        if parent_run_id is not None:
            query = query.where(table.c.parent_run_id == parent_run_id)
        if conversation_id is not None:
            query = query.where(table.c.conversation_id == conversation_id)
        query = query.order_by(table.c.started_at, table.c.run_id)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                rows = (await connection.execute(query)).mappings().all()
                run_ids = tuple(str(row["run_id"]) for row in rows)
                initial_epochs = {
                    str(row["run_id"]): self._run_epoch(row)
                    for row in rows
                }
                metrics = await self._load_index_metrics(connection, run_ids)
                final_epochs = await self._read_run_epochs(connection, run_ids)
                stable = all(
                    final_epochs.get(run_id) == epoch
                    for run_id, epoch in initial_epochs.items()
                )
                if stable:
                    for row in rows:
                        self._validate_metrics(row, metrics[str(row["run_id"])])
                    result = [_run_from_sql(row) for row in rows]
            if stable:
                return result
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_ready()
        if "ai_step_events" not in self._metadata.tables:
            return []
        from sqlalchemy import select

        table = self._metadata.tables["ai_step_events"]
        key = _namespace_digest(self.namespace)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                run_row = await self._read_run_row(connection, run_id)
                rows = (
                    await connection.execute(
                        select(table)
                        .where(
                            table.c.namespace_digest == key,
                            table.c.tenant_id == self.tenant_id,
                            table.c.run_id == run_id,
                        )
                        .order_by(table.c.event_index)
                    )
                ).mappings().all()
                if run_row is None:
                    if rows:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return []
                initial = self._run_epoch(run_row)
                metrics = (await self._load_index_metrics(connection, (run_id,)))[run_id]
                final = (await self._read_run_epochs(connection, (run_id,))).get(run_id)
                stable = final == initial
                if stable:
                    self._validate_metrics(run_row, metrics)
                    result = [_event_from_sql(row) for row in rows]
            if stable:
                return result
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _execution_projection(self, *, run_id: str) -> _ExecutionProjection | None:
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._ensure_ready()
        from sqlalchemy import select

        key = _namespace_digest(self.namespace)
        events = self._metadata.tables["ai_step_events"]
        snapshots = self._metadata.tables["ai_step_snapshots"]
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                run_row = await self._read_run_row(connection, run_id)
                if run_row is None:
                    return None
                event_rows = (
                    await connection.execute(
                        select(events)
                        .where(
                            events.c.namespace_digest == key,
                            events.c.tenant_id == self.tenant_id,
                            events.c.run_id == run_id,
                        )
                        .order_by(events.c.event_index)
                    )
                ).mappings().all()
                snapshot_rows = (
                    await connection.execute(
                        select(snapshots)
                        .where(
                            snapshots.c.namespace_digest == key,
                            snapshots.c.tenant_id == self.tenant_id,
                            snapshots.c.runtime_domain == self._runtime_domain.value,
                            snapshots.c.run_id == run_id,
                        )
                        .order_by(snapshots.c.snapshot_index)
                    )
                ).mappings().all()
                initial = self._run_epoch(run_row)
                final = (await self._read_run_epochs(connection, (run_id,))).get(run_id)
                stable = final == initial
                if stable:
                    _validate_step_indexes(initial[0], event_rows, "event_index")
                    _validate_step_indexes(initial[1], snapshot_rows, "snapshot_index")
                    result = _ExecutionProjection(
                        run=_run_from_sql(run_row),
                        events=tuple(_event_from_sql(row) for row in event_rows),
                        snapshots=tuple(dict(row) for row in snapshot_rows),
                    )
            if stable:
                return _ExecutionProjection(
                    run=result.run,
                    events=result.events,
                    snapshots=tuple(
                        [
                            await _snapshot_from_sql(row, self.object_store)
                            for row in result.snapshots
                        ]
                    ),
                )
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _recovery_effect_projection(self, *, run_id: str) -> _RecoveryEffectProjection | None:
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._ensure_ready()
        from sqlalchemy import select

        key = _namespace_digest(self.namespace)
        effects = self._metadata.tables["ai_step_effects"]
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                run_row = await self._read_run_row(connection, run_id)
                if run_row is None:
                    return None
                effect_rows = (
                    await connection.execute(
                        select(effects)
                        .where(
                            effects.c.namespace_digest == key,
                            effects.c.tenant_id == self.tenant_id,
                            effects.c.run_id == run_id,
                        )
                        .order_by(effects.c.effect_index)
                    )
                ).mappings().all()
                initial = self._run_epoch(run_row)
                final = (await self._read_run_epochs(connection, (run_id,))).get(run_id)
                stable = final == initial
                if stable:
                    _validate_step_indexes(initial[2], effect_rows, "effect_index")
                    result = _RecoveryEffectProjection(
                        run=_run_from_sql(run_row),
                        effects=tuple(_effect_from_sql(row) for row in effect_rows),
                    )
            if stable:
                return result
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_ready()
        from sqlalchemy import and_, func, select

        table = self._metadata.tables["ai_step_snapshots"]
        key = _namespace_digest(self.namespace)
        async with self._begin() as connection:
            runs = self._metadata.tables["ai_step_runs"]
            metrics = (
                select(
                    table.c.namespace_digest.label("metric_namespace_digest"),
                    table.c.tenant_id.label("metric_tenant_id"),
                    table.c.runtime_domain.label("metric_runtime_domain"),
                    table.c.run_id.label("metric_run_id"),
                    func.count().label("metric_row_count"),
                    func.count(func.distinct(table.c.snapshot_index)).label(
                        "metric_distinct_count"
                    ),
                    func.min(table.c.snapshot_index).label("metric_min_index"),
                    func.max(table.c.snapshot_index).label("metric_max_index"),
                )
                .where(
                    table.c.namespace_digest == key,
                    table.c.tenant_id == self.tenant_id,
                    table.c.runtime_domain == self._runtime_domain.value,
                    table.c.run_id == run_id,
                )
                .group_by(
                    table.c.namespace_digest,
                    table.c.tenant_id,
                    table.c.runtime_domain,
                    table.c.run_id,
                )
                .subquery("snapshot_metrics")
            )
            snapshot_columns = tuple(
                table.c[column].label(f"snapshot_{column}")
                for column in table.c.keys()
            )
            statement = (
                select(
                    runs.c.run_id.label("run_id"),
                    runs.c.last_snapshot_index.label("last_snapshot_index"),
                    metrics.c.metric_row_count,
                    metrics.c.metric_distinct_count,
                    metrics.c.metric_min_index,
                    metrics.c.metric_max_index,
                    *snapshot_columns,
                )
                .select_from(
                    runs.outerjoin(
                        metrics,
                        and_(
                            metrics.c.metric_namespace_digest == runs.c.namespace_digest,
                            metrics.c.metric_tenant_id == runs.c.tenant_id,
                            metrics.c.metric_runtime_domain == runs.c.runtime_domain,
                            metrics.c.metric_run_id == runs.c.run_id,
                        ),
                    ).outerjoin(
                        table,
                        and_(
                            table.c.namespace_digest == key,
                            table.c.tenant_id == self.tenant_id,
                            table.c.runtime_domain == self._runtime_domain.value,
                            table.c.run_id == run_id,
                            table.c.snapshot_index == metrics.c.metric_max_index,
                        ),
                    )
                )
                .where(
                    runs.c.namespace_digest == key,
                    runs.c.tenant_id == self.tenant_id,
                    runs.c.runtime_domain == self._runtime_domain.value,
                    runs.c.run_id == run_id,
                )
            )
            row = (await connection.execute(statement)).mappings().first()
            if row is None:
                return None
            last = int(row["last_snapshot_index"])
            row_count = 0 if row["metric_row_count"] is None else int(row["metric_row_count"])
            distinct_count = (
                0
                if row["metric_distinct_count"] is None
                else int(row["metric_distinct_count"])
            )
            min_index = None if row["metric_min_index"] is None else int(row["metric_min_index"])
            max_index = None if row["metric_max_index"] is None else int(row["metric_max_index"])
            snapshot_row = None
            if row["snapshot_id"] is not None:
                snapshot_row = {
                    column: row[f"snapshot_{column}"]
                    for column in table.c.keys()
                }
            if last == 0:
                valid = (
                    row_count == 0
                    and distinct_count == 0
                    and min_index is None
                    and max_index is None
                    and snapshot_row is None
                )
            else:
                valid = (
                    row_count == last
                    and distinct_count == last
                    and min_index == 1
                    and max_index == last
                    and snapshot_row is not None
                    and int(snapshot_row["snapshot_index"]) == last
                )
            if not valid:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            row = snapshot_row
        if row is None or (not include_interrupted and str(row["state"]) != "complete"):
            return None
        return await _snapshot_from_sql(row, self.object_store)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        await self._ensure_ready()
        if "ai_step_effects" not in self._metadata.tables:
            return None
        from sqlalchemy import select

        table = self._metadata.tables["ai_step_effects"]
        key = _namespace_digest(self.namespace)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                run_row = await self._read_run_row(connection, run_id)
                row = (
                    await connection.execute(
                        select(table)
                        .where(
                            table.c.namespace_digest == key,
                            table.c.tenant_id == self.tenant_id,
                            table.c.run_id == run_id,
                            table.c.tool_call_id == tool_call_id,
                        )
                        .order_by(table.c.effect_index.desc())
                        .limit(1)
                    )
                ).mappings().first()
                if run_row is None:
                    if row is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return None
                initial = self._run_epoch(run_row)
                metrics = (await self._load_index_metrics(connection, (run_id,)))[run_id]
                final = (await self._read_run_epochs(connection, (run_id,))).get(run_id)
                stable = final == initial
                if stable:
                    self._validate_metrics(run_row, metrics)
            if stable:
                return None if row is None else _effect_from_sql(row)
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        await self._ensure_ready()
        if "ai_step_effects" not in self._metadata.tables:
            return []
        from sqlalchemy import and_, func, select

        table = self._metadata.tables["ai_step_effects"]
        key = _namespace_digest(self.namespace)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            async with self._begin() as connection:
                run_row = await self._read_run_row(connection, run_id)
                latest = (
                    select(
                        table.c.namespace_digest.label("latest_namespace_digest"),
                        table.c.tenant_id.label("latest_tenant_id"),
                        table.c.run_id.label("latest_run_id"),
                        table.c.tool_call_id.label("latest_tool_call_id"),
                        func.min(table.c.effect_index).label("first_index"),
                        func.max(table.c.effect_index).label("latest_index"),
                    )
                    .where(
                        table.c.namespace_digest == key,
                        table.c.tenant_id == self.tenant_id,
                        table.c.run_id == run_id,
                    )
                    .group_by(
                        table.c.namespace_digest,
                        table.c.tenant_id,
                        table.c.run_id,
                        table.c.tool_call_id,
                    )
                    .subquery("latest_effects")
                )
                effect_columns = tuple(
                    table.c[column].label(f"effect_{column}")
                    for column in table.c.keys()
                )
                statement = (
                    select(*effect_columns, latest.c.first_index)
                    .select_from(
                        latest.join(
                            table,
                            and_(
                                table.c.namespace_digest == latest.c.latest_namespace_digest,
                                table.c.tenant_id == latest.c.latest_tenant_id,
                                table.c.run_id == latest.c.latest_run_id,
                                table.c.tool_call_id == latest.c.latest_tool_call_id,
                                table.c.effect_index == latest.c.latest_index,
                            ),
                        )
                    )
                    .where(table.c.status == "started")
                    .order_by(latest.c.first_index)
                )
                rows = (await connection.execute(statement)).mappings().all()
                if run_row is None:
                    if rows:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return []
                initial = self._run_epoch(run_row)
                metrics = (await self._load_index_metrics(connection, (run_id,)))[run_id]
                final = (await self._read_run_epochs(connection, (run_id,))).get(run_id)
                stable = final == initial
                if stable:
                    self._validate_metrics(run_row, metrics)
                    result = [
                        _effect_from_sql(
                            {
                                column: row[f"effect_{column}"]
                                for column in table.c.keys()
                            }
                        )
                        for row in rows
                    ]
            if stable:
                return result
            if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_ready()
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        table = self._metadata.tables["ai_step_runs"]
        key = _namespace_digest(self.namespace)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            try:
                async with self._begin() as connection:
                    await connection.execute(
                        insert(table).values(
                            namespace_digest=key,
                            tenant_id=self.tenant_id,
                            runtime_domain=self._runtime_domain.value,
                            run_id=record.run_id,
                            identity_digest=_step_identity_digest(
                                self._runtime_domain.value,
                                record.run_id,
                            ),
                            conversation_id=record.conversation_id,
                            parent_run_id=record.parent_run_id,
                            agent_name=record.agent_name,
                            metadata=dict(record.metadata),
                            last_event_index=0,
                            last_snapshot_index=0,
                            last_effect_index=0,
                            started_at=record.started_at,
                        )
                    )
                return
            except IntegrityError as error:
                if self._context.dialect.classify_integrity_error(error).value != "unique_conflict":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                current, _ = await self._stable_run(record.run_id)
                if current is not None and _run_from_sql(current) == record:
                    return
                raise AIError(ErrorCode.STORAGE_CONFLICT) from None
            except BaseException as error:
                if not is_retryable_sql_transaction(error):
                    raise
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                _logger.debug(
                    "SQL step transaction retry: operation=register_run run=%s attempt=%s",
                    record.run_id,
                    attempt + 1,
                )
                await asyncio.sleep(0)

    async def _candidate_rows(
        self,
        connection: object,
        table: object,
        *,
        family: str,
        run_id: str,
        value: object,
    ) -> Sequence[Mapping[str, object]]:
        from sqlalchemy import select

        predicates = [
            table.c.namespace_digest == _namespace_digest(self.namespace),
            table.c.tenant_id == self.tenant_id,
            table.c.run_id == run_id,
        ]
        if family == "event":
            event = value
            fields = {
                "kind": event.kind,
                "step_index": event.step_index,
                "conversation_id": event.conversation_id,
                "parent_run_id": event.parent_run_id,
                "agent_name": event.agent_name,
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
            }
        elif family == "snapshot":
            snapshot, payload = value
            fields = {
                "runtime_domain": self._runtime_domain.value,
                "step_index": snapshot.step_index,
                "state": snapshot.state,
                "conversation_id": snapshot.conversation_id,
                "parent_run_id": snapshot.parent_run_id,
                "agent_name": snapshot.agent_name,
                "media_store_id": payload.get("object_store_id"),
            }
        else:
            record = value
            fields = {
                "tool_call_id": record.tool_call_id,
                "tool_name": record.tool_name,
                "status": record.status,
                "idempotency_key": record.idempotency_key,
            }
        for name, field_value in fields.items():
            column = table.c[name]
            predicates.append(
                column.is_(None) if field_value is None else column == field_value
            )
        return (
            await connection.execute(select(table).where(*predicates))
        ).mappings().all()

    async def _reserve_run_index(
        self,
        connection: object,
        run_row: Mapping[str, object],
        family: str,
    ) -> int:
        from sqlalchemy import update

        counters = self._run_epoch(run_row)
        names = {
            "event": "last_event_index",
            "snapshot": "last_snapshot_index",
            "effect": "last_effect_index",
        }
        column_name = names[family]
        table = self._metadata.tables["ai_step_runs"]
        values = {column_name: counters[("event", "snapshot", "effect").index(family)] + 1}
        statement = update(table).where(
            table.c.namespace_digest == _namespace_digest(self.namespace),
            table.c.tenant_id == self.tenant_id,
            table.c.runtime_domain == self._runtime_domain.value,
            table.c.run_id == str(run_row["run_id"]),
            table.c.last_event_index == counters[0],
            table.c.last_snapshot_index == counters[1],
            table.c.last_effect_index == counters[2],
        ).values(**values)
        result = await connection.execute(statement)
        if result.rowcount == 0:
            raise _StepRetry
        if result.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values[column_name]

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_ready()
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = self._metadata.tables["ai_step_events"]
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        key = _namespace_digest(self.namespace)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            try:
                duplicate = False
                async with self._begin() as connection:
                    run_row = await self._read_run_row(connection, event.run_id)
                    if run_row is None:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    metrics = (await self._load_index_metrics(connection, (event.run_id,)))[event.run_id]
                    current_epoch = self._run_epoch(run_row)
                    final_epoch = (await self._read_run_epochs(connection, (event.run_id,))).get(event.run_id)
                    if final_epoch != current_epoch:
                        raise _StepRetry
                    self._validate_metrics(run_row, metrics)
                    candidates = await self._candidate_rows(
                        connection,
                        events,
                        family="event",
                        run_id=event.run_id,
                        value=event,
                    )
                    duplicate = any(_event_from_sql(row) == event for row in candidates)
                    if not duplicate:
                        index = await self._reserve_run_index(connection, run_row, "event")
                        try:
                            await connection.execute(
                                insert(events).values(
                                    namespace_digest=key,
                                    tenant_id=self.tenant_id,
                                    run_id=event.run_id,
                                    event_index=index,
                                    identity_digest=_step_identity_digest(
                                        "event", event.run_id, index
                                    ),
                                    kind=event.kind,
                                    step_index=event.step_index,
                                    timestamp=event.timestamp,
                                    conversation_id=event.conversation_id,
                                    parent_run_id=event.parent_run_id,
                                    agent_name=event.agent_name,
                                    tool_call_id=event.tool_call_id,
                                    tool_name=event.tool_name,
                                    error=event.error,
                                    metadata=dict(event.metadata),
                                )
                            )
                        except IntegrityError as error:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if duplicate:
                    return
                _logger.debug(
                    "SQL step event appended: run=%s index=%s attempt=%s",
                    event.run_id,
                    index,
                    attempt + 1,
                )
                return
            except _StepRetry:
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await asyncio.sleep(0)
            except BaseException as error:
                if not is_retryable_sql_transaction(error):
                    raise
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                _logger.debug(
                    "SQL step transaction retry: operation=append_event run=%s attempt=%s",
                    event.run_id,
                    attempt + 1,
                )
                await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_ready()
        if self._runtime_domain not in _ARCHIVE_DOMAINS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        preflight, _ = await self._stable_run(snapshot.run_id)
        if preflight is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        payload = await _snapshot_json(
            snapshot,
            self.object_store,
            self._media_prefix(),
        )
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        snapshots = self._metadata.tables["ai_step_snapshots"]
        key = _namespace_digest(self.namespace)
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            try:
                duplicate = False
                async with self._begin() as connection:
                    run_row = await self._read_run_row(connection, snapshot.run_id)
                    if run_row is None:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    metrics = (await self._load_index_metrics(connection, (snapshot.run_id,)))[snapshot.run_id]
                    current_epoch = self._run_epoch(run_row)
                    final_epoch = (await self._read_run_epochs(connection, (snapshot.run_id,))).get(snapshot.run_id)
                    if final_epoch != current_epoch:
                        raise _StepRetry
                    self._validate_metrics(run_row, metrics)
                    candidates = await self._candidate_rows(
                        connection,
                        snapshots,
                        family="snapshot",
                        run_id=snapshot.run_id,
                        value=(snapshot, payload),
                    )
                    duplicate = any(
                        _snapshot_row_matches(row, payload, snapshot)
                        for row in candidates
                    )
                    if not duplicate:
                        index = await self._reserve_run_index(
                            connection,
                            run_row,
                            "snapshot",
                        )
                        try:
                            await connection.execute(
                                insert(snapshots).values(
                                    namespace_digest=key,
                                    tenant_id=self.tenant_id,
                                    runtime_domain=self._runtime_domain.value,
                                    run_id=snapshot.run_id,
                                    snapshot_index=index,
                                    identity_digest=_step_identity_digest(
                                        "snapshot",
                                        self._runtime_domain.value,
                                        snapshot.run_id,
                                        index,
                                    ),
                                    step_index=snapshot.step_index,
                                    state=snapshot.state,
                                    conversation_id=snapshot.conversation_id,
                                    parent_run_id=snapshot.parent_run_id,
                                    agent_name=snapshot.agent_name,
                                    timestamp=snapshot.timestamp,
                                    media_store_id=self.object_store.store_id,
                                    messages=payload["messages"],
                                )
                            )
                        except IntegrityError as error:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if duplicate:
                    return
                _logger.debug(
                    "SQL step snapshot saved: run=%s index=%s attempt=%s",
                    snapshot.run_id,
                    index,
                    attempt + 1,
                )
                return
            except _StepRetry:
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await asyncio.sleep(0)
            except BaseException as error:
                if not is_retryable_sql_transaction(error):
                    raise
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                _logger.debug(
                    "SQL step transaction retry: operation=save_snapshot run=%s attempt=%s",
                    snapshot.run_id,
                    attempt + 1,
                )
                await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_ready()
        if self._runtime_domain is not RuntimeDomain.RECOVERY:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        key = _namespace_digest(self.namespace)
        effects = self._metadata.tables["ai_step_effects"]
        for attempt in range(_SQL_OPTIMISTIC_RETRY_LIMIT):
            try:
                duplicate = False
                async with self._begin() as connection:
                    run_row = await self._read_run_row(connection, record.run_id)
                    if run_row is None:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    metrics = (await self._load_index_metrics(connection, (record.run_id,)))[record.run_id]
                    current_epoch = self._run_epoch(run_row)
                    final_epoch = (await self._read_run_epochs(connection, (record.run_id,))).get(record.run_id)
                    if final_epoch != current_epoch:
                        raise _StepRetry
                    self._validate_metrics(run_row, metrics)
                    candidates = await self._candidate_rows(
                        connection,
                        effects,
                        family="effect",
                        run_id=record.run_id,
                        value=record,
                    )
                    duplicate = any(_effect_from_sql(row) == record for row in candidates)
                    if not duplicate:
                        index = await self._reserve_run_index(
                            connection,
                            run_row,
                            "effect",
                        )
                        try:
                            await connection.execute(
                                insert(effects).values(
                                    namespace_digest=key,
                                    tenant_id=self.tenant_id,
                                    run_id=record.run_id,
                                    effect_index=index,
                                    identity_digest=_step_identity_digest(
                                        "effect", record.run_id, index
                                    ),
                                    tool_call_id=record.tool_call_id,
                                    tool_name=record.tool_name,
                                    status=record.status,
                                    started_at=record.started_at,
                                    ended_at=record.ended_at,
                                    idempotency_key=record.idempotency_key,
                                    effect_summary=json.dumps(
                                        record.effect_summary,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ),
                                )
                            )
                        except IntegrityError as error:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if duplicate:
                    return
                _logger.debug(
                    "SQL step effect recorded: run=%s index=%s attempt=%s",
                    record.run_id,
                    index,
                    attempt + 1,
                )
                return
            except _StepRetry:
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await asyncio.sleep(0)
            except BaseException as error:
                if not is_retryable_sql_transaction(error):
                    raise
                if attempt + 1 == _SQL_OPTIMISTIC_RETRY_LIMIT:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
                _logger.debug(
                    "SQL step transaction retry: operation=record_tool_effect run=%s attempt=%s",
                    record.run_id,
                    attempt + 1,
                )
                await asyncio.sleep(0)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def close(self) -> None:
        self._ready = False
        if self._owns_context:
            await self._context.close()

    async def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def _validate_current_state(self) -> None:
        from sqlalchemy import select

        key = _namespace_digest(self.namespace)
        runs = self._metadata.tables["ai_step_runs"]
        async with self._begin() as connection:
            run_rows = (
                await connection.execute(
                    select(runs).where(
                        runs.c.namespace_digest == key,
                        runs.c.tenant_id == self.tenant_id,
                        runs.c.runtime_domain == self._runtime_domain.value,
                    )
                )
            ).mappings().all()
            run_ids = {str(row["run_id"] ) for row in run_rows}
            event_rows: Sequence[Mapping[str, object]] = ()
            if "ai_step_events" in self._metadata.tables:
                events = self._metadata.tables["ai_step_events"]
                event_rows = (
                    await connection.execute(
                        select(events).where(
                            events.c.namespace_digest == key,
                            events.c.tenant_id == self.tenant_id,
                        ).order_by(events.c.run_id, events.c.event_index)
                    )
                ).mappings().all()
            snapshot_rows: Sequence[Mapping[str, object]] = ()
            if "ai_step_snapshots" in self._metadata.tables:
                snapshots = self._metadata.tables["ai_step_snapshots"]
                snapshot_rows = (
                    await connection.execute(
                        select(snapshots).where(
                            snapshots.c.namespace_digest == key,
                            snapshots.c.tenant_id == self.tenant_id,
                            snapshots.c.runtime_domain == self._runtime_domain.value,
                        ).order_by(snapshots.c.run_id, snapshots.c.snapshot_index)
                    )
                ).mappings().all()
            effect_rows: Sequence[Mapping[str, object]] = ()
            if "ai_step_effects" in self._metadata.tables:
                effects = self._metadata.tables["ai_step_effects"]
                effect_rows = (
                    await connection.execute(
                        select(effects).where(
                            effects.c.namespace_digest == key,
                            effects.c.tenant_id == self.tenant_id,
                        ).order_by(effects.c.run_id, effects.c.effect_index)
                    )
                ).mappings().all()
            for child_rows, index_name in ((event_rows, "event_index"), (snapshot_rows, "snapshot_index"), (effect_rows, "effect_index")):
                if any(str(row["run_id"] ) not in run_ids for row in child_rows):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            grouped_events = _group_step_rows(event_rows)
            grouped_snapshots = _group_step_rows(snapshot_rows)
            grouped_effects = _group_step_rows(effect_rows)
            for run_row in run_rows:
                run_id = str(run_row["run_id"] )
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

    def _media_prefix(self) -> str:
        return f"v1/step/{self._runtime_domain.value}/{_namespace_digest(self.namespace)}/{_scope_key(self.tenant_id)}"

    @asynccontextmanager
    async def _begin(self) -> AsyncIterator[object]:
        async with self._context.sessions.begin() as session:
            yield session


class RuntimeStepStore(StepStore):
    """Own the staging facts and the fixed Step archive routing matrix."""

    def __init__(
        self,
        staging: StagingStepStore,
        *,
        conversation_archive: StepStore,
        execution_archive: "StepStore | None",
        recovery_archive: "StepStore | None",
        conversation_retention: RuntimeRetentionMode,
        execution_retention: RuntimeRetentionMode,
        recovery_retention: RuntimeRetentionMode,
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
    def _validate_archive(runtime_domain: RuntimeDomain, retention: RuntimeRetentionMode, archive: StepStore | None) -> None:
        if not isinstance(retention, RuntimeRetentionMode):
            raise ValueError("Step archive retention is invalid")
        if runtime_domain is RuntimeDomain.CONVERSATION and archive is None:
            raise ValueError("Conversation Step archive is required")
        if retention is RuntimeRetentionMode.TRANSIENT:
            if runtime_domain is RuntimeDomain.CONVERSATION and not isinstance(archive, InMemoryStepArchive):
                raise ValueError("transient Conversation archive must be in-memory")
            if runtime_domain is RuntimeDomain.CONVERSATION and archive.runtime_domain is not runtime_domain:
                raise ValueError("Step archive owner is invalid")
            if runtime_domain is not RuntimeDomain.CONVERSATION and archive is not None:
                raise ValueError("transient Step archive must be absent")
            return
        if retention is RuntimeRetentionMode.VOLATILE:
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
        except BaseException as primary:
            for child in reversed(initialized):
                cleanup = asyncio.create_task(child.close(), name="linktools-step-child-cleanup")
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    try:
                        await asyncio.shield(cleanup)
                    except BaseException:
                        _logger.error("step child cleanup failed after cancellation", exc_info=True)
                except BaseException:
                    _logger.error("step child cleanup failed", exc_info=environ.debug)
            raise primary
        self._initialized = initialized
        self._business_unavailable = False
        _logger.debug("step persistence initialized: children=%s", len(initialized))

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
            await _materialize_snapshot(
                self._staging,
                archive,
                run,
                snapshot,
                require_complete=require_complete,
            )

    async def materialize_conversation(self, *, step_run_id: str) -> None:
        run = await self._staging.get_run(run_id=step_run_id)
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id)
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if run is None or snapshot is None or snapshot.state != "complete" or archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await _materialize_snapshot(self._staging, archive, run, snapshot, require_complete=True)

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
            await _materialize_snapshot(source, destination, source_run, source_snapshot, require_complete=True)
            return
        if run_exists:
            source = recovery if recovery is not None else self._staging
            source_run, source_snapshot = await _read_complete_current_snapshot(source, step_run_id)
            if execution is not None:
                await _materialize_snapshot(source, execution, source_run, source_snapshot, require_complete=True)
            return
        if execution is None and recovery is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if execution is not None and recovery is not None:
            recovery_run, recovery_snapshot = await _read_complete_current_snapshot(recovery, step_run_id)
            recovery_projection = await _read_recovery_projection(recovery, step_run_id)
            if recovery_projection is None or recovery_projection.run != recovery_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await _materialize_snapshot(recovery, execution, recovery_run, recovery_snapshot, require_complete=True)
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
                await self.verify_terminal_attempts(
                    candidate_step_run_ids=(run_id,),
                    required_step_run_id=None,
                )
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
            self._business_unavailable = True
            if self._close_preflight_ok:
                return
            _logger.debug("step preflight close started")
            try:
                for run_id in sorted(self._staging._fact_run_ids()):
                    await self._verify_staging_run(run_id, required=False)
            except BaseException:
                _logger.warning("step preflight close failed", exc_info=True)
                raise
            self._close_preflight_ok = True
            _logger.debug("step preflight close completed")

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


async def _materialize_snapshot(
    source: StepStore,
    target: StepStore,
    run: RunRecord,
    snapshot: ContinuableSnapshot,
    *,
    require_complete: bool,
) -> None:
    _logger.debug(
        "materializing step snapshot: run=%s state=%s require_complete=%s",
        run.run_id,
        snapshot.state,
        require_complete,
    )
    await _materialize_run(source, target, run.run_id)
    await target.save_snapshot(snapshot)
    current = await target.latest_snapshot(run_id=run.run_id, include_interrupted=True)
    if current is None or current != snapshot:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if require_complete and current.state != "complete":
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


def _namespace_digest(namespace: str) -> str:
    return _digest(namespace)


def _scope_key(tenant_id: str) -> str:
    return _digest(tenant_id)


def _step_identity_digest(*parts: str | int) -> str:
    return canonical_identity_digest(
        "step-fact",
        {"parts": [str(part) for part in parts]},
    )


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
        grouped.setdefault(str(row["run_id"] ), []).append(row)
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
    return RunRecord(run_id=str(value["run_id"]), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), metadata=dict(value.get("metadata") or {}), started_at=_datetime(value["started_at"]))


def _event_json(event: StepEvent) -> dict[str, object]:
    return {"run_id": event.run_id, "kind": event.kind, "step_index": event.step_index, "timestamp": event.timestamp.astimezone(timezone.utc).isoformat(), "conversation_id": event.conversation_id, "parent_run_id": event.parent_run_id, "agent_name": event.agent_name, "tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "error": event.error, "metadata": dict(event.metadata)}


def _event_from_json(value: dict[str, object]) -> StepEvent:
    return StepEvent(run_id=str(value["run_id"]), kind=value["kind"], step_index=int(value["step_index"]), timestamp=_datetime(value["timestamp"]), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), tool_call_id=value.get("tool_call_id"), tool_name=value.get("tool_name"), error=value.get("error"), metadata=dict(value.get("metadata", {})))


def _event_from_sql(value: Mapping[str, object]) -> StepEvent:
    return StepEvent(run_id=str(value["run_id"]), kind=str(value["kind"]), step_index=int(value["step_index"]), timestamp=_datetime(value["timestamp"]), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), tool_call_id=value.get("tool_call_id"), tool_name=value.get("tool_name"), error=value.get("error"), metadata=dict(value.get("metadata") or {}))


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
    state = _snapshot_state(value.get("state", "complete"))
    messages = value.get("messages", [])
    await _materialize_media(messages, object_store)
    return ContinuableSnapshot(run_id=str(value["run_id"]), step_index=int(value["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(messages), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), timestamp=_datetime(value["timestamp"]), state=state)


async def _snapshot_from_sql(value: Mapping[str, object], object_store: ObjectStore) -> ContinuableSnapshot:
    object_store_id = value.get("media_store_id")
    if not isinstance(object_store_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if object_store_id != object_store.store_id:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    state = _snapshot_state(value.get("state", "complete"))
    messages = value.get("messages") or []
    await _materialize_media(messages, object_store)
    return ContinuableSnapshot(run_id=str(value["run_id"]), step_index=int(value["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(messages), conversation_id=value.get("conversation_id"), parent_run_id=value.get("parent_run_id"), agent_name=value.get("agent_name"), timestamp=_datetime(value["timestamp"]), state=state)


def _snapshot_state(value: object) -> str:
    if value not in ("complete", "interrupted"):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return str(value)


def _snapshot_row_matches(row: Mapping[str, object], payload: Mapping[str, object], snapshot: ContinuableSnapshot) -> bool:
    return (
        str(row["run_id"] ) == snapshot.run_id
        and int(row["step_index"]) == snapshot.step_index
        and row.get("messages") == payload.get("messages")
        and row.get("conversation_id") == snapshot.conversation_id
        and row.get("parent_run_id") == snapshot.parent_run_id
        and row.get("agent_name") == snapshot.agent_name
        and _datetime(row["timestamp"]) == snapshot.timestamp.astimezone(timezone.utc)
        and str(row["state"]) == snapshot.state
        and row.get("media_store_id") == payload.get("object_store_id")
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
    return ToolEffectRecord(tool_call_id=str(value["tool_call_id"]), tool_name=str(value["tool_name"]), run_id=str(value["run_id"]), status=str(value["status"]), started_at=_datetime(value["started_at"]), ended_at=None if value.get("ended_at") is None else _datetime(value["ended_at"]), idempotency_key=value.get("idempotency_key"), effect_summary=summary)


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
    "RuntimeStepStore",
    "SqlStepArchive",
    "StagingStepStore",
]
