#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PydanticAI StepStore adapter backed by Runtime StateStore facts."""

import asyncio
import base64
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Protocol, runtime_checkable

from linktools.core import environ
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
)

from ...errors import AIError, ErrorCode
from ._codec import decode_domain, decode_envelope, encode_domain, encode_envelope
from ._plan import RuntimeDomain, RuntimeRetentionMode
from ._store import (
    FactQuery,
    RecordQuery,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    parent_digest,
    partition_digest,
    record_key_digest,
    scope_digest,
    sequence_key,
    sortable_timestamp,
    stream_digest,
    subject_digest,
)

_logger = environ.get_logger("ai.runtime.state.steps")


@runtime_checkable
class _StepArchiveBatch(Protocol):
    async def sync_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
    ) -> None: ...

    async def materialize_snapshot(self, run: RunRecord, snapshot: ContinuableSnapshot) -> None: ...

    async def materialize_effect(self, run: RunRecord, effect: ToolEffectRecord) -> None: ...


@dataclass(slots=True)
class _ProjectionOffset:
    events: int = 0
    snapshots: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionProjectionBatch:
    run: RunRecord
    events: tuple[StepEvent, ...]
    snapshots: tuple[ContinuableSnapshot, ...]
    base_event_offset: int
    base_snapshot_offset: int
    target_event_offset: int
    target_snapshot_offset: int


class ExecutionProjectionCheckpoint:
    """Hold a captured projection batch until its durable commit is confirmed."""

    def __init__(self, store: "RuntimeStepStore", batch: ExecutionProjectionBatch) -> None:
        self._store = store
        self.batch = batch
        self._active = True

    async def acknowledge(self) -> None:
        if not self._active:
            raise RuntimeError("projection checkpoint is no longer active")
        await self._store._acknowledge_execution_projection(self.batch)

    def _deactivate(self) -> None:
        self._active = False


@dataclass
class _ProjectionLockEntry:
    lock: asyncio.Lock
    references: int = 0


class _PerRunProjectionLocks:
    def __init__(self) -> None:
        self._entries: dict[str, _ProjectionLockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, run_id: str):
        async with self._guard:
            entry = self._entries.get(run_id)
            if entry is None:
                entry = _ProjectionLockEntry(asyncio.Lock())
                self._entries[run_id] = entry
            entry.references += 1
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._guard:
                entry.references -= 1
                if entry.references == 0 and self._entries.get(run_id) is entry:
                    self._entries.pop(run_id, None)


class StagingStepStore(StepStore):
    """Process-local facts collected before owner materialization."""

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
        self._ensure_open()
        async with self._lock:
            previous = self._runs.get(record.run_id)
            if previous is not None and previous != record:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[record.run_id] = record

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        self._ensure_open()
        return self._runs.get(run_id)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        self._ensure_open()
        values = [
            value
            for value in self._runs.values()
            if (parent_run_id is None or value.parent_run_id == parent_run_id)
            and (conversation_id is None or value.conversation_id == conversation_id)
        ]
        return sorted(values, key=lambda value: (value.started_at, value.run_id))

    async def append_event(self, event: StepEvent) -> None:
        self._ensure_open()
        if event.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            values = self._events.setdefault(event.run_id, [])
            if event not in values:
                values.append(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        self._ensure_open()
        return list(self._events.get(run_id, ()))

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]:
        self._ensure_open()
        return list(self._snapshots.get(run_id, ()))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        self._ensure_open()
        if snapshot.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            values = self._snapshots.setdefault(snapshot.run_id, [])
            if snapshot not in values:
                values.append(snapshot)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        self._ensure_open()
        values = self._snapshots.get(run_id, ())
        if not values:
            return None
        latest = values[-1]
        return latest if include_interrupted or latest.state == "complete" else None

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        self._ensure_open()
        if record.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._lock:
            values = self._effects.setdefault(record.run_id, [])
            if record in values:
                return
            values.append(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        self._ensure_open()
        return next(
            (value for value in reversed(self._effects.get(run_id, ())) if value.tool_call_id == tool_call_id),
            None,
        )

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        self._ensure_open()
        latest: dict[str, ToolEffectRecord] = {}
        for value in self._effects.get(run_id, ()):
            latest[value.tool_call_id] = value
        return [value for value in latest.values() if value.status == "started"]

    async def release_run(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
        self._events.pop(run_id, None)
        self._snapshots.pop(run_id, None)
        self._effects.pop(run_id, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)


class InMemoryStepArchive(StagingStepStore):
    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        super().__init__()
        self._runtime_domain = runtime_domain

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    async def sync_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
    ) -> None:
        self._ensure_open()
        async with self._lock:
            previous = self._runs.get(run.run_id)
            if previous is not None and previous != run:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[run.run_id] = run
            event_values = self._events.setdefault(run.run_id, [])
            snapshot_values = self._snapshots.setdefault(run.run_id, [])
            for event in events:
                if event not in event_values:
                    event_values.append(event)
            for snapshot in snapshots:
                if snapshot not in snapshot_values:
                    snapshot_values.append(snapshot)

    async def materialize_snapshot(self, run: RunRecord, snapshot: ContinuableSnapshot) -> None:
        await self.sync_projection(run, events=(), snapshots=(snapshot,))

    async def materialize_effect(self, run: RunRecord, effect: ToolEffectRecord) -> None:
        self._ensure_open()
        async with self._lock:
            previous = self._runs.get(run.run_id)
            if previous is not None and previous != run:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[run.run_id] = run
            values = self._effects.setdefault(run.run_id, [])
            if effect not in values:
                values.append(effect)


class StateStepArchive(StepStore):
    """Durable Step owner archive using StateStore Record and Fact primitives."""

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str, runtime_domain: RuntimeDomain) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self._closed = False

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @property
    def state_store(self) -> StateStore:
        return self._store

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    def _run_key(self, run_id: str) -> bytes:
        return record_key_digest(self._namespace, self._tenant_id, self._runtime_domain.value, "step_run", run_id)

    def _stream(self, run_id: str, family: str) -> bytes:
        return stream_digest(self._namespace, self._tenant_id, self._runtime_domain.value, family, run_id)

    def _sequence(self, run_id: str, family: str) -> bytes:
        return sequence_key(self._namespace, self._tenant_id, self._runtime_domain.value, family, run_id)

    async def register_run(self, record: RunRecord) -> None:
        self._ensure_open()
        value = self._stored_run(record)

        async def mutate(transaction: StateTransaction) -> None:
            stored = await transaction.get_record(self._run_key(record.run_id))
            if stored is not None:
                if _decode_step(stored.data) != record:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return
            await transaction.insert_record(value)

        await self._store.mutate(mutate)

    def _stored_run(self, record: RunRecord) -> StoredRecord:
        return StoredRecord(
            self._run_key(record.run_id),
            partition_digest(self._namespace, self._tenant_id, self._runtime_domain.value, "step_run"),
            None
            if record.conversation_id is None
            else scope_digest(
                self._namespace,
                self._tenant_id,
                self._runtime_domain.value,
                "step_run",
                "conversation",
                record.conversation_id,
            ),
            None
            if record.parent_run_id is None
            else parent_digest(
                self._namespace,
                self._tenant_id,
                self._runtime_domain.value,
                "step_run",
                "parent",
                record.parent_run_id,
            ),
            "step_run",
            sortable_timestamp(record.started_at, record.run_id),
            None,
            0,
            None,
            0,
            None,
            _encode_step(record),
        )

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        stored = await self._store.read(lambda transaction: transaction.get_record(self._run_key(run_id)))
        return None if stored is None else _decode_step(stored.data)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        if parent_run_id is not None:
            query = RecordQuery(
                kind="step_run",
                parent_digest=parent_digest(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "step_run",
                    "parent",
                    parent_run_id,
                )
            )
        elif conversation_id is not None:
            query = RecordQuery(
                kind="step_run",
                scope_digest=scope_digest(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "step_run",
                    "conversation",
                    conversation_id,
                )
            )
        else:
            query = RecordQuery(
                kind="step_run",
                partition_digest=partition_digest(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "step_run",
                )
            )
        records = await self._store.read(lambda transaction: transaction.list_records(query))
        values = [_decode_step(record.data) for record in records]
        return [value for value in values if isinstance(value, RunRecord)]

    async def sync_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
    ) -> None:
        self._ensure_open()
        await self._store.mutate(
            lambda transaction: self._sync_projection_in_transaction(
                transaction,
                run,
                events=events,
                snapshots=snapshots,
            )
        )

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
    ) -> None:
        self._ensure_open()
        facts = tuple(
            ("event", event, _step_event_kind(event)) for event in events
        ) + tuple(("snapshot", snapshot, snapshot.state) for snapshot in snapshots)
        if not facts:
            owner = self._run_key(run.run_id)
            current = await transaction.get_record(owner)
            if current is None:
                await transaction.insert_record(self._stored_run(run))
            elif _decode_step(current.data) != run:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return
        owner = self._run_key(run.run_id)
        owner_record = await transaction.get_record(owner)
        if owner_record is None:
            await transaction.insert_record(self._stored_run(run))
            owner_record = await transaction.get_record(owner)
        elif _decode_step(owner_record.data) != run:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if owner_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if await transaction.guard_record(
            owner,
            expected_storage_version=owner_record.storage_version,
        ) is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        grouped: dict[str, list[object]] = {"event": [], "snapshot": []}
        kinds: dict[str, list[str]] = {"event": [], "snapshot": []}
        for family, value, kind in facts:
            grouped[family].append(value)
            kinds[family].append(kind)
        stored_facts: list[StoredFact] = []
        reservation_requests = {
            self._sequence(run.run_id, family): len(grouped[family])
            for family in ("event", "snapshot")
            if grouped[family]
        }
        high_waters = await transaction.reserve_sequences(reservation_requests)
        for family in ("event", "snapshot"):
            values = grouped[family]
            if not values:
                continue
            sequence_key_value = self._sequence(run.run_id, family)
            final = high_waters[sequence_key_value]
            sequences = tuple(range(final - len(values) + 1, final + 1))
            stream = self._stream(run.run_id, family)
            fact_kind = "step_event" if family == "event" else "step_snapshot"
            for sequence, value, kind in zip(sequences, values, kinds[family], strict=True):
                stored_facts.append(
                    StoredFact(
                        stream,
                        sequence,
                        owner,
                        fact_kind,
                        None,
                        kind,
                        _encode_step(value),
                    )
                )
        await _insert_facts(transaction, tuple(stored_facts))

    async def materialize_snapshot(self, run: RunRecord, snapshot: ContinuableSnapshot) -> None:
        await self._store.mutate(
            lambda transaction: self._materialize_snapshot_in_transaction(
                transaction,
                run,
                snapshot,
            )
        )

    async def _materialize_snapshot_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
    ) -> None:
        await self._materialize_fact_in_transaction(
            transaction,
            run,
            "snapshot",
            snapshot,
            snapshot.state,
        )

    async def materialize_effect(self, run: RunRecord, effect: ToolEffectRecord) -> None:
        await self._store.mutate(
            lambda transaction: self._materialize_effect_in_transaction(
                transaction,
                run,
                effect,
            )
        )

    async def _materialize_effect_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        effect: ToolEffectRecord,
    ) -> None:
        await self._materialize_fact_in_transaction(
            transaction,
            run,
            "effect",
            effect,
            effect.status,
        )

    async def _materialize_fact_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        family: str,
        value: object,
        kind: str,
    ) -> None:
        stream = self._stream(run.run_id, family)
        owner = self._run_key(run.run_id)
        subject = _step_subject(value)
        fact_kind = {
            "snapshot": "step_snapshot",
            "effect": "step_effect",
        }[family]
        data = _encode_step(value)

        owner_record = await transaction.get_record(owner)
        if owner_record is None:
            await transaction.insert_record(self._stored_run(run))
            owner_record = await transaction.get_record(owner)
        elif _decode_step(owner_record.data) != run:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if owner_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing = (
            await transaction.list_facts(FactQuery(stream, subject_digest=subject, latest=True))
            if subject is not None
            else await transaction.list_facts(FactQuery(stream, latest=True))
        )
        if any(fact.data == data and fact.state == kind for fact in existing):
            return
        if await transaction.guard_record(
            owner,
            expected_storage_version=owner_record.storage_version,
        ) is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        sequence = (await _reserve_sequences(transaction, self._sequence(run.run_id, family), 1))[0]
        await _insert_facts(
            transaction,
            (StoredFact(stream, sequence, owner, fact_kind, subject, kind, data),),
        )

    async def append_event(self, event: StepEvent) -> None:
        await self._append(event.run_id, "event", event, _step_event_kind(event))

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        values = await self._facts(run_id, "event")
        return [_decode_step(value.data) for value in values]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._append(snapshot.run_id, "snapshot", snapshot, snapshot.state)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        values = await self._facts(run_id, "snapshot", latest=True)
        if not values:
            return None
        latest = _decode_step(values[0].data)
        return latest if include_interrupted or latest.state == "complete" else None

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._append(record.run_id, "effect", record, record.status)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await self._store.read(
            lambda transaction: self._get_tool_effect_in_transaction(
                transaction,
                run_id=run_id,
                tool_call_id=tool_call_id,
            )
        )

    async def _get_tool_effect_in_transaction(
        self,
        transaction: StateTransaction,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> ToolEffectRecord | None:
        values = await transaction.list_facts(
            FactQuery(
                self._stream(run_id, "effect"),
                subject_digest=subject_digest(["tool_call", tool_call_id]),
                latest=True,
            )
        )
        return None if not values else _decode_step(values[0].data)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        values = [_decode_step(value.data) for value in await self._facts(run_id, "effect", latest_per_subject=True)]
        return [value for value in values if value.status == "started"]

    async def release_run(self, run_id: str) -> None:
        async def mutate(transaction: StateTransaction) -> None:
            await transaction.delete_record(self._run_key(run_id))
            await transaction.delete_sequences(
                tuple(self._sequence(run_id, family) for family in ("event", "snapshot", "effect"))
            )

        await self._store.mutate(mutate)

    async def _append(self, run_id: str, family: str, value: object, kind: str) -> None:
        stream = self._stream(run_id, family)
        owner = self._run_key(run_id)
        subject = _step_subject(value)
        fact_kind = {
            "event": "step_event",
            "snapshot": "step_snapshot",
            "effect": "step_effect",
        }[family]
        data = _encode_step(value)

        async def mutate(transaction: StateTransaction) -> None:
            owner_record = await transaction.get_record(owner)
            if owner_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if subject is not None:
                existing = await transaction.list_facts(
                    FactQuery(stream, subject_digest=subject, latest=True)
                )
                if any(fact.data == data and fact.state == kind for fact in existing):
                    return
            if await transaction.guard_record(
                owner,
                expected_storage_version=owner_record.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            sequence = await transaction.next_sequence(self._sequence(run_id, family))
            await transaction.insert_fact(StoredFact(stream, sequence, owner, fact_kind, subject, kind, data))

        await self._store.mutate(mutate)

    async def _facts(
        self,
        run_id: str,
        family: str,
        *,
        subject: bytes | None = None,
        latest: bool = False,
        latest_per_subject: bool = False,
    ) -> tuple[StoredFact, ...]:
        if latest and latest_per_subject:
            raise ValueError("latest and latest_per_subject cannot both be set")
        return await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    self._stream(run_id, family),
                    subject_digest=subject,
                    latest=latest,
                    latest_per_subject=latest_per_subject,
                )
            )
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)


class RuntimeStepStore(StepStore):
    """Route staging facts to their owning durable StateStore archive."""

    def __init__(
        self,
        staging: StagingStepStore,
        *,
        conversation_archive: StepStore,
        execution_archive: StepStore | None,
        recovery_archive: StepStore | None,
        conversation_retention: RuntimeRetentionMode,
        execution_retention: RuntimeRetentionMode,
        recovery_retention: RuntimeRetentionMode,
    ) -> None:
        del conversation_retention, execution_retention, recovery_retention
        self._staging = staging
        self._archives = {
            RuntimeDomain.CONVERSATION: conversation_archive,
            **({RuntimeDomain.EXECUTION: execution_archive} if execution_archive is not None else {}),
            **({RuntimeDomain.RECOVERY: recovery_archive} if recovery_archive is not None else {}),
        }
        self._initialized = False
        self._preflight = False
        self._projection_offsets: dict[str, _ProjectionOffset] = {}
        self._projection_dirty: set[str] = set()
        self._projection_locks = _PerRunProjectionLocks()

    async def initialize(self) -> None:
        await self._staging.initialize()
        for archive in self._archives.values():
            await archive.initialize()
        self._projection_offsets.clear()
        self._projection_dirty.clear()
        self._initialized = True

    async def register_run(self, record: RunRecord) -> None:
        await self._ensure_business()
        await self._staging.register_run(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_business()
        return await self._staging.get_run(run_id=run_id)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        await self._ensure_business()
        return await self._staging.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def append_event(self, event: StepEvent) -> None:
        await self._ensure_business()
        async with self._projection_locks.hold(event.run_id):
            await self._staging.append_event(event)
            self._projection_dirty.add(event.run_id)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_business()
        return await self._staging.list_events(run_id=run_id)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._ensure_business()
        async with self._projection_locks.hold(snapshot.run_id):
            await self._staging.save_snapshot(snapshot)
            self._projection_dirty.add(snapshot.run_id)
            recovery = self._archives.get(RuntimeDomain.RECOVERY)
            if recovery is not None:
                run = await self._staging.get_run(run_id=snapshot.run_id)
                if run is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                await _materialize_snapshot(recovery, run, snapshot)
            if snapshot.state == "complete":
                await self._flush_execution_projection_locked(snapshot.run_id)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_business()
        return await self._staging.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._ensure_business()
        async with self._projection_locks.hold(record.run_id):
            await self._staging.record_tool_effect(record)
            archive = self._archives.get(RuntimeDomain.RECOVERY)
            if archive is not None:
                run = await self._staging.get_run(run_id=record.run_id)
                if run is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                current = await archive.get_tool_effect(
                    run_id=record.run_id,
                    tool_call_id=record.tool_call_id,
                )
                if current is None or current.status != record.status:
                    await _materialize_effect(archive, run, record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        await self._ensure_business()
        return await self._staging.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        await self._ensure_business()
        return await self._staging.list_unresolved_tool_effects(run_id=run_id)

    def read_store(self, runtime_domain: RuntimeDomain) -> StepStore:
        if runtime_domain not in self._archives:
            return self._staging
        return self._archives[runtime_domain]

    async def materialize_recovery_snapshot(self, *, step_run_id: str, require_complete: bool) -> None:
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id, include_interrupted=True)
        run = await self._staging.get_run(run_id=step_run_id)
        archive = self._archives.get(RuntimeDomain.RECOVERY)
        if snapshot is None or run is None:
            if require_complete:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if require_complete and snapshot.state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if archive is not None:
            await _materialize_snapshot(archive, run, snapshot)

    async def materialize_conversation(self, *, step_run_id: str) -> None:
        run = await self._staging.get_run(run_id=step_run_id)
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id)
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if run is None or snapshot is None or archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await _materialize_snapshot(archive, run, snapshot)

    async def materialize_from_recovery(self, *, target: RuntimeDomain, step_run_id: str) -> None:
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        destination = self._archives.get(target)
        if recovery is None or destination is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        run = await recovery.get_run(run_id=step_run_id)
        snapshot = await recovery.latest_snapshot(run_id=step_run_id)
        if run is None or snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await _materialize_snapshot(destination, run, snapshot)

    async def flush_execution_projection(self, step_run_id: str) -> None:
        async with self.execution_projection_checkpoint(step_run_id) as checkpoint:
            await self._flush_execution_projection_locked(step_run_id, checkpoint.batch)

    async def _flush_execution_projection_locked(
        self,
        step_run_id: str,
        projection: ExecutionProjectionBatch | None = None,
    ) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if archive is None or step_run_id not in self._projection_dirty:
            return
        if projection is None:
            projection = await self._capture_execution_projection_locked(step_run_id)
        if not projection.events and not projection.snapshots:
            return
        started = monotonic()
        await _sync_projection(
            archive,
            projection.run,
            projection.events,
            projection.snapshots,
        )
        await self._acknowledge_execution_projection(projection)
        _logger.debug(
            "step projection flushed: domain=%s backend=%s run=%s "
            "events=%s snapshots=%s duration_ms=%.3f",
            RuntimeDomain.EXECUTION.value,
            type(archive).__name__,
            step_run_id,
            len(projection.events),
            len(projection.snapshots),
            (monotonic() - started) * 1000,
        )

    @asynccontextmanager
    async def execution_projection_checkpoint(
        self,
        step_run_id: str,
    ) -> AsyncIterator[ExecutionProjectionCheckpoint]:
        await self._ensure_business()
        async with self._projection_locks.hold(step_run_id):
            projection = await self._capture_execution_projection_locked(step_run_id)
            _logger.debug(
                "step projection checkpoint captured: run=%s "
                "base_events=%s target_events=%s "
                "base_snapshots=%s target_snapshots=%s",
                step_run_id,
                projection.base_event_offset,
                projection.target_event_offset,
                projection.base_snapshot_offset,
                projection.target_snapshot_offset,
            )
            checkpoint = ExecutionProjectionCheckpoint(self, projection)
            try:
                yield checkpoint
            finally:
                checkpoint._deactivate()

    async def _capture_execution_projection_locked(
        self,
        step_run_id: str,
    ) -> ExecutionProjectionBatch:
        run = await self._staging.get_run(run_id=step_run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = await self._staging.list_events(run_id=step_run_id)
        snapshots = await self._staging.list_snapshots(run_id=step_run_id)
        offset = self._projection_offsets.setdefault(step_run_id, _ProjectionOffset())
        return ExecutionProjectionBatch(
            run,
            tuple(events[offset.events:]),
            tuple(snapshots[offset.snapshots:]),
            offset.events,
            offset.snapshots,
            len(events),
            len(snapshots),
        )

    async def _acknowledge_execution_projection(self, projection: ExecutionProjectionBatch) -> None:
        offset = self._projection_offsets.setdefault(projection.run.run_id, _ProjectionOffset())
        offset.events = max(offset.events, projection.target_event_offset)
        offset.snapshots = max(offset.snapshots, projection.target_snapshot_offset)
        self._projection_dirty.discard(projection.run.run_id)

    async def verify_terminal_attempts(
        self, *, candidate_step_run_ids: tuple[str, ...], required_step_run_id: str | None
    ) -> None:
        for run_id in dict.fromkeys(candidate_step_run_ids):
            if required_step_run_id != run_id:
                continue
            snapshot = await self._staging.latest_snapshot(run_id=run_id, include_interrupted=True)
            if snapshot is None:
                for archive in self._archives.values():
                    snapshot = await archive.latest_snapshot(
                        run_id=run_id,
                        include_interrupted=True,
                    )
                    if snapshot is not None:
                        break
            if snapshot is None or snapshot.state != "complete":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def release_staging_many(self, *, candidate_step_run_ids: tuple[str, ...]) -> None:
        for run_id in dict.fromkeys(candidate_step_run_ids):
            async with self._projection_locks.hold(run_id):
                await self._staging.release_run(run_id)
                self._projection_offsets.pop(run_id, None)
                self._projection_dirty.discard(run_id)

    async def release_archive(self, runtime_domain: RuntimeDomain, step_run_id: str) -> None:
        archive = self._archives.get(runtime_domain)
        if archive is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        async with self._projection_locks.hold(step_run_id):
            await archive.release_run(step_run_id)
            self._projection_offsets.pop(step_run_id, None)
            self._projection_dirty.discard(step_run_id)

    async def preflight_close(self) -> None:
        self._preflight = True

    async def close(self) -> None:
        if not self._preflight:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._staging.close()
        for archive in self._archives.values():
            await archive.close()
        self._initialized = False

    async def _ensure_business(self) -> None:
        if not self._initialized:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


async def _sync_projection(
    target: StepStore,
    run: RunRecord,
    events: tuple[StepEvent, ...],
    snapshots: tuple[ContinuableSnapshot, ...],
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.sync_projection(run, events=events, snapshots=snapshots)
        return
    if await target.get_run(run_id=run.run_id) is None:
        await target.register_run(run)
    for event in events:
        await target.append_event(event)
    for snapshot in snapshots:
        await target.save_snapshot(snapshot)


async def _materialize_effect(target: StepStore, run: RunRecord, effect: ToolEffectRecord) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.materialize_effect(run, effect)
        return
    if await target.get_run(run_id=run.run_id) is None:
        await target.register_run(run)
    await target.record_tool_effect(effect)


async def _reserve_sequences(
    transaction: StateTransaction,
    key: bytes,
    count: int,
) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("sequence reservation count must be positive")
    final = await transaction.reserve_sequence(key, count)
    return tuple(range(final - count + 1, final + 1))


async def _insert_facts(transaction: StateTransaction, facts: tuple[StoredFact, ...]) -> None:
    await transaction.insert_facts(facts)


async def _materialize_snapshot(target: StepStore, run: RunRecord, snapshot: ContinuableSnapshot) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.materialize_snapshot(run, snapshot)
        return
    existing_run = await target.get_run(run_id=run.run_id)
    existing_snapshot = await target.latest_snapshot(
        run_id=run.run_id,
        include_interrupted=True,
    )
    if existing_run == run and existing_snapshot == snapshot:
        return
    await target.register_run(run)
    await target.save_snapshot(snapshot)


def _encode_step(value: object) -> dict[str, object]:
    if isinstance(value, ContinuableSnapshot):
        return encode_envelope(
            {
                "type": "ContinuableSnapshot",
                "run_id": value.run_id,
                "step_index": value.step_index,
                "messages": base64.b64encode(ModelMessagesTypeAdapter.dump_json(value.messages)).decode("ascii"),
                "conversation_id": value.conversation_id,
                "parent_run_id": value.parent_run_id,
                "agent_name": value.agent_name,
                "timestamp": value.timestamp.isoformat(),
                "state": value.state,
            }
        )
    return encode_envelope({"type": value.__class__.__name__, "payload": encode_domain(value)})


def _step_subject(value: object) -> bytes | None:
    if isinstance(value, ToolEffectRecord):
        return subject_digest(["tool_call", value.tool_call_id])
    return None


def _step_event_kind(value: StepEvent) -> str:
    return str(value.kind)


def _decode_step(value: Mapping[str, object]) -> object:
    payload = decode_envelope(value)
    kind = payload.get("type")
    if kind == "ContinuableSnapshot":
        messages = ModelMessagesTypeAdapter.validate_json(base64.b64decode(str(payload["messages"])))
        return ContinuableSnapshot(
            run_id=str(payload["run_id"]),
            step_index=int(payload["step_index"]),
            messages=messages,
            conversation_id=payload.get("conversation_id"),
            parent_run_id=payload.get("parent_run_id"),
            agent_name=payload.get("agent_name"),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            state=str(payload["state"]),
        )
    target = {"RunRecord": RunRecord, "StepEvent": StepEvent, "ToolEffectRecord": ToolEffectRecord}.get(str(kind))
    if target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return decode_domain(payload.get("payload"), target)


__all__ = [
    "ExecutionProjectionBatch",
    "ExecutionProjectionCheckpoint",
    "InMemoryStepArchive",
    "RuntimeStepStore",
    "StagingStepStore",
    "StateStepArchive",
]
