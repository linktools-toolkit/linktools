#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PydanticAI StepStore adapter backed by Runtime StateStore facts."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from enum import Enum
from time import monotonic
from typing import Protocol, runtime_checkable
from uuid import uuid4

from linktools.core import environ
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
    ToolEffectRecord,
)

from ...errors import AIError, ErrorCode
from ...storage import ObjectStore
from ._codec import (
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_step_envelope,
)
from ._contracts import (
    ContextProjection,
    ConversationHistoryRepository,
    ExecutionHistoryHeadRecord,
    ExecutionHistoryState,
    ExecutionRepository,
    ExecutionRunSealHead,
    HistoryQuality,
    LoadedContextMessage,
    LoadedModelContext,
    StoredStepSnapshot,
    TranscriptChunk,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptSpanRef,
)
from ._durability import (
    CommitObservation,
    DurableCommitState,
    run_durable_commit,
)
from ._history import (
    TranscriptCapture,
    TranscriptRepository,
    _conversation_overlap_signature,
    _exact_message_signature,
    _overlap_signature,
    suffix_prefix_overlap,
)
from ._plan import RuntimeDomain, RuntimeRetentionMode
from ._store import (
    FactQuery,
    RecordQuery,
    StateLockOrderError,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    active_state_scope,
    enter_run_history_lock,
    exit_run_history_lock,
    parent_digest,
    partition_digest,
    record_key_digest,
    require_no_run_history_lock,
    scope_digest,
    sequence_key,
    sortable_timestamp,
    stream_digest,
    subject_digest,
)
from ._views import (
    count_execution_transcript_items,
    count_session_history_items,
    project_execution_transcript_message,
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
        execution_id: str | None = None,
    ) -> None: ...

    async def materialize_snapshot(
        self,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None: ...

    async def materialize_effect(
        self,
        run: RunRecord,
        effect: ToolEffectRecord,
        *,
        execution_id: str | None = None,
    ) -> None: ...


@dataclass(slots=True)
class _ProjectionOffset:
    events: int = 0
    snapshots: int = 0
    transcript_messages: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionProjectionBatch:
    run: RunRecord
    events: tuple[StepEvent, ...]
    snapshots: tuple[ContinuableSnapshot, ...]
    base_event_offset: int
    base_snapshot_offset: int
    target_event_offset: int
    target_snapshot_offset: int
    base_message_index: int
    target_message_index: int


@dataclass(frozen=True, slots=True)
class PreparedStepSnapshot:
    owner_id: str
    stored: StoredStepSnapshot
    chunks: tuple[TranscriptChunk, ...]
    projection: ContextProjection
    history_quality: HistoryQuality = HistoryQuality.COMPLETE
    chunk_session_history_item_counts: tuple[int, ...] = ()
    chunk_execution_transcript_item_counts: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedStepSnapshotBatch:
    run_id: str
    snapshots: tuple[PreparedStepSnapshot, ...]
    target_event_offset: int
    target_snapshot_offset: int
    target_transcript_message_count: int

    def __iter__(self):
        return iter(self.snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> PreparedStepSnapshot:
        return self.snapshots[index]


@dataclass(frozen=True, slots=True)
class PreparedExecutionProjection:
    run: RunRecord
    events: tuple[StepEvent, ...]
    snapshots: tuple[PreparedStepSnapshot, ...]
    base_event_offset: int
    base_snapshot_offset: int
    target_event_offset: int
    target_snapshot_offset: int
    target_transcript_message_count: int
    durable_projection_digest: str = "empty"

    @property
    def projection_digest(self) -> str:
        if not self.snapshots:
            return self.durable_projection_digest
        return self.snapshots[-1].projection.digest


@dataclass(frozen=True, slots=True)
class ExecutionTerminalSealPlan:
    execution_id: str
    binding_digest: str
    projections: tuple[PreparedExecutionProjection, ...]
    seal_tokens: tuple[tuple[str, str], ...]
    terminal_attempt_token: str = ""

    def token_for(self, run_id: str) -> str:
        for candidate, token in self.seal_tokens:
            if candidate == run_id:
                return token
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _chunk_message_groups(
    messages: Sequence[ModelMessage],
    chunks: Sequence[TranscriptChunk],
    base_index: int,
) -> "list[Sequence[ModelMessage]]":
    """Group source messages back onto prepared chunks for item counting."""
    groups: list[Sequence[ModelMessage]] = []
    cursor = 0
    for chunk in chunks:
        start = chunk.first_message_index - base_index
        groups.append(messages[start : start + chunk.message_count])
        cursor = start + chunk.message_count
    del cursor
    return groups


class _RunDurabilityKind(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    PROJECTION = "projection"
    SNAPSHOT = "snapshot"
    TOOL_EFFECT = "tool_effect"
    RECOVERY_MATERIALIZATION = "recovery_materialization"
    RELEASE = "release"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class _RunDurabilityFlight:
    """Registered in-flight durability work for one run.

    ``completion`` resolves only after the durable outcome is final; waiters
    must exit the run lock before awaiting it.
    """

    run_id: str
    token: str
    kind: _RunDurabilityKind
    completion: "asyncio.Future[None]"


_RunProjectionFlight = _RunDurabilityFlight


@dataclass(frozen=True, slots=True)
class CapturedExecutionProjection:
    """Immutable capture of one run's staged projection state."""

    run: RunRecord
    events: tuple[StepEvent, ...]
    snapshots: tuple[ContinuableSnapshot, ...]
    base_event_offset: int
    base_snapshot_offset: int
    target_event_offset: int
    target_snapshot_offset: int


@dataclass(frozen=True, slots=True)
class _LocalExecutionTerminalSeal:
    """Local freeze ownership of one run's staging for a terminal attempt."""

    execution_id: str
    token: str


@dataclass
class _ProjectionLockEntry:
    lock: asyncio.Lock
    references: int = 0
    owner: asyncio.Task[object] | None = None
    depth: int = 0


@dataclass(frozen=True, slots=True)
class _HeldRunHistoryLock:
    run_id: str
    task: asyncio.Task[object]


class LockOrderError(RuntimeError):
    """Raised when one task attempts to hold more than one run lock."""


_held_run_history_locks: ContextVar[tuple[_HeldRunHistoryLock, ...]] = ContextVar(
    "linktools_ai_held_run_history_locks",
    default=(),
)


class _RunHistoryLock:
    def __init__(self) -> None:
        self._entries: dict[str, _ProjectionLockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, run_id: str):
        current_task = asyncio.current_task()
        if current_task is None:
            raise LockOrderError("run history lock requires an asyncio task")
        if active_state_scope() is not None:
            raise StateLockOrderError(
                "StateStore callback cannot acquire a RunHistoryLock"
            )
        held = _held_run_history_locks.get()
        if any(value.task is not current_task for value in held):
            raise LockOrderError("a child task cannot inherit a RunHistoryLock")
        if any(value.run_id != run_id for value in held):
            raise LockOrderError(
                "one asyncio task cannot hold multiple RunHistoryLocks"
            )
        async with self._guard:
            entry = self._entries.get(run_id)
            if entry is None:
                entry = _ProjectionLockEntry(asyncio.Lock())
                self._entries[run_id] = entry
            entry.references += 1
            if held and entry.owner is not current_task:
                entry.references -= 1
                if entry.references == 0 and self._entries.get(run_id) is entry:
                    self._entries.pop(run_id, None)
                raise LockOrderError("run history lock ownership is inconsistent")
            if entry.owner is current_task:
                entry.depth += 1
                nested = True
            else:
                nested = False
        if nested:
            lock_token = enter_run_history_lock(run_id)
            try:
                yield
            finally:
                exit_run_history_lock(lock_token)
                async with self._guard:
                    entry.depth -= 1
                    entry.references -= 1
                    if entry.references == 0 and self._entries.get(run_id) is entry:
                        self._entries.pop(run_id, None)
            return
        acquired = False
        token: Token[tuple[_HeldRunHistoryLock, ...]] | None = None
        lock_token: Token[tuple[str, ...]] | None = None
        try:
            await entry.lock.acquire()
            acquired = True
            async with self._guard:
                entry.owner = current_task
                entry.depth = 1
            token = _held_run_history_locks.set(
                held + (_HeldRunHistoryLock(run_id, current_task),)
            )
            lock_token = enter_run_history_lock(run_id)
            yield
        finally:
            if lock_token is not None:
                exit_run_history_lock(lock_token)
            if token is not None:
                _held_run_history_locks.reset(token)
            if acquired:
                entry.lock.release()
            async with self._guard:
                entry.owner = None
                entry.depth = 0
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

    async def register_run(
        self,
        record: RunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            self.register_run_local(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        self._ensure_open()
        return self.get_run_local(run_id)

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

    async def append_event(
        self,
        event: StepEvent,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            self.append_event_local(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        self._ensure_open()
        return self.list_events_local(run_id)

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]:
        self._ensure_open()
        return self.list_snapshots_local(run_id)

    async def save_snapshot(
        self,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            self.save_snapshot_local(snapshot)

    async def latest_snapshot(
        self,
        *,
        run_id: str,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        self._ensure_open()
        return self.latest_snapshot_local(
            run_id,
            include_interrupted=include_interrupted,
        )

    async def record_tool_effect(
        self,
        record: ToolEffectRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            self.record_tool_effect_local(record)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        self._ensure_open()
        return self.get_tool_effect_local(run_id, tool_call_id)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        self._ensure_open()
        return self.list_unresolved_tool_effects_local(run_id)

    async def release_run(
        self,
        run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self.release_run_local(run_id)

    def register_run_local(self, record: RunRecord) -> None:
        self._ensure_open()
        previous = self._runs.get(record.run_id)
        if previous is not None and previous != record:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._runs[record.run_id] = record

    def get_run_local(self, run_id: str) -> RunRecord | None:
        self._ensure_open()
        return self._runs.get(run_id)

    def append_event_local(self, event: StepEvent) -> None:
        self._ensure_open()
        if event.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        values = self._events.setdefault(event.run_id, [])
        if event not in values:
            values.append(event)

    def list_events_local(self, run_id: str) -> list[StepEvent]:
        self._ensure_open()
        return list(self._events.get(run_id, ()))

    def list_snapshots_local(self, run_id: str) -> list[ContinuableSnapshot]:
        self._ensure_open()
        return list(self._snapshots.get(run_id, ()))

    def save_snapshot_local(self, snapshot: ContinuableSnapshot) -> None:
        self._ensure_open()
        if snapshot.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        values = self._snapshots.setdefault(snapshot.run_id, [])
        if snapshot not in values:
            values.append(snapshot)

    def latest_snapshot_local(
        self,
        run_id: str,
        *,
        include_interrupted: bool = False,
    ) -> ContinuableSnapshot | None:
        self._ensure_open()
        values = self._snapshots.get(run_id, ())
        if not values:
            return None
        latest = values[-1]
        return latest if include_interrupted or latest.state == "complete" else None

    def record_tool_effect_local(self, record: ToolEffectRecord) -> None:
        self._ensure_open()
        if record.run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        values = self._effects.setdefault(record.run_id, [])
        if record not in values:
            values.append(record)

    def get_tool_effect_local(
        self,
        run_id: str,
        tool_call_id: str,
    ) -> ToolEffectRecord | None:
        self._ensure_open()
        return next(
            (
                value
                for value in reversed(self._effects.get(run_id, ()))
                if value.tool_call_id == tool_call_id
            ),
            None,
        )

    def list_unresolved_tool_effects_local(
        self,
        run_id: str,
    ) -> list[ToolEffectRecord]:
        self._ensure_open()
        latest: dict[str, ToolEffectRecord] = {}
        for value in self._effects.get(run_id, ()):
            latest[value.tool_call_id] = value
        return [value for value in latest.values() if value.status == "started"]

    def release_run_local(self, run_id: str) -> None:
        self._ensure_open()
        self._runs.pop(run_id, None)
        self._events.pop(run_id, None)
        self._snapshots.pop(run_id, None)
        self._effects.pop(run_id, None)

    def capture_projection_local(
        self,
        run_id: str,
        offset: _ProjectionOffset,
    ) -> ExecutionProjectionBatch | None:
        self._ensure_open()
        run = self._runs.get(run_id)
        if run is None:
            return None
        events = self._events.get(run_id, ())
        snapshots = self._snapshots.get(run_id, ())
        return ExecutionProjectionBatch(
            run,
            tuple(events[offset.events:]),
            tuple(snapshots[offset.snapshots:]),
            offset.events,
            offset.snapshots,
            len(events),
            len(snapshots),
            offset.transcript_messages,
            offset.transcript_messages,
        )

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
        execution_id: str | None = None,
    ) -> None:
        del execution_id
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

    async def materialize_snapshot(
        self,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self.sync_projection(
            run,
            events=(),
            snapshots=(snapshot,),
            execution_id=execution_id,
        )

    async def materialize_effect(
        self,
        run: RunRecord,
        effect: ToolEffectRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            previous = self._runs.get(run.run_id)
            if previous is not None and previous != run:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[run.run_id] = run
            values = self._effects.setdefault(run.run_id, [])
            if effect not in values:
                values.append(effect)

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
        if not refs:
            return ()
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def execution_transcript_item_count(self, run_id: str) -> int:
        snapshot = self.latest_snapshot_local(run_id, include_interrupted=True)
        if snapshot is None:
            return 0
        return count_execution_transcript_items(snapshot.messages)

    async def iter_execution_transcript_item_range(
        self,
        run_id: str,
        *,
        start: int,
        end: int,
    ) -> AsyncIterator[str]:
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        snapshot = self.latest_snapshot_local(run_id, include_interrupted=True)
        messages = () if snapshot is None else snapshot.messages
        values = tuple(
            value
            for message in messages
            for value in project_execution_transcript_message(message)
        )
        if end > len(values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for value in values[start:end]:
            yield value

    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]:
        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)
        if snapshot is not None:
            for message in snapshot.messages:
                yield message

    async def load_model_context(self, *, run_id: str) -> tuple[object, ...]:
        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)
        return () if snapshot is None else tuple(snapshot.messages)


class StateStepArchive(StepStore):
    """Durable Step owner archive using StateStore Record and Fact primitives."""

    def __init__(
        self,
        store: StateStore,
        *,
        object_store: "ObjectStore | None",
        namespace: str,
        tenant_id: str,
        runtime_domain: RuntimeDomain,
        context_sources: Mapping[RuntimeDomain, TranscriptRepository] | None = None,
        history_repository: "ConversationHistoryRepository | None" = None,
        execution_repository: "ExecutionRepository | None" = None,
    ) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self._history = TranscriptRepository(
            store,
            object_store=object_store,
            namespace=namespace,
            tenant_id=tenant_id,
            runtime_domain=runtime_domain,
            context_sources=context_sources,
            history_repository=history_repository,
        )
        self._execution_repository = execution_repository
        self._context_baselines: dict[str, LoadedModelContext] = {}
        self._history_lock = _RunHistoryLock()
        self._closed = False

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @property
    def state_store(self) -> StateStore:
        return self._store

    @property
    def transcript_repository(self) -> TranscriptRepository:
        return self._history

    async def validate_integrity(self) -> None:
        await self._history.validate_integrity()

    async def execution_history_head(
        self,
        run_id: str,
    ) -> tuple[int, int, int, str]:
        values = await self.execution_history_heads((run_id,))
        head = values.get(run_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return (
            head.event_count,
            head.snapshot_count,
            head.transcript_message_count,
            head.projection_digest,
        )

    async def execution_history_heads(
        self,
        run_ids: Sequence[str],
    ) -> Mapping[str, ExecutionRunSealHead]:
        require_no_run_history_lock("StateStepArchive.execution_history_heads")
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        unique_run_ids = tuple(dict.fromkeys(run_ids))
        if not unique_run_ids:
            return {}
        sequence_keys = tuple(
            key
            for run_id in unique_run_ids
            for key in (
                self._sequence(run_id, "event"),
                self._sequence(run_id, "snapshot"),
            )
        )
        run_keys = tuple(self._run_key(run_id) for run_id in unique_run_ids)
        projection_keys = tuple(
            self._history.projection_key(run_id) for run_id in unique_run_ids
        )
        head_keys = tuple(self._history.head_key(run_id) for run_id in unique_run_ids)
        async def read(
            transaction: StateTransaction,
        ) -> tuple[Mapping[bytes, StoredRecord], Mapping[bytes, int]]:
            records = await transaction.get_records(
                (*run_keys, *projection_keys, *head_keys)
            )
            sequences = await transaction.get_sequences(sequence_keys)
            return records, sequences

        records, sequences = await self._store.read(read)
        result: dict[str, ExecutionRunSealHead] = {}
        for run_id in unique_run_ids:
            run_record = records.get(self._run_key(run_id))
            head_record = records.get(self._history.head_key(run_id))
            projection_record = records.get(self._history.projection_key(run_id))
            event_count = sequences.get(self._sequence(run_id, "event"), 0)
            snapshot_count = sequences.get(self._sequence(run_id, "snapshot"), 0)
            if head_record is None:
                if run_record is not None or projection_record is not None or event_count or snapshot_count:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result[run_id] = ExecutionRunSealHead(run_id, 0, 0, 0, "empty")
                continue
            if run_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = self._history.decode_head(head_record)
            projection_digest = "empty"
            if projection_record is not None:
                projection = _decode_enveloped_domain(
                    projection_record.data,
                    ContextProjection,
                )
                projection_digest = projection.digest
            result[run_id] = ExecutionRunSealHead(
                run_id,
                event_count,
                snapshot_count,
                head.message_count,
                projection_digest,
            )
        return result

    async def verify_execution_projection_head(
        self,
        projection: PreparedExecutionProjection,
    ) -> bool:
        require_no_run_history_lock(
            "StateStepArchive.verify_execution_projection_head"
        )
        if await self.get_run(run_id=projection.run.run_id) != projection.run:
            return False
        head = await self.execution_history_head(projection.run.run_id)
        return head == (
            projection.target_event_offset,
            projection.target_snapshot_offset,
            projection.target_transcript_message_count,
            projection.projection_digest,
        )

    def bind_history_lock(self, history_lock: _RunHistoryLock) -> None:
        self._history_lock = history_lock

    def register_context_baseline(
        self,
        step_run_id: str,
        context: LoadedModelContext,
    ) -> None:
        self._context_baselines[step_run_id] = context

    async def prepare_snapshots(
        self,
        run: RunRecord,
        snapshots: Sequence[ContinuableSnapshot],
    ) -> PreparedStepSnapshotBatch:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.prepare_snapshots")
        return await self._prepare_snapshots(
            run,
            snapshots,
        )

    async def prepare_snapshots_after_seal(
        self,
        run: RunRecord,
        snapshots: Sequence[ContinuableSnapshot],
    ) -> PreparedStepSnapshotBatch:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.prepare_snapshots_after_seal")
        if active_state_scope() is not None or _held_run_history_locks.get():
            raise LockOrderError(
                "sealed snapshot preparation requires no StateStore or run lock"
            )
        return await self._prepare_snapshots(
            run,
            snapshots,
        )

    async def initialize(self) -> None:
        self._closed = False
        self._context_baselines.clear()

    async def close(self) -> None:
        self._closed = True
        self._context_baselines.clear()

    def _run_key(self, run_id: str) -> bytes:
        return record_key_digest(self._namespace, self._tenant_id, self._runtime_domain.value, "step_run", run_id)

    def _stream(self, run_id: str, family: str) -> bytes:
        return stream_digest(self._namespace, self._tenant_id, self._runtime_domain.value, family, run_id)

    def _sequence(self, run_id: str, family: str) -> bytes:
        return sequence_key(self._namespace, self._tenant_id, self._runtime_domain.value, family, run_id)

    async def register_run(
        self,
        record: RunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.register_run")

        async def mutate(transaction: StateTransaction) -> None:
            history_head_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
            _owner_record, created = await self._ensure_run_with_head_in_transaction(
                transaction,
                record,
            )
            if created:
                await self._advance_execution_history_head_in_transaction(
                    transaction,
                    history_head_guard,
                )

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
        require_no_run_history_lock("StateStepArchive.get_run")
        stored = await self._store.read(lambda transaction: transaction.get_record(self._run_key(run_id)))
        return None if stored is None else _decode_step(stored.data)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        require_no_run_history_lock("StateStepArchive.list_runs")
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

    async def _prepare_snapshots(
        self,
        run: RunRecord,
        snapshots: Sequence[ContinuableSnapshot],
    ) -> PreparedStepSnapshotBatch:
        prepared: list[PreparedStepSnapshot] = []
        owner_id = run.run_id
        if self._runtime_domain is RuntimeDomain.CONVERSATION:
            owner_id = self._history_id(run)
        head = await self._history.get_head(owner_id)
        if head is None:
            if await self.get_run(run_id=run.run_id) is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = self._history.empty_head(owner_id)
        context_bound = max(
            (len(snapshot.messages) for snapshot in snapshots),
            default=0,
        )
        suffix_start = max(0, head.message_count - context_bound)
        suffix_messages = (
            ()
            if suffix_start == head.message_count
            else await self._history.load_message_span(
                owner_id,
                suffix_start,
                head.message_count,
            )
        )
        working_messages = list(suffix_messages)
        working_start = suffix_start
        target_message_count = head.message_count
        target_quality = head.quality
        for snapshot in snapshots:
            incoming = tuple(snapshot.messages)
            signature = (
                _conversation_overlap_signature
                if self._runtime_domain is RuntimeDomain.CONVERSATION
                else _overlap_signature
            )
            incoming_signatures = tuple(signature(message) for message in incoming)
            stored_signatures = tuple(
                signature(message) for message in working_messages
            )
            overlap = suffix_prefix_overlap(stored_signatures, incoming_signatures)
            delta = list(incoming[overlap:])
            if overlap == 0 and stored_signatures and incoming_signatures:
                target_quality = HistoryQuality.CONSERVATIVE
            base_message_count = target_message_count
            capture = TranscriptCapture(
                base_message_count,
                tuple(delta),
                tuple(
                    TranscriptOrigin.RAW
                    if message.run_id == run.run_id
                    else TranscriptOrigin.UNKNOWN
                    for message in delta
                ),
                target_quality,
            )
            (
                chunks,
                chunk_session_history_item_counts,
                chunk_execution_transcript_item_counts,
            ) = (
                await self._prepare_captured_chunks(
                    owner_id,
                    capture,
                    message_index_offset=0,
                )
            )
            sources = self._message_sources(
                owner_id,
                incoming,
                tuple(working_messages) + tuple(delta),
                captured_indices=(
                    tuple(
                        range(
                            working_start,
                            working_start + len(working_messages),
                        )
                    )
                    + tuple(
                        range(
                            base_message_count,
                            base_message_count + len(delta),
                        )
                    )
                ),
                overlap=overlap,
                stored_message_count=len(working_messages),
            )
            origins = self._message_origins(sources)
            projection = self._history.project_context(
                owner_id,
                snapshot.messages,
                origins=origins,
                sources=sources,
            )
            self._validate_projection_sources(projection, sources)
            projection = await self._history.prepare_projection(owner_id, projection)
            prepared.append(
                PreparedStepSnapshot(
                    owner_id,
                    StoredStepSnapshot(
                        run.run_id,
                        snapshot.step_index,
                        snapshot.timestamp,
                        snapshot.state,
                        projection.digest,
                    ),
                    chunks,
                    projection,
                    target_quality,
                    chunk_session_history_item_counts,
                    chunk_execution_transcript_item_counts,
                )
            )
            working_messages.extend(delta)
            target_message_count += len(delta)
            if len(working_messages) > context_bound:
                trim = len(working_messages) - context_bound
                working_messages = working_messages[trim:]
                working_start += trim
        return PreparedStepSnapshotBatch(
            run.run_id,
            tuple(prepared),
            0,
            0,
            target_message_count,
        )

    def _validate_projection_sources(
        self,
        projection: ContextProjection,
        sources: Sequence[TranscriptMessageRef | None],
    ) -> None:
        allowed: dict[tuple[RuntimeDomain, str], list[int]] = {}
        for source in sources:
            if source is None:
                continue
            allowed.setdefault(
                (source.source_domain, source.owner_id),
                [],
            ).append(source.message_index)
        spans: dict[tuple[RuntimeDomain, str], list[tuple[int, int]]] = {}
        for key, indexes in allowed.items():
            for index in sorted(set(indexes)):
                values = spans.setdefault(key, [])
                if values and values[-1][1] == index:
                    values[-1] = (values[-1][0], index + 1)
                else:
                    values.append((index, index + 1))
        for item in projection.items:
            if not isinstance(item, TranscriptSpanRef):
                continue
            if item.end <= item.start:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            allowed_spans = spans.get((item.source_domain, item.owner_id), ())
            if not any(
                item.start >= start and item.end <= end
                for start, end in allowed_spans
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _history_id(self, run: RunRecord) -> str:
        history_id = run.metadata.get("history_id")
        return history_id or run.run_id

    async def _prepare_captured_chunks(
        self,
        owner_id: str,
        capture: TranscriptCapture,
        *,
        message_index_offset: int = 0,
    ) -> tuple[
        tuple[TranscriptChunk, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        messages = capture.messages
        origins = capture.origins
        result: list[TranscriptChunk] = []
        offset = message_index_offset + capture.first_message_index
        start = 0
        while start < len(messages):
            origin = origins[start]
            end = start + 1
            while end < len(messages) and origins[end] is origin:
                end += 1
            result.extend(
                await self._history.prepare_chunks(
                    owner_id,
                    messages[start:end],
                    first_message_index=offset,
                    origin=origin,
                )
            )
            offset += end - start
            start = end
        session_counts = tuple(
            count_session_history_items(chunk_messages)
            for chunk_messages in _chunk_message_groups(
                messages,
                result,
                message_index_offset + capture.first_message_index,
            )
        )
        execution_counts = tuple(
            count_execution_transcript_items(chunk_messages)
            for chunk_messages in _chunk_message_groups(
                messages,
                result,
                message_index_offset + capture.first_message_index,
            )
        )
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            session_counts = tuple(0 for _ in result)
        if self._runtime_domain is RuntimeDomain.CONVERSATION:
            execution_counts = tuple(0 for _ in result)
        return tuple(result), session_counts, execution_counts

    def _message_sources(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        captured_messages: Sequence[ModelMessage],
        captured_indices: Sequence[int] | None = None,
        *,
        overlap: int,
        stored_message_count: int,
    ) -> tuple[TranscriptMessageRef | None, ...]:
        sources: list[TranscriptMessageRef | None] = []
        actual_indices = (
            tuple(range(len(captured_messages)))
            if captured_indices is None
            else tuple(captured_indices)
        )
        if len(actual_indices) != len(captured_messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if overlap < 0 or overlap > len(messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            stored_message_count < overlap
            or stored_message_count + len(messages) - overlap
            != len(captured_messages)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        exact_captured = tuple(
            _exact_message_signature(value) for value in captured_messages
        )
        for index, message in enumerate(messages):
            if index < overlap:
                captured_position = stored_message_count - overlap + index
            else:
                captured_position = stored_message_count + index - overlap
            actual_index = actual_indices[captured_position]
            if exact_captured[captured_position] == _exact_message_signature(message):
                sources.append(
                    TranscriptMessageRef(
                        self._runtime_domain,
                        owner_id,
                        actual_index,
                    )
                )
                continue
            sources.append(None)
        return tuple(sources)

    def _message_origins(
        self,
        sources: Sequence[TranscriptMessageRef | None],
    ) -> tuple[TranscriptOrigin, ...]:
        return tuple(
            TranscriptOrigin.RAW
            if source is not None
            else TranscriptOrigin.UNKNOWN
            for source in sources
        )

    async def _normalize_snapshots_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        snapshots: Sequence[PreparedStepSnapshot],
    ) -> tuple[PreparedStepSnapshot, ...]:
        del transaction, run
        values = tuple(snapshots)
        if any(not isinstance(snapshot, PreparedStepSnapshot) for snapshot in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def _execution_history_guard_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str | None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None,
    ) -> tuple[ExecutionHistoryHeadRecord, StoredRecord] | None:
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            if history_head_guard is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
        if not execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._execution_repository is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if history_head_guard is not None:
            head, _record = history_head_guard
            if (
                head.execution_id != execution_id
                or head.tenant_id != self._tenant_id
                or head.state is not ExecutionHistoryState.OPEN
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return history_head_guard
        return await self._execution_repository.require_open_history_head_in_transaction(
            transaction,
            execution_id,
        )

    async def _advance_execution_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None,
    ) -> None:
        if history_head_guard is None:
            return
        if self._execution_repository is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        head, record = history_head_guard
        await self._execution_repository.replace_history_head_in_transaction(
            transaction,
            record,
            replace(head, revision=head.revision + 1),
        )

    async def sync_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.sync_projection")
        prepared = await self._prepare_snapshots(
            run,
            snapshots,
        )
        await self._store.mutate(
            lambda transaction: self._sync_projection_in_transaction(
                transaction,
                run,
                events=events,
                snapshots=prepared.snapshots,
                execution_id=execution_id,
            )
        )

    async def sync_prepared_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[PreparedStepSnapshot],
        execution_id: str | None = None,
    ) -> None:
        """Commit a prepared projection without preparing its payload twice."""
        self._ensure_open()
        require_no_run_history_lock(
            "StateStepArchive.sync_prepared_projection"
        )
        values = tuple(snapshots)
        if any(not isinstance(value, PreparedStepSnapshot) for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.mutate(
            lambda transaction: self._sync_projection_in_transaction(
                transaction,
                run,
                events=events,
                snapshots=values,
                execution_id=execution_id,
            )
        )

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[PreparedStepSnapshot],
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        self._ensure_open()
        snapshots = await self._normalize_snapshots_in_transaction(
            transaction,
            run,
            snapshots,
        )
        facts = tuple(
            ("event", event, _step_event_kind(event)) for event in events
        ) + tuple(
            ("snapshot", snapshot.stored, snapshot.stored.state)
            for snapshot in snapshots
        )
        supplied_history_head_guard = history_head_guard is not None
        history_head_guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            history_head_guard,
        )
        if not facts:
            _owner_record, created = await self._ensure_run_with_head_in_transaction(
                transaction,
                run,
            )
            if created and history_head_guard is not None and not supplied_history_head_guard:
                await self._advance_execution_history_head_in_transaction(
                    transaction,
                    history_head_guard,
                )
            return
        owner = self._run_key(run.run_id)
        owner_record = await self._ensure_run_in_transaction(transaction, run)
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
        if snapshots:
            await self._history.append_chunks(
                transaction,
                snapshots[0].owner_id,
                tuple(
                    chunk
                    for snapshot in snapshots
                    for chunk in snapshot.chunks
                ),
                tuple(
                    count
                    for snapshot in snapshots
                    for count in snapshot.chunk_session_history_item_counts
                ),
                tuple(
                    count
                    for snapshot in snapshots
                    for count in snapshot.chunk_execution_transcript_item_counts
                ),
                min(
                    (snapshot.history_quality for snapshot in snapshots),
                    key=lambda value: value is HistoryQuality.COMPLETE,
                    default=HistoryQuality.COMPLETE,
                ),
            )
            await self._history.store_projection(
                transaction,
                snapshots[-1].owner_id,
                snapshots[-1].projection,
            )
        if await transaction.guard_record(
            owner,
            expected_storage_version=owner_record.storage_version,
        ) is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await _insert_facts(transaction, tuple(stored_facts))
        if history_head_guard is not None and not supplied_history_head_guard:
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

    async def materialize_snapshot(
        self,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.materialize_snapshot")
        prepared = await self._prepare_snapshots(
            run,
            (snapshot,),
        )
        await self._store.mutate(
            lambda transaction: self._materialize_snapshot_in_transaction(
                transaction,
                run,
                prepared.snapshots[0],
                execution_id=execution_id,
            )
        )

    async def _materialize_snapshot_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        snapshot: PreparedStepSnapshot,
        *,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        if not isinstance(snapshot, PreparedStepSnapshot):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        supplied_history_head_guard = history_head_guard is not None
        history_head_guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            history_head_guard,
        )
        if await self._has_existing_fact_in_transaction(
            transaction,
            run,
            "snapshot",
            snapshot.stored,
            snapshot.stored.state,
        ):
            return
        await self._ensure_run_in_transaction(transaction, run)
        await self._history.append_chunks(
            transaction,
            snapshot.owner_id,
            snapshot.chunks,
            snapshot.chunk_session_history_item_counts or None,
            snapshot.chunk_execution_transcript_item_counts or None,
            snapshot.history_quality,
        )
        await self._history.store_projection(
            transaction,
            snapshot.owner_id,
            snapshot.projection,
        )
        await self._materialize_fact_in_transaction(
            transaction,
            run,
            "snapshot",
            snapshot.stored,
            snapshot.stored.state,
        )
        if history_head_guard is not None and not supplied_history_head_guard:
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

    async def materialize_snapshot_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        snapshot: PreparedStepSnapshot,
        *,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        require_no_run_history_lock(
            "StateStepArchive.materialize_snapshot_in_transaction"
        )
        await self._materialize_snapshot_in_transaction(
            transaction,
            run,
            snapshot,
            execution_id=execution_id,
            history_head_guard=history_head_guard,
        )

    async def sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[PreparedStepSnapshot],
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        require_no_run_history_lock(
            "StateStepArchive.sync_projection_in_transaction"
        )
        await self._sync_projection_in_transaction(
            transaction,
            run,
            events=events,
            snapshots=snapshots,
            execution_id=execution_id,
            history_head_guard=history_head_guard,
        )

    async def materialize_effect(
        self,
        run: RunRecord,
        effect: ToolEffectRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.materialize_effect")
        await self._store.mutate(
            lambda transaction: self._materialize_effect_in_transaction(
                transaction,
                run,
                effect,
                execution_id=execution_id,
            )
        )

    async def _materialize_effect_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        effect: ToolEffectRecord,
        *,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        supplied_history_head_guard = history_head_guard is not None
        history_head_guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            history_head_guard,
        )
        if await self._has_existing_fact_in_transaction(
            transaction,
            run,
            "effect",
            effect,
            effect.status,
        ):
            return
        await self._materialize_fact_in_transaction(
            transaction,
            run,
            "effect",
            effect,
            effect.status,
        )
        if history_head_guard is not None and not supplied_history_head_guard:
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
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

        owner_record = await self._ensure_run_in_transaction(transaction, run)
        if await self._has_existing_fact_in_transaction(
            transaction,
            run,
            family,
            value,
            kind,
        ):
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

    async def _ensure_run_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
    ) -> StoredRecord:
        owner_record, _created = await self._ensure_run_with_head_in_transaction(
            transaction,
            run,
        )
        return owner_record

    async def _ensure_run_with_head_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
    ) -> tuple[StoredRecord, bool]:
        owner = self._run_key(run.run_id)
        history_owner = (
            self._history_id(run)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else run.run_id
        )
        head_key = self._history.head_key(history_owner)
        records = await transaction.get_records((owner, head_key))
        owner_record = records.get(owner)
        head_record = records.get(head_key)
        if owner_record is None:
            stored_run = self._stored_run(run)
            if head_record is None:
                await transaction.insert_records(
                    (
                        stored_run,
                        self._history.empty_head_record(history_owner),
                    )
                )
                _logger.debug(
                    "step run admitted with transcript head: run=%s",
                    run.run_id,
                )
            else:
                self._history.decode_head(head_record)
                await transaction.insert_records((stored_run,))
                _logger.debug(
                    "step run admitted using existing transcript head: run=%s",
                    run.run_id,
                )
            return stored_run, True
        elif _decode_step(owner_record.data) != run:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif head_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._history.decode_head(head_record)
        return owner_record, False

    async def _has_existing_fact_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        family: str,
        value: object,
        kind: str,
    ) -> bool:
        stream = self._stream(run.run_id, family)
        subject = _step_subject(value)
        data = _encode_step(value)
        existing = await transaction.list_facts(
            FactQuery(
                stream,
                subject_digest=subject,
                latest=True,
            )
        )
        return any(fact.data == data and fact.state == kind for fact in existing)

    async def materialize_effect_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        effect: ToolEffectRecord,
        *,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        require_no_run_history_lock(
            "StateStepArchive.materialize_effect_in_transaction"
        )
        await self._materialize_effect_in_transaction(
            transaction,
            run,
            effect,
            execution_id=execution_id,
            history_head_guard=history_head_guard,
        )

    async def append_event(
        self,
        event: StepEvent,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.append_event")
        await self._append(
            event.run_id,
            "event",
            event,
            _step_event_kind(event),
            execution_id=execution_id,
        )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        require_no_run_history_lock("StateStepArchive.list_events")
        values = await self._facts(run_id, "event")
        return [_decode_step(value.data) for value in values]

    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]:
        require_no_run_history_lock("StateStepArchive.iter_messages")
        async for message in self._history.iter_messages(run_id):
            yield message

    async def load_model_context(
        self,
        *,
        run_id: str,
    ) -> tuple[object, ...]:
        require_no_run_history_lock("StateStepArchive.load_model_context")
        values = await self._history.load_model_context(
            run_id,
        )
        return values.model_messages()

    async def load_loaded_model_context(
        self,
        *,
        owner_id: str,
    ) -> LoadedModelContext:
        require_no_run_history_lock("StateStepArchive.load_loaded_model_context")
        if self._runtime_domain is RuntimeDomain.CONVERSATION:
            return await self._history.load_session_model_context(
                owner_id,
                tenant_id=self._tenant_id,
            )
        values = await self._history.load_model_context(
            owner_id,
        )
        return values

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
        require_no_run_history_lock(
            "StateStepArchive.resolve_transcript_message_refs"
        )
        return await self._history.resolve_transcript_message_refs(refs)

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        require_no_run_history_lock("StateStepArchive.iter_session_messages")
        async for message in self._history.iter_session_messages(
            history_id,
            tenant_id=tenant_id,
        ):
            yield message

    async def session_message_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int:
        require_no_run_history_lock("StateStepArchive.session_message_count")
        return await self._history.history_message_count(
            history_id,
            tenant_id=tenant_id,
        )

    async def session_history_item_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int:
        require_no_run_history_lock(
            "StateStepArchive.session_history_item_count"
        )
        return await self._history.session_history_item_total_count(
            history_id,
            tenant_id=tenant_id,
        )

    def iter_session_history_item_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._history.iter_session_history_item_range(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )

    def iter_session_message_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._history.iter_session_message_range(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )

    async def load_session_model_context(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> tuple[object, ...]:
        require_no_run_history_lock("StateStepArchive.load_session_model_context")
        return (
            await self._history.load_session_model_context(
                history_id,
                tenant_id=tenant_id,
            )
        ).model_messages()

    async def verify_snapshot_projection(
        self,
        *,
        run_id: str,
        snapshot: ContinuableSnapshot,
    ) -> bool:
        require_no_run_history_lock(
            "StateStepArchive.verify_snapshot_projection"
        )
        values = await self._facts(run_id, "snapshot", latest=True)
        if not values:
            return False
        stored = _decode_step(values[0].data)
        if not isinstance(stored, StoredStepSnapshot):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        projection = await self._history.load_projection(run_id)
        if projection is None or projection.digest != stored.projection_digest:
            return False
        context = await self._history.load_model_context(run_id)
        return (
            stored.run_id == snapshot.run_id
            and stored.step_index == snapshot.step_index
            and stored.timestamp == snapshot.timestamp
            and stored.state == snapshot.state
            and context.model_messages() == tuple(snapshot.messages)
        )

    async def save_snapshot(
        self,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.save_snapshot")
        run = await self.get_run(run_id=snapshot.run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        await self.materialize_snapshot(
            run,
            snapshot,
            execution_id=execution_id,
        )

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        require_no_run_history_lock("StateStepArchive.latest_snapshot")
        values = await self._facts(run_id, "snapshot", latest=True)
        if not values:
            return None
        latest = _decode_step(values[0].data)
        if not isinstance(latest, StoredStepSnapshot):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = (await self._history.load_model_context(run_id)).model_messages()
        run = await self.get_run(run_id=run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        latest = ContinuableSnapshot(
            run_id=latest.run_id,
            step_index=latest.step_index,
            messages=list(messages),
            conversation_id=run.conversation_id,
            parent_run_id=run.parent_run_id,
            agent_name=run.agent_name,
            timestamp=latest.timestamp,
            state=latest.state,
        )
        return latest if include_interrupted or latest.state == "complete" else None

    async def record_tool_effect(
        self,
        record: ToolEffectRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.record_tool_effect")
        await self._append(
            record.run_id,
            "effect",
            record,
            record.status,
            execution_id=execution_id,
        )

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        require_no_run_history_lock("StateStepArchive.get_tool_effect")
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
        require_no_run_history_lock(
            "StateStepArchive.list_unresolved_tool_effects"
        )
        values = [_decode_step(value.data) for value in await self._facts(run_id, "effect", latest_per_subject=True)]
        return [value for value in values if value.status == "started"]

    async def release_run(
        self,
        run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.release_run")

        async def mutate(transaction: StateTransaction) -> None:
            history_head_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
            await transaction.delete_record(self._run_key(run_id))
            await transaction.delete_sequences(
                tuple(
                    self._sequence(run_id, family)
                    for family in ("event", "snapshot", "effect")
                )
            )
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

        await self._store.mutate(mutate)
        self._context_baselines.pop(run_id, None)

    def release_runtime_cache(self, run_id: str) -> None:
        self._context_baselines.pop(run_id, None)

    async def _append(
        self,
        run_id: str,
        family: str,
        value: object,
        kind: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive._append")
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
            history_head_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
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
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

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
        require_no_run_history_lock("StateStepArchive._facts")
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
        self._durability_flights: dict[str, _RunDurabilityFlight] = {}
        self._terminal_seals: dict[str, _LocalExecutionTerminalSeal] = {}
        self._history_lock = _RunHistoryLock()
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                archive.bind_history_lock(self._history_lock)

    async def initialize(self) -> None:
        await self._staging.initialize()
        for archive in self._archives.values():
            await archive.initialize()
        self._projection_offsets.clear()
        self._projection_dirty.clear()
        self._durability_flights.clear()
        self._terminal_seals.clear()
        self._initialized = True

    async def validate_integrity(self) -> None:
        await self._ensure_business()
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                await archive.validate_integrity()

    def register_context_baseline(
        self,
        step_run_id: str,
        context: LoadedModelContext,
    ) -> None:
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                archive.register_context_baseline(step_run_id, context)

    async def register_run(
        self,
        record: RunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        async with self._history_lock.hold(record.run_id):
            self._ensure_run_mutable(record.run_id)
            self._staging.register_run_local(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_business()
        return await self._staging.get_run(run_id=run_id)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        await self._ensure_business()
        return await self._staging.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def append_event(
        self,
        event: StepEvent,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        async with self._history_lock.hold(event.run_id):
            self._ensure_run_mutable(event.run_id)
            self._staging.append_event_local(event)
            self._projection_dirty.add(event.run_id)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_business()
        return await self._staging.list_events(run_id=run_id)

    async def save_snapshot(
        self,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        while True:
            completion: asyncio.Future[None] | None = None
            recovery: StepStore | None = None
            recovery_run: RunRecord | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(snapshot.run_id):
                existing = self._durability_flights.get(snapshot.run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(snapshot.run_id)
                    self._staging.save_snapshot_local(snapshot)
                    self._projection_dirty.add(snapshot.run_id)
                    recovery = self._archives.get(RuntimeDomain.RECOVERY)
                    recovery_run = self._staging.get_run_local(snapshot.run_id)
                    if recovery is not None:
                        flight = self._install_durability_flight_locked(
                            snapshot.run_id,
                            _RunDurabilityKind.SNAPSHOT,
                        )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if recovery is None:
                return
            if recovery_run is None or flight is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)

            async def operation(
                target_recovery: StepStore = recovery,
                target_run: RunRecord = recovery_run,
                target_snapshot: ContinuableSnapshot = snapshot,
            ) -> None:
                await _materialize_snapshot(
                    target_recovery,
                    target_run,
                    target_snapshot,
                )

            async def readback(
                target_recovery: StepStore = recovery,
                target_run: RunRecord = recovery_run,
                target_snapshot: ContinuableSnapshot = snapshot,
            ) -> CommitObservation[None]:
                try:
                    observed_run = await target_recovery.get_run(
                        run_id=target_snapshot.run_id
                    )
                    observed_snapshot = await target_recovery.latest_snapshot(
                        run_id=target_snapshot.run_id,
                        include_interrupted=True,
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if observed_run == target_run and observed_snapshot == target_snapshot:
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            return

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_business()
        return await self._staging.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def record_tool_effect(
        self,
        record: ToolEffectRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        while True:
            completion: asyncio.Future[None] | None = None
            recovery: StepStore | None = None
            recovery_run: RunRecord | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(record.run_id):
                existing = self._durability_flights.get(record.run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(record.run_id)
                    self._staging.record_tool_effect_local(record)
                    recovery = self._archives.get(RuntimeDomain.RECOVERY)
                    recovery_run = self._staging.get_run_local(record.run_id)
                    if recovery is not None:
                        flight = self._install_durability_flight_locked(
                            record.run_id,
                            _RunDurabilityKind.TOOL_EFFECT,
                        )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if recovery is None:
                return
            if recovery_run is None or flight is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)

            async def operation(
                target_recovery: StepStore = recovery,
                target_run: RunRecord = recovery_run,
                target_record: ToolEffectRecord = record,
            ) -> None:
                await _materialize_effect(
                    target_recovery,
                    target_run,
                    target_record,
                )

            async def readback(
                target_recovery: StepStore = recovery,
                target_record: ToolEffectRecord = record,
            ) -> CommitObservation[None]:
                try:
                    observed = await target_recovery.get_tool_effect(
                        run_id=target_record.run_id,
                        tool_call_id=target_record.tool_call_id,
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if observed == target_record:
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            return

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

    async def load_loaded_model_context(
        self,
        runtime_domain: RuntimeDomain,
        owner_id: str,
    ) -> LoadedModelContext:
        archive = self._archives.get(runtime_domain)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.load_loaded_model_context(
            owner_id=owner_id,
        )

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.resolve_transcript_message_refs(refs)

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        async for message in archive.transcript_repository.iter_session_messages(
            history_id,
            tenant_id=tenant_id,
        ):
            yield message

    async def session_message_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.transcript_repository.history_message_count(
            history_id,
            tenant_id=tenant_id,
        )

    async def session_history_item_count(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> int:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.session_history_item_count(
            history_id,
            tenant_id=tenant_id,
        )

    def iter_session_history_item_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._iter_session_history_item_range(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )

    async def _iter_session_history_item_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        async for item in archive.iter_session_history_item_range(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        ):
            yield item

    async def execution_transcript_item_count(self, run_id: str) -> int:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.transcript_repository.execution_transcript_item_count(
            run_id
        )

    def iter_execution_transcript_item_range(
        self,
        run_id: str,
        *,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._iter_execution_transcript_item_range(
            run_id,
            start=start,
            end=end,
        )

    async def _iter_execution_transcript_item_range(
        self,
        run_id: str,
        *,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        async for message in archive.transcript_repository.iter_execution_transcript_item_range(
            run_id,
            start=start,
            end=end,
        ):
            yield message

    def iter_session_message_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._iter_session_message_range(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )

    async def _iter_session_message_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        async for message in archive.transcript_repository.iter_session_message_range(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        ):
            yield message

    async def load_session_model_context(
        self,
        history_id: str,
    ) -> LoadedModelContext:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.load_loaded_model_context(
            owner_id=history_id,
        )

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

    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: str | None = None,
    ) -> None:
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        destination = self._archives.get(target)
        if recovery is None or destination is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        run = await recovery.get_run(run_id=step_run_id)
        snapshot = await recovery.latest_snapshot(run_id=step_run_id)
        if run is None or snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if target is RuntimeDomain.EXECUTION and execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        while True:
            completion: asyncio.Future[None] | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    if target is RuntimeDomain.EXECUTION:
                        self._ensure_run_mutable(step_run_id)
                    flight = self._install_durability_flight_locked(
                        step_run_id,
                        _RunDurabilityKind.RECOVERY_MATERIALIZATION,
                    )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if flight is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def operation() -> None:
                await _materialize_snapshot(
                    destination,
                    run,
                    snapshot,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    observed_run = await destination.get_run(run_id=run.run_id)
                    observed_snapshot = await destination.latest_snapshot(
                        run_id=run.run_id,
                        include_interrupted=True,
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if observed_run == run and observed_snapshot == snapshot:
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            return

    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        await self._ensure_business()
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        ordered_run_ids = tuple(sorted(dict.fromkeys(run_ids)))
        if not ordered_run_ids:
            _logger.info(
                "execution terminal seal prepared: execution=%s runs=0",
                execution_id,
            )
            return ExecutionTerminalSealPlan(
                execution_id,
                binding_digest,
                (),
                (),
            )
        captured: list[tuple[_LocalExecutionTerminalSeal, ExecutionProjectionBatch]] = []
        installed: list[str] = []
        terminal_attempt_token = uuid4().hex
        try:
            for run_id in ordered_run_ids:
                while True:
                    archive_run: RunRecord | None = None
                    captured_seal: _LocalExecutionTerminalSeal | None = None
                    needs_archive_run = False
                    async with self._history_lock.hold(run_id):
                        existing = self._durability_flights.get(run_id)
                        if existing is not None:
                            completion = existing.completion
                        else:
                            seal = self._terminal_seals.get(run_id)
                            if seal is None:
                                seal = _LocalExecutionTerminalSeal(
                                    execution_id,
                                    terminal_attempt_token,
                                )
                                self._terminal_seals[run_id] = seal
                                self._install_durability_flight_locked(
                                    run_id,
                                    _RunDurabilityKind.TERMINAL,
                                    token=terminal_attempt_token,
                                )
                                installed.append(run_id)
                            elif seal.execution_id != execution_id or seal.token != terminal_attempt_token:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                            captured_seal = seal
                            run = self._staging.get_run_local(run_id)
                            if run is None:
                                needs_archive_run = True
                            else:
                                projection = self._capture_projection_snapshot_locked(run_id)
                                captured.append((seal, projection))
                                break
                    if needs_archive_run:
                        archive_run = await archive.get_run(run_id=run_id)
                        if archive_run is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if captured_seal is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        captured.append(
                            (
                                captured_seal,
                                ExecutionProjectionBatch(
                                    archive_run,
                                    (),
                                    (),
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ),
                            )
                        )
                        break
                    await asyncio.shield(completion)
            prepared: list[PreparedExecutionProjection] = []
            durable_heads = await archive.execution_history_heads(
                tuple(projection.run.run_id for _seal, projection in captured)
            )
            for _seal, projection in captured:
                durable_head = durable_heads.get(projection.run.run_id)
                if durable_head is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                batch = await archive.prepare_snapshots_after_seal(
                    projection.run,
                    projection.snapshots,
                )
                prepared.append(
                    PreparedExecutionProjection(
                        projection.run,
                        projection.events,
                        batch.snapshots,
                        projection.base_event_offset,
                        projection.base_snapshot_offset,
                        durable_head.event_count
                        + projection.target_event_offset
                        - projection.base_event_offset,
                        durable_head.snapshot_count
                        + projection.target_snapshot_offset
                        - projection.base_snapshot_offset,
                        batch.target_transcript_message_count,
                        "empty"
                        if not batch.snapshots
                        and durable_head.projection_digest == "empty"
                        else (
                            batch.snapshots[-1].projection.digest
                            if batch.snapshots
                            else durable_head.projection_digest
                        ),
                    )
                )
            plan = ExecutionTerminalSealPlan(
                execution_id,
                binding_digest,
                tuple(prepared),
                tuple(
                    (projection.run.run_id, seal.token)
                    for seal, projection in captured
                ),
                terminal_attempt_token,
            )
            _logger.info(
                "execution terminal seal prepared: execution=%s runs=%s",
                execution_id,
                len(plan.projections),
            )
            return plan
        except BaseException:
            for run_id in installed:
                await self._release_terminal_seal_if_owned(
                    run_id,
                    execution_id=execution_id,
                    token=terminal_attempt_token,
                )
            raise

    async def finalize_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        for projection in plan.projections:
            completion: asyncio.Future[None] | None = None
            async with self._history_lock.hold(projection.run.run_id):
                seal = self._terminal_seals.get(projection.run.run_id)
                if seal is None or seal.token != plan.token_for(
                    projection.run.run_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                flight = self._durability_flights.get(projection.run.run_id)
                if (
                    flight is None
                    or flight.kind is not _RunDurabilityKind.TERMINAL
                    or flight.token != seal.token
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                del self._durability_flights[projection.run.run_id]
                completion = flight.completion
                offset = self._projection_offsets.setdefault(
                    projection.run.run_id,
                    _ProjectionOffset(),
                )
                offset.events = max(offset.events, projection.target_event_offset)
                offset.snapshots = max(offset.snapshots, projection.target_snapshot_offset)
                offset.transcript_messages = max(
                    offset.transcript_messages,
                    projection.target_transcript_message_count,
                )
                self._projection_dirty.discard(projection.run.run_id)
            if completion is not None and not completion.done():
                completion.set_result(None)
        _logger.info(
            "execution terminal seal finalized: execution=%s runs=%s",
            plan.execution_id,
            len(plan.projections),
        )

    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None:
        for run_id in dict.fromkeys(
            projection.run.run_id for projection in plan.projections
        ):
            await self._release_terminal_seal_if_owned(
                run_id,
                execution_id=plan.execution_id,
                token=plan.token_for(run_id),
            )

    async def _release_terminal_seal_if_owned(
        self,
        run_id: str,
        *,
        execution_id: str,
        token: str,
    ) -> None:
        """Release one run's terminal seal only when this attempt owns it."""
        completion: asyncio.Future[None] | None = None
        async with self._history_lock.hold(run_id):
            seal = self._terminal_seals.get(run_id)
            if seal is None:
                return
            if seal.execution_id != execution_id or seal.token != token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            flight = self._durability_flights.get(run_id)
            if flight is not None:
                if (
                    flight.kind is not _RunDurabilityKind.TERMINAL
                    or flight.token != token
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                del self._durability_flights[run_id]
                completion = flight.completion
            del self._terminal_seals[run_id]
        if completion is not None and not completion.done():
            completion.set_result(None)

    def _capture_projection_snapshot_locked(
        self,
        run_id: str,
    ) -> ExecutionProjectionBatch:
        offset = self._projection_offsets.setdefault(run_id, _ProjectionOffset())
        projection = self._staging.capture_projection_local(run_id, offset)
        if projection is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return projection

    def _ensure_run_mutable(self, run_id: str) -> None:
        if run_id in self._terminal_seals:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def capture_execution_projection(
        self,
        step_run_id: str,
    ) -> "tuple[CapturedExecutionProjection, _RunProjectionFlight] | None":
        """CAPTURE: snapshot staged state under the run lock with no durable I/O."""
        await self._ensure_business()
        while True:
            completion: asyncio.Future[None] | None = None
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(step_run_id)
                    offset = self._projection_offsets.setdefault(
                        step_run_id,
                        _ProjectionOffset(),
                    )
                    projection = self._staging.capture_projection_local(
                        step_run_id,
                        offset,
                    )
                    if projection is None:
                        return None
                    flight = self._install_durability_flight_locked(
                        step_run_id,
                        _RunDurabilityKind.PROJECTION,
                    )
                    captured = CapturedExecutionProjection(
                        projection.run,
                        projection.events,
                        projection.snapshots,
                        projection.base_event_offset,
                        projection.base_snapshot_offset,
                        projection.target_event_offset,
                        projection.target_snapshot_offset,
                    )
                    _logger.debug(
                        "projection flight captured: run=%s token=%s "
                        "events=%s snapshots=%s",
                        step_run_id,
                        flight.token,
                        len(captured.events),
                        len(captured.snapshots),
                    )
                    return captured, flight
            await asyncio.shield(completion)

    async def wait_projection_flight(self, step_run_id: str) -> None:
        """Wait for an active flight without holding the run lock."""
        await self._ensure_business()
        while True:
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is None:
                    return
                completion = existing.completion
            await asyncio.shield(completion)

    async def abandon_execution_projection(
        self,
        flight: _RunProjectionFlight,
    ) -> None:
        """Remove a flight after a definitely-not-committed outcome."""
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.info(
            "projection flight abandoned: run=%s token=%s",
            flight.run_id,
            flight.token,
        )

    async def finalize_execution_projection(
        self,
        flight: _RunProjectionFlight,
        captured: CapturedExecutionProjection,
        *,
        target_transcript_message_count: int | None = None,
    ) -> None:
        """FINALIZE: advance offsets and clear dirty state after durable success."""
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
            offset = self._projection_offsets.setdefault(
                flight.run_id,
                _ProjectionOffset(),
            )
            offset.events = max(offset.events, captured.target_event_offset)
            offset.snapshots = max(offset.snapshots, captured.target_snapshot_offset)
            if target_transcript_message_count is not None:
                offset.transcript_messages = max(
                    offset.transcript_messages,
                    target_transcript_message_count,
                )
            self._projection_dirty.discard(flight.run_id)
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.debug(
            "projection flight finalized: run=%s token=%s events=%s snapshots=%s",
            flight.run_id,
            flight.token,
            captured.target_event_offset,
            captured.target_snapshot_offset,
        )

    async def commit_captured_execution_projection(
        self,
        captured: CapturedExecutionProjection,
        flight: _RunProjectionFlight,
        *,
        execution_id: str,
    ) -> None:
        """PREPARE + DURABLE COMMIT without the run lock, then FINALIZE."""
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            if not isinstance(archive, _StepArchiveBatch):
                await self.abandon_execution_projection(flight)
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

            async def operation() -> None:
                await _sync_projection(
                    archive,
                    captured.run,
                    captured.events,
                    captured.snapshots,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    stored_run = await archive.get_run(run_id=captured.run.run_id)
                    stored_events = await archive.list_events(
                        run_id=captured.run.run_id
                    )
                    stored_snapshot = await archive.latest_snapshot(
                        run_id=captured.run.run_id,
                        include_interrupted=True,
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if stored_run != captured.run:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.events and tuple(stored_events[-len(captured.events) :]) != captured.events:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.snapshots and stored_snapshot != captured.snapshots[-1]:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(DurableCommitState.COMMITTED)

            result = await run_durable_commit(operation, readback)
            if result.state is DurableCommitState.COMMITTED:
                await self.finalize_execution_projection(flight, captured)
                if result.cancelled:
                    raise asyncio.CancelledError
                return
            if result.state is DurableCommitState.NOT_COMMITTED:
                await self.abandon_execution_projection(flight)
                if result.error is not None:
                    raise result.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                await self._fence_durability_flight(
                    flight,
                    AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        "projection commit left partial durable state",
                    ),
                )
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "projection commit left partial durable state",
                ) from result.error
            _logger.error(
                "projection commit unresolved; flight retained: run=%s token=%s",
                flight.run_id,
                flight.token,
            )
            unknown = AIError(
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                "projection commit outcome is unresolved",
            )
            await self._fence_durability_flight(flight, unknown)
            raise unknown from result.error
        if not captured.events and not captured.snapshots:
            await self.finalize_execution_projection(flight, captured)
            return
        started = monotonic()
        try:
            prepared = await archive.prepare_snapshots(
                captured.run,
                captured.snapshots,
            )
        except BaseException:
            await self.abandon_execution_projection(flight)
            raise
        expected_head = ExecutionRunSealHead(
            captured.run.run_id,
            captured.target_event_offset,
            captured.target_snapshot_offset,
            prepared.target_transcript_message_count,
            prepared.snapshots[-1].projection.digest
            if prepared.snapshots
            else "empty",
        )

        async def operation() -> ExecutionRunSealHead:
            await archive.sync_prepared_projection(
                captured.run,
                events=captured.events,
                snapshots=prepared.snapshots,
                execution_id=execution_id,
            )
            event_count, snapshot_count, message_count, projection_digest = (
                await archive.execution_history_head(captured.run.run_id)
            )
            if (
                event_count != expected_head.event_count
                or snapshot_count != expected_head.snapshot_count
                or message_count != expected_head.transcript_message_count
                or (
                    prepared.snapshots
                    and projection_digest != expected_head.projection_digest
                )
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return ExecutionRunSealHead(
                captured.run.run_id,
                event_count,
                snapshot_count,
                message_count,
                projection_digest,
            )

        async def readback() -> CommitObservation[ExecutionRunSealHead]:
            try:
                event_count, snapshot_count, message_count, projection_digest = (
                    await archive.execution_history_head(captured.run.run_id)
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            if event_count != expected_head.event_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if snapshot_count != expected_head.snapshot_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if message_count != expected_head.transcript_message_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if prepared.snapshots and projection_digest != expected_head.projection_digest:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            return CommitObservation(
                DurableCommitState.COMMITTED,
                value=ExecutionRunSealHead(
                    captured.run.run_id,
                    event_count,
                    snapshot_count,
                    message_count,
                    projection_digest,
                ),
            )

        result = await run_durable_commit(operation, readback)
        if result.state is DurableCommitState.COMMITTED:
            if result.value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self.finalize_execution_projection(
                flight,
                captured,
                target_transcript_message_count=result.value.transcript_message_count,
            )
            if result.cancelled:
                raise asyncio.CancelledError
        elif result.state is DurableCommitState.NOT_COMMITTED:
            await self.abandon_execution_projection(flight)
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            integrity = AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "projection commit left partial durable state",
            )
            await self._fence_durability_flight(flight, integrity)
            raise integrity from result.error
        else:
            _logger.error(
                "projection commit unresolved; flight retained: run=%s token=%s",
                flight.run_id,
                flight.token,
            )
            unknown = AIError(
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                "projection commit outcome is unresolved",
            )
            await self._fence_durability_flight(flight, unknown)
            raise unknown from result.error
        _logger.debug(
            "step projection flushed: domain=%s backend=%s run=%s "
            "events=%s snapshots=%s duration_ms=%.3f",
            RuntimeDomain.EXECUTION.value,
            type(archive).__name__,
            flight.run_id,
            len(captured.events),
            len(captured.snapshots),
            (monotonic() - started) * 1000,
        )

    async def flush_execution_projection(
        self,
        step_run_id: str,
        *,
        execution_id: str,
    ) -> None:
        captured = await self.capture_execution_projection(step_run_id)
        if captured is None:
            return
        projection, flight = captured
        await self.commit_captured_execution_projection(
            projection,
            flight,
            execution_id=execution_id,
        )

    async def flush_dirty_execution_projections(self, *, execution_id: str) -> None:
        for run_id in tuple(self._projection_dirty):
            await self.flush_execution_projection(run_id, execution_id=execution_id)


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

    async def release_staging_many(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None:
        for run_id in dict.fromkeys(candidate_step_run_ids):
            while True:
                completion: asyncio.Future[None] | None = None
                seal_owner: str | None = None
                seal_token: str | None = None
                async with self._history_lock.hold(run_id):
                    existing = self._durability_flights.get(run_id)
                    if existing is not None:
                        completion = existing.completion
                    else:
                        if run_id in self._projection_dirty:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        self._staging.release_run_local(run_id)
                        self._projection_offsets.pop(run_id, None)
                        self._projection_dirty.discard(run_id)
                        for archive in self._archives.values():
                            if isinstance(archive, StateStepArchive):
                                archive.release_runtime_cache(run_id)
                        seal = self._terminal_seals.get(run_id)
                        if seal is not None and seal.execution_id == execution_id:
                            seal_owner = seal.execution_id
                            seal_token = seal.token
                if completion is None:
                    break
                await asyncio.shield(completion)
            if seal_owner is not None and seal_token is not None:
                await self._release_terminal_seal_if_owned(
                    run_id,
                    execution_id=seal_owner,
                    token=seal_token,
                )
                _logger.warning(
                    "terminal seal discarded on staging release: run=%s "
                    "execution=%s",
                    run_id,
                    seal_owner,
                )

    async def release_archive(
        self,
        runtime_domain: RuntimeDomain,
        step_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        archive = self._archives.get(runtime_domain)
        if archive is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        while True:
            completion: asyncio.Future[None] | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    if runtime_domain is RuntimeDomain.EXECUTION:
                        self._ensure_run_mutable(step_run_id)
                    self._projection_offsets.pop(step_run_id, None)
                    self._projection_dirty.discard(step_run_id)
                    flight = self._install_durability_flight_locked(
                        step_run_id,
                        _RunDurabilityKind.RELEASE,
                    )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if flight is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def operation() -> None:
                await archive.release_run(
                    step_run_id,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    observed = await archive.get_run(run_id=step_run_id)
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                return (
                    CommitObservation(DurableCommitState.COMMITTED)
                    if observed is None
                    else CommitObservation(DurableCommitState.NOT_COMMITTED)
                )

            await self._settle_durability_flight(flight, operation, readback)
            return

    def _install_durability_flight_locked(
        self,
        run_id: str,
        kind: _RunDurabilityKind,
        *,
        token: str | None = None,
    ) -> _RunDurabilityFlight:
        existing = self._durability_flights.get(run_id)
        if existing is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        flight = _RunDurabilityFlight(
            run_id,
            token or uuid4().hex,
            kind,
            asyncio.get_running_loop().create_future(),
        )
        self._durability_flights[run_id] = flight
        _logger.debug(
            "durability flight captured: run=%s token=%s kind=%s",
            run_id,
            flight.token,
            kind.value,
        )
        return flight

    async def _finalize_durability_flight(
        self,
        flight: _RunDurabilityFlight,
    ) -> None:
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.debug(
            "durability flight finalized: run=%s token=%s kind=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
        )

    async def _abandon_durability_flight(
        self,
        flight: _RunDurabilityFlight,
    ) -> None:
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.info(
            "durability flight abandoned: run=%s token=%s kind=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
        )

    async def _fence_durability_flight(
        self,
        flight: _RunDurabilityFlight,
        error: AIError,
    ) -> None:
        async with self._history_lock.hold(flight.run_id):
            current = self._durability_flights.get(flight.run_id)
            if current is not flight or current.token != flight.token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            completion = flight.completion
        if not completion.done():
            completion.set_exception(error)

            def consume(future: asyncio.Future[None]) -> None:
                future.exception()

            completion.add_done_callback(consume)
        _logger.error(
            "durability flight fenced: run=%s token=%s kind=%s code=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
            error.code.value,
        )

    async def _settle_durability_flight(
        self,
        flight: _RunDurabilityFlight,
        operation: Callable[[], Awaitable[None]],
        readback: Callable[[], Awaitable[CommitObservation[None]]],
    ) -> None:
        result = await run_durable_commit(operation, readback)
        if result.state is DurableCommitState.COMMITTED:
            await self._finalize_durability_flight(flight)
            if result.cancelled:
                raise asyncio.CancelledError
            return
        if result.state is DurableCommitState.NOT_COMMITTED:
            await self._abandon_durability_flight(flight)
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            integrity = AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "durability flight left partial durable state",
            )
            await self._fence_durability_flight(flight, integrity)
            raise integrity from result.error
        _logger.error(
            "durability flight unresolved: run=%s token=%s kind=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
        )
        unknown = AIError(
            ErrorCode.STORAGE_COMMIT_UNKNOWN,
            "durability flight commit outcome is unresolved",
        )
        await self._fence_durability_flight(flight, unknown)
        raise unknown from result.error

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
    *,
    execution_id: str | None = None,
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.sync_projection(
            run,
            events=events,
            snapshots=snapshots,
            execution_id=execution_id,
        )
        return
    if await target.get_run(run_id=run.run_id) is None:
        await target.register_run(run, execution_id=execution_id)
    for event in events:
        await target.append_event(event, execution_id=execution_id)
    for snapshot in snapshots:
        await target.save_snapshot(snapshot, execution_id=execution_id)


async def _materialize_effect(
    target: StepStore,
    run: RunRecord,
    effect: ToolEffectRecord,
    *,
    execution_id: str | None = None,
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.materialize_effect(
            run,
            effect,
            execution_id=execution_id,
        )
        return
    if await target.get_run(run_id=run.run_id) is None:
        await target.register_run(run, execution_id=execution_id)
    await target.record_tool_effect(effect, execution_id=execution_id)


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


async def _materialize_snapshot(
    target: StepStore,
    run: RunRecord,
    snapshot: ContinuableSnapshot,
    *,
    execution_id: str | None = None,
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.materialize_snapshot(
            run,
            snapshot,
            execution_id=execution_id,
        )
        return
    existing_run = await target.get_run(run_id=run.run_id)
    existing_snapshot = await target.latest_snapshot(
        run_id=run.run_id,
        include_interrupted=True,
    )
    if existing_run == run and existing_snapshot == snapshot:
        return
    await target.register_run(run, execution_id=execution_id)
    await target.save_snapshot(snapshot, execution_id=execution_id)


def _encode_step(value: object) -> dict[str, object]:
    return _encode_step_envelope(value)


def _step_subject(value: object) -> bytes | None:
    if isinstance(value, ToolEffectRecord):
        return subject_digest(["tool_call", value.tool_call_id])
    return None


def _step_event_kind(value: StepEvent) -> str:
    return str(value.kind)


def _decode_step(value: Mapping[str, object]) -> object:
    return _decode_step_envelope(value)


__all__ = [
    "ExecutionProjectionBatch",
    "ExecutionTerminalSealPlan",
    "InMemoryStepArchive",
    "LockOrderError",
    "PreparedExecutionProjection",
    "PreparedStepSnapshot",
    "PreparedStepSnapshotBatch",
    "RuntimeStepStore",
    "StagingStepStore",
    "StateStepArchive",
]
