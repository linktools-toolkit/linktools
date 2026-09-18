#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step staging, archive, projection, and history-lock owners."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import ModelMessage

from ...core import canonical_json_bytes
from ...errors import AIError, ErrorCode
from ...storage import ObjectStore, StoredPayload
from .._message import decode_model_messages
from .._model_interaction import (
    StagedContextSpan,
    StagedModelInteraction,
    context_projection_to_durable,
    extend_prefix_digest,
    message_prefix_digest,
)
from ._codec import (
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_step_envelope,
)
from ._contracts import (
    ContextProjection,
    ExecutionHistoryHeadRecord,
    ExecutionHistoryState,
    ExecutionRunSealHead,
    HistoryQuality,
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    ModelInteractionRecord,
    RuntimePayloadRef,
    StoredStepSnapshot,
    TranscriptChunk,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptSpanRef,
)
from ._history import (
    TranscriptCapture,
    TranscriptRepository,
    _conversation_overlap_signature,
    _exact_message_signature,
    _overlap_signature,
    suffix_prefix_overlap,
)
from ._plan import RuntimeDomain
from ._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    StepStore,
)
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
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None: ...

    async def materialize_snapshot(
        self,
        run: RunRecord,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None: ...


@dataclass(slots=True)
class _ProjectionOffset:
    events: int = 0
    snapshots: int = 0
    transcript_messages: int = 0
    interactions: int = 0


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
    interactions: tuple[StagedModelInteraction, ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0


@dataclass(frozen=True, slots=True)
class PreparedStepSnapshot:
    owner_id: str
    stored: StoredStepSnapshot
    chunks: tuple[TranscriptChunk, ...]
    projection: ContextProjection
    history_quality: HistoryQuality = HistoryQuality.COMPLETE


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
    interactions: tuple["ModelInteractionRecord", ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0

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


class _RunDurabilityKind(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    PROJECTION = "projection"
    SNAPSHOT = "snapshot"
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
    interactions: tuple[StagedModelInteraction, ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0


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
                async with self._guard:
                    entry.owner = None
                    entry.depth = 0
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
        self._interactions: dict[str, list[StagedModelInteraction]] = {}
        self._payloads: dict[str, dict[str, bytes]] = {}
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

    def intern_payload(self, run_id: str, payload: bytes) -> tuple[str, int]:
        self._ensure_open()
        digest = hashlib.sha256(payload).hexdigest()
        self._payloads.setdefault(run_id, {}).setdefault(digest, bytes(payload))
        return digest, len(payload)

    def staged_payload(self, run_id: str, digest: str) -> bytes:
        self._ensure_open()
        try:
            return self._payloads[run_id][digest]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def stage_model_interaction(self, interaction: object) -> None:
        self._ensure_open()
        if not isinstance(interaction, StagedModelInteraction):
            raise TypeError("staged model interaction is invalid")
        self._interactions.setdefault(interaction.run_id, []).append(interaction)

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        self._ensure_open()
        _validate_interaction_page(after_request_sequence, limit)
        selected: list[object] = []
        for interaction in self._interactions.get(run_id, ()):
            if (
                after_request_sequence is not None
                and interaction.request_sequence <= after_request_sequence
            ):
                continue
            selected.append(interaction)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    async def resolve_model_interaction(self, interaction: object) -> object:
        del interaction
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]:
        del interactions
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def load_loaded_model_context(
        self,
        *,
        owner_id: str,
    ) -> LoadedModelContext:
        snapshot = await self.latest_snapshot(
            run_id=owner_id,
            include_interrupted=True,
        )
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        messages = (
            snapshot.messages
            if snapshot.context_messages is None
            else snapshot.context_messages
        )
        return LoadedModelContext(
            tuple(LoadedContextMessage(message, None) for message in messages)
        )

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

    def release_run_local(self, run_id: str) -> None:
        self._ensure_open()
        self._runs.pop(run_id, None)
        self._events.pop(run_id, None)
        self._snapshots.pop(run_id, None)
        self._interactions.pop(run_id, None)
        self._payloads.pop(run_id, None)

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
        interactions = self._interactions.get(run_id, ())
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
            tuple(interactions[offset.interactions:]),
            offset.interactions,
            len(interactions),
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
        interactions: Sequence[ModelInteractionRecord] = (),
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
            interaction_values = self._interactions.setdefault(run.run_id, [])
            for interaction in interactions:
                if interaction not in interaction_values:
                    interaction_values.append(interaction)

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
            interactions=(),
            execution_id=execution_id,
        )

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
        if not refs:
            return ()
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]:
        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)
        if snapshot is not None:
            for message in snapshot.messages:
                yield message

    async def transcript_message_count(self, run_id: str) -> int:
        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)
        return 0 if snapshot is None else len(snapshot.messages)

    async def iter_message_range(
        self,
        *,
        run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)
        total = 0 if snapshot is None else len(snapshot.messages)
        if end > total:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if snapshot is not None:
            for message in snapshot.messages[start:end]:
                yield message

    async def load_model_context(self, *, run_id: str) -> tuple[object, ...]:
        snapshot = await self.latest_snapshot(run_id=run_id, include_interrupted=True)
        return () if snapshot is None else tuple(snapshot.messages)

    async def resolve_model_interaction(
        self,
        interaction: object,
    ) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...] | None, bytes]:
        if not isinstance(interaction, ModelInteractionRecord):
            raise TypeError("model interaction is invalid")
        snapshot = await self.latest_snapshot(
            run_id=interaction.run_id,
            include_interrupted=True,
        )
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        def resolve(projection: ContextProjection) -> tuple[ModelMessage, ...]:
            values: list[ModelMessage] = []
            for item in projection.items:
                if isinstance(item, TranscriptSpanRef):
                    if item.start < 0 or item.end > len(snapshot.messages):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    values.extend(snapshot.messages[item.start : item.end])
                    continue
                payload = item.content.payload
                if payload.kind != "inline":
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                raw = payload.decode()
                if not isinstance(raw, bytes):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values.extend(decode_model_messages(raw))
            return tuple(values)

        envelope = interaction.request_envelope.payload.decode()
        if not isinstance(envelope, bytes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        response = (
            None
            if interaction.response_context is None
            else resolve(interaction.response_context)
        )
        return resolve(interaction.request_context), response, envelope

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]:
        return [
            await self.resolve_model_interaction(interaction)
            for interaction in interactions
        ]


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

    async def execution_history_head_record(
        self,
        run_id: str,
    ) -> ExecutionRunSealHead:
        values = await self.execution_history_heads((run_id,))
        head = values.get(run_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return head

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
                self._sequence(run_id, "interaction"),
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
            interaction_count = sequences.get(
                self._sequence(run_id, "interaction"), 0
            )
            if head_record is None:
                if (
                    run_record is not None
                    or projection_record is not None
                    or event_count
                    or snapshot_count
                    or interaction_count
                ):
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
                interaction_count,
            )
        return result

    async def resolve_model_interaction(
        self,
        interaction: object,
    ) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...] | None, bytes]:
        if not isinstance(interaction, ModelInteractionRecord):
            raise TypeError("model interaction is invalid")
        request = await self._history.load_projected_context(
            interaction.run_id,
            interaction.request_context,
        )
        response = None
        if interaction.response_context is not None:
            response = (
                await self._history.load_projected_context(
                    interaction.run_id,
                    interaction.response_context,
                )
            ).model_messages()
        envelope = await self._history.read_payload(
            interaction.request_envelope.payload
        )
        return request.model_messages(), response, envelope

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]:
        require_no_run_history_lock(
            "StateStepArchive.resolve_model_interactions"
        )
        values = tuple(interactions)
        if not values:
            return []
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise TypeError("model interaction is invalid")
        records = tuple(value for value in values if isinstance(value, ModelInteractionRecord))
        projections = tuple(
            projection
            for record in records
            for projection in (
                record.request_context,
                *(
                    ()
                    if record.response_context is None
                    else (record.response_context,)
                ),
            )
        )
        contexts = await self._history.load_projected_contexts(
            records[0].run_id,
            projections,
        )
        envelopes: dict[str, bytes] = {}
        result: list[object] = []
        context_index = 0
        for record in records:
            request = contexts[context_index].model_messages()
            context_index += 1
            response = None
            if record.response_context is not None:
                response = contexts[context_index].model_messages()
                context_index += 1
            digest = record.request_envelope.payload.digest
            if digest not in envelopes:
                envelopes[digest] = await self._history.read_payload(
                    record.request_envelope.payload
                )
            result.append((request, response, envelopes[digest]))
        return result

    async def prepare_interactions(
        self,
        run: RunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> tuple[ModelInteractionRecord, ...]:
        values = tuple(interactions)
        if not values:
            return ()
        request_sequences: set[int] = set()
        for interaction in values:
            if (
                interaction.run_id != run.run_id
                or interaction.request_sequence in request_sequences
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request_sequences.add(interaction.request_sequence)
        max_source_count = max(
            (
                projection.source_message_count
                for interaction in values
                for projection in (
                    interaction.request_context,
                    *(
                        ()
                        if interaction.response_context is None
                        else (interaction.response_context,)
                    ),
                )
                if projection.source_prefix_digest != "0" * 64
            ),
            default=0,
        )
        prefix_messages = (
            tuple(source_messages)
            if source_messages is not None
            and len(source_messages) >= max_source_count
            else await self._history.load_messages(run.run_id)
        )
        prefix_checkpoints = {0: message_prefix_digest(())}
        prefix_digest = prefix_checkpoints[0]
        for index, message in enumerate(prefix_messages[:max_source_count], 1):
            prefix_digest = extend_prefix_digest(prefix_digest, message)
            prefix_checkpoints[index] = prefix_digest
        _logger.debug(
            "model interaction prefix checkpoints: run=%s counts=%s kinds=%s",
            run.run_id,
            tuple(prefix_checkpoints),
            tuple(type(message).__name__ for message in prefix_messages),
        )
        for interaction in values:
            for projection in (
                interaction.request_context,
                *(
                    ()
                    if interaction.response_context is None
                    else (interaction.response_context,)
                ),
            ):
                if projection.source_prefix_digest == "0" * 64:
                    continue
                if projection.source_message_count > len(prefix_messages):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if any(
                    isinstance(item, StagedContextSpan)
                    and item.end > len(prefix_messages)
                    for item in projection.items
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if prefix_checkpoints[projection.source_message_count] != (
                    projection.source_prefix_digest
                ):
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        "model interaction source prefix digest mismatch",
                    )
        result: list[ModelInteractionRecord] = []
        for staged in values:
            request_context = context_projection_to_durable(
                staged.request_context,
                owner_id=run.run_id,
                source_domain=self._runtime_domain,
                payload=payload,
            )
            request_context = await self._history.prepare_projection(
                run.run_id,
                request_context,
            )
            request_envelope = await self._prepare_inline_payload(
                run.run_id,
                RuntimePayloadRef(
                    StoredPayload.inline_bytes(payload(staged.request_envelope_digest)),
                    self._runtime_domain,
                ),
            )
            response_context = None
            if staged.response_context is not None:
                response_context = context_projection_to_durable(
                    staged.response_context,
                    owner_id=run.run_id,
                    source_domain=self._runtime_domain,
                    payload=payload,
                )
                response_context = await self._history.prepare_projection(
                    run.run_id,
                    response_context,
                )
            result.append(
                ModelInteractionRecord(
                    staged.run_id,
                    staged.step_index,
                    staged.request_sequence,
                    staged.purpose,
                    staged.output_retry_index,
                    staged.model,
                    request_context,
                    request_envelope,
                    response_context,
                    staged.status,
                    staged.error_code,
                    staged.duration_ns,
                    staged.usage,
                )
            )
        return tuple(result)

    async def _prepare_inline_payload(
        self,
        run_id: str,
        content: RuntimePayloadRef,
    ) -> RuntimePayloadRef:
        projection = await self._history.prepare_projection(
            run_id,
            ContextProjection((InlineContextBlock(content),)),
        )
        item = projection.items[0]
        if not isinstance(item, InlineContextBlock):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return item.content

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
        record = await self.execution_history_head_record(projection.run.run_id)
        return head == (
            projection.target_event_offset,
            projection.target_snapshot_offset,
            projection.target_transcript_message_count,
            projection.projection_digest,
        ) and record.interaction_count == projection.target_interaction_offset

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
                observed_head=head,
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
            chunks = await self._prepare_captured_chunks(
                owner_id,
                capture,
                message_index_offset=0,
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
            projection_messages = (
                incoming
                if snapshot.context_messages is None
                else tuple(snapshot.context_messages)
            )
            projection_sources = self._projection_sources(
                projection_messages,
                incoming,
                sources,
            )
            projection = self._history.project_context(
                owner_id,
                projection_messages,
                origins=self._message_origins(projection_sources),
                sources=projection_sources,
            )
            self._validate_projection_sources(projection, projection_sources)
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
                        snapshot.context_messages is not None,
                    ),
                    chunks,
                    projection,
                    target_quality,
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

    def _projection_sources(
        self,
        projection_messages: Sequence[ModelMessage],
        incoming: Sequence[ModelMessage],
        incoming_sources: Sequence[TranscriptMessageRef | None],
    ) -> tuple[TranscriptMessageRef | None, ...]:
        if len(incoming) != len(incoming_sources):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        by_signature: dict[bytes, list[int]] = {}
        for index, message in enumerate(incoming):
            signature = _exact_message_signature(message)
            by_signature.setdefault(signature, []).append(index)
        used: set[int] = set()
        result: list[TranscriptMessageRef | None] = []
        for message in projection_messages:
            signature = _exact_message_signature(message)
            candidates = by_signature.get(signature, [])
            index = next((value for value in candidates if value not in used), None)
            if index is None:
                result.append(None)
                continue
            used.add(index)
            result.append(incoming_sources[index])
        return tuple(result)

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
    ) -> tuple[TranscriptChunk, ...]:
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
        return tuple(result)

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
        interactions: Sequence[ModelInteractionRecord] = (),
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
                interactions=interactions,
                execution_id=execution_id,
            )
        )

    async def sync_prepared_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[PreparedStepSnapshot],
        interactions: Sequence[ModelInteractionRecord] = (),
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
                interactions=interactions,
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
        interactions: Sequence[ModelInteractionRecord] = (),
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
        ) + tuple(
            ("interaction", interaction, interaction.status)
            for interaction in interactions
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
        grouped: dict[str, list[object]] = {
            "event": [],
            "snapshot": [],
            "interaction": [],
        }
        kinds: dict[str, list[str]] = {
            "event": [],
            "snapshot": [],
            "interaction": [],
        }
        for family, value, kind in facts:
            grouped[family].append(value)
            kinds[family].append(kind)
        stored_facts: list[StoredFact] = []
        reservation_requests = {
            self._sequence(run.run_id, family): len(grouped[family])
            for family in ("event", "snapshot", "interaction")
            if grouped[family]
        }
        high_waters = await transaction.reserve_sequences(reservation_requests)
        for family in ("event", "snapshot", "interaction"):
            values = grouped[family]
            if not values:
                continue
            sequence_key_value = self._sequence(run.run_id, family)
            final = high_waters[sequence_key_value]
            sequences = tuple(range(final - len(values) + 1, final + 1))
            stream = self._stream(run.run_id, family)
            fact_kind = {
                "event": "step_event",
                "snapshot": "step_snapshot",
                "interaction": "model_interaction",
            }[family]
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
        interactions: Sequence[ModelInteractionRecord] = (),
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
            interactions=interactions,
            execution_id=execution_id,
            history_head_guard=history_head_guard,
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

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        require_no_run_history_lock("StateStepArchive.list_model_interactions")
        _validate_interaction_page(after_request_sequence, limit)
        values = await self._facts(run_id, "interaction")
        result: list[object] = []
        for value in values:
            interaction = _decode_step(value.data)
            if not isinstance(interaction, ModelInteractionRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                after_request_sequence is not None
                and interaction.request_sequence <= after_request_sequence
            ):
                continue
            result.append(interaction)
            if limit is not None and len(result) >= limit:
                break
        return result

    async def iter_messages(self, *, run_id: str) -> AsyncIterator[object]:
        require_no_run_history_lock("StateStepArchive.iter_messages")
        async for message in self._history.iter_messages(run_id):
            yield message

    async def transcript_message_count(self, run_id: str) -> int:
        require_no_run_history_lock(
            "StateStepArchive.transcript_message_count"
        )
        return await self._history.transcript_message_count(run_id)

    def iter_message_range(
        self,
        *,
        run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._history.iter_message_range(
            run_id,
            start=start,
            end=end,
        )

    async def iter_raw_messages(self, *, run_id: str) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock("StateStepArchive.iter_raw_messages")
        async for message in self._history.iter_raw_messages(run_id):
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
        expected_messages = (
            snapshot.messages
            if snapshot.context_messages is None
            else snapshot.context_messages
        )
        return (
            stored.run_id == snapshot.run_id
            and stored.step_index == snapshot.step_index
            and stored.timestamp == snapshot.timestamp
            and stored.state == snapshot.state
            and stored.has_context_projection
            == (snapshot.context_messages is not None)
            and context.model_messages() == tuple(expected_messages)
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
        raw_messages = tuple(
            [message async for message in self._history.iter_raw_messages(run_id)]
        )
        if not raw_messages:
            raw_messages = tuple(messages)
        run = await self.get_run(run_id=run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        latest = ContinuableSnapshot(
            run_id=latest.run_id,
            step_index=latest.step_index,
            messages=list(raw_messages),
            conversation_id=run.conversation_id,
            parent_run_id=run.parent_run_id,
            agent_name=run.agent_name,
            timestamp=latest.timestamp,
            state=latest.state,
            context_messages=(
                list(messages) if latest.has_context_projection else None
            ),
        )
        return latest if include_interrupted or latest.state == "complete" else None

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
                    for family in ("event", "snapshot", "interaction")
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



def _validate_interaction_page(
    after_request_sequence: int | None,
    limit: int | None,
) -> None:
    if after_request_sequence is not None and (
        isinstance(after_request_sequence, bool)
        or not isinstance(after_request_sequence, int)
        or after_request_sequence < 0
    ):
        raise ValueError("interaction sequence must be non-negative")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("interaction limit must be positive")


async def _sync_projection(
    target: StepStore,
    run: RunRecord,
    events: tuple[StepEvent, ...],
    snapshots: tuple[ContinuableSnapshot, ...],
    interactions: tuple[ModelInteractionRecord, ...] = (),
    *,
    execution_id: str | None = None,
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.sync_projection(
            run,
            events=events,
            snapshots=snapshots,
            interactions=interactions,
            execution_id=execution_id,
        )
        return
    if await target.get_run(run_id=run.run_id) is None:
        await target.register_run(run, execution_id=execution_id)
    for event in events:
        await target.append_event(event, execution_id=execution_id)
    for snapshot in snapshots:
        await target.save_snapshot(snapshot, execution_id=execution_id)


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
    if isinstance(value, ModelInteractionRecord):
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "run_id": value.run_id,
                    "request_sequence": value.request_sequence,
                }
            )
        ).digest()
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
    "StagingStepStore",
    "StateStepArchive",
]
