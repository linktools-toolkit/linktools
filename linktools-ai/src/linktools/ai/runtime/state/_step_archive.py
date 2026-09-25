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
from .._message import decode_model_messages, encode_model_messages
from .._model_interaction import (
    StagedContextSpan,
    StagedModelInteraction,
    context_projection_to_durable,
)
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
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    ModelInteractionRecord,
    RuntimePayloadRef,
    StoredAgentRunCheckpoint,
    TranscriptChunk,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptSpanRef,
)
from ._history import (
    TranscriptCapture,
    TranscriptRepository,
    _exact_message_signature,
)
from ._plan import RuntimeDomain
from ._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
    StepEvent,
    AgentRunStore,
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
    record_key_digest,
    require_no_run_history_lock,
    scope_digest,
    sequence_key,
    sortable_timestamp,
    stream_digest,
)

_logger = environ.get_logger("ai.runtime.state.run_store")


@runtime_checkable
class _StepArchiveBatch(Protocol):
    async def sync_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None: ...

    async def materialize_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
    ) -> None: ...


@dataclass(slots=True)
class _ProjectionOffset:
    events: int = 0
    checkpoints: int = 0
    transcript_messages: int = 0
    interactions: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionProjectionBatch:
    run: AgentRunRecord
    events: tuple[StepEvent, ...]
    checkpoints: tuple[AgentRunCheckpoint, ...]
    base_event_offset: int
    base_checkpoint_offset: int
    target_event_offset: int
    target_checkpoint_offset: int
    base_message_index: int
    target_message_index: int
    interactions: tuple[StagedModelInteraction, ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0


@dataclass(frozen=True, slots=True)
class PreparedAgentRunCheckpoint:
    owner_id: str
    stored: StoredAgentRunCheckpoint
    chunks: tuple[TranscriptChunk, ...]
    projection: ContextProjection
    history_quality: HistoryQuality = HistoryQuality.COMPLETE


@dataclass(frozen=True, slots=True)
class PreparedAgentRunCheckpointBatch:
    agent_run_id: str
    checkpoints: tuple[PreparedAgentRunCheckpoint, ...]
    target_event_offset: int
    target_checkpoint_offset: int
    target_transcript_message_count: int

    def __iter__(self):
        return iter(self.checkpoints)

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __getitem__(self, index: int) -> PreparedAgentRunCheckpoint:
        return self.checkpoints[index]


@dataclass(frozen=True, slots=True)
class PreparedExecutionProjection:
    run: AgentRunRecord
    events: tuple[StepEvent, ...]
    checkpoints: tuple[PreparedAgentRunCheckpoint, ...]
    base_event_offset: int
    base_checkpoint_offset: int
    target_event_offset: int
    target_checkpoint_offset: int
    target_transcript_message_count: int
    durable_projection_digest: str = "empty"
    interactions: tuple["ModelInteractionRecord", ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0

    @property
    def projection_digest(self) -> str:
        if not self.checkpoints:
            return self.durable_projection_digest
        return self.checkpoints[-1].projection.digest


@dataclass(frozen=True, slots=True)
class ExecutionTerminalSealPlan:
    execution_id: str
    binding_digest: str
    projections: tuple[PreparedExecutionProjection, ...]
    seal_tokens: tuple[tuple[str, str], ...]
    terminal_attempt_token: str = ""

    def token_for(self, agent_run_id: str) -> str:
        for candidate, token in self.seal_tokens:
            if candidate == agent_run_id:
                return token
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class _AgentRunDurabilityKind(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    PROJECTION = "projection"
    CHECKPOINT = "checkpoint"
    RECOVERY_MATERIALIZATION = "recovery_materialization"
    RELEASE = "release"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class _AgentRunDurabilityFlight:
    """Registered in-flight durability work for one run.

    ``completion`` resolves only after the durable outcome is final; waiters
    must exit the run lock before awaiting it.
    """

    agent_run_id: str
    token: str
    kind: _AgentRunDurabilityKind
    completion: "asyncio.Future[None]"


_AgentRunProjectionFlight = _AgentRunDurabilityFlight


@dataclass(frozen=True, slots=True)
class CapturedExecutionProjection:
    """Immutable capture of one run's staged projection state."""

    run: AgentRunRecord
    events: tuple[StepEvent, ...]
    checkpoints: tuple[AgentRunCheckpoint, ...]
    base_event_offset: int
    base_checkpoint_offset: int
    target_event_offset: int
    target_checkpoint_offset: int
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
    agent_run_id: str
    task: asyncio.Task[object]


class LockOrderError(RuntimeError):
    """Raised when one task attempts to hold more than one run lock."""


_held_run_history_locks: ContextVar[tuple[_HeldRunHistoryLock, ...]] = ContextVar(
    "linktools_ai_held_run_history_locks",
    default=(),
)


class _AgentRunHistoryLock:
    def __init__(self) -> None:
        self._entries: dict[str, _ProjectionLockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, agent_run_id: str):
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
        if any(value.agent_run_id != agent_run_id for value in held):
            raise LockOrderError(
                "one asyncio task cannot hold multiple RunHistoryLocks"
            )
        async with self._guard:
            entry = self._entries.get(agent_run_id)
            if entry is None:
                entry = _ProjectionLockEntry(asyncio.Lock())
                self._entries[agent_run_id] = entry
            entry.references += 1
            if held and entry.owner is not current_task:
                entry.references -= 1
                if entry.references == 0 and self._entries.get(agent_run_id) is entry:
                    self._entries.pop(agent_run_id, None)
                raise LockOrderError("run history lock ownership is inconsistent")
            if entry.owner is current_task:
                entry.depth += 1
                nested = True
            else:
                nested = False
        if nested:
            lock_token = enter_run_history_lock(agent_run_id)
            try:
                yield
            finally:
                exit_run_history_lock(lock_token)
                async with self._guard:
                    entry.depth -= 1
                    entry.references -= 1
                    if entry.references == 0 and self._entries.get(agent_run_id) is entry:
                        self._entries.pop(agent_run_id, None)
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
                held + (_HeldRunHistoryLock(agent_run_id, current_task),)
            )
            lock_token = enter_run_history_lock(agent_run_id)
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
                if entry.references == 0 and self._entries.get(agent_run_id) is entry:
                    self._entries.pop(agent_run_id, None)


class StagingAgentRunStore(AgentRunStore):
    """Process-local facts collected before owner materialization."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRunRecord] = {}
        self._events: dict[str, list[StepEvent]] = {}
        self._checkpoints: dict[str, list[AgentRunCheckpoint]] = {}
        self._interactions: dict[str, list[StagedModelInteraction]] = {}
        self._payloads: dict[str, dict[str, bytes]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def initialize(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def register_agent_run(
        self,
        record: AgentRunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            self.register_agent_run_local(record)

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        self._ensure_open()
        return self.get_agent_run_local(agent_run_id)

    async def list_agent_runs(
        self, *, parent_agent_run_id: str | None = None, agent_conversation_id: str | None = None
    ) -> list[AgentRunRecord]:
        self._ensure_open()
        values = [
            value
            for value in self._runs.values()
            if (parent_agent_run_id is None or value.parent_agent_run_id == parent_agent_run_id)
            and (agent_conversation_id is None or value.agent_conversation_id == agent_conversation_id)
        ]
        return sorted(values, key=lambda value: (value.started_at, value.agent_run_id))

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

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        self._ensure_open()
        return self.list_events_local(agent_run_id)

    async def list_checkpoints(self, *, agent_run_id: str) -> list[AgentRunCheckpoint]:
        self._ensure_open()
        return self.list_checkpoints_local(agent_run_id)

    async def save_checkpoint(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            self.save_checkpoint_local(checkpoint)

    async def latest_checkpoint(
        self,
        *,
        agent_run_id: str,
        include_interrupted: bool = False,
    ) -> AgentRunCheckpoint | None:
        self._ensure_open()
        return self.latest_checkpoint_local(
            agent_run_id,
            include_interrupted=include_interrupted,
        )

    def intern_payload(self, agent_run_id: str, payload: bytes) -> tuple[str, int]:
        self._ensure_open()
        digest = hashlib.sha256(payload).hexdigest()
        self._payloads.setdefault(agent_run_id, {}).setdefault(digest, bytes(payload))
        return digest, len(payload)

    def staged_payload(self, agent_run_id: str, digest: str) -> bytes:
        self._ensure_open()
        try:
            return self._payloads[agent_run_id][digest]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def stage_model_interaction(self, interaction: object) -> None:
        self._ensure_open()
        if not isinstance(interaction, StagedModelInteraction):
            raise TypeError("staged model interaction is invalid")
        self._interactions.setdefault(interaction.agent_run_id, []).append(interaction)

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        self._ensure_open()
        _validate_interaction_page(after_request_sequence, limit)
        selected: list[object] = []
        for interaction in self._interactions.get(agent_run_id, ()):
            if (
                after_request_sequence is not None
                and interaction.request_sequence <= after_request_sequence
            ):
                continue
            selected.append(interaction)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        self._ensure_open()
        values = self._interactions.get(agent_run_id, ())
        if not values:
            return 0
        sequences = tuple(value.request_sequence for value in values)
        if sequences != tuple(range(1, len(values) + 1)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return len(values)

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
        checkpoint = await self.latest_checkpoint(
            agent_run_id=owner_id,
            include_interrupted=True,
        )
        if checkpoint is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        messages = (
            checkpoint.messages
            if checkpoint.context_messages is None
            else checkpoint.context_messages
        )
        return LoadedModelContext(
            tuple(LoadedContextMessage(message, None) for message in messages)
        )

    async def release_agent_run(
        self,
        agent_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self.release_agent_run_local(agent_run_id)

    def register_agent_run_local(self, record: AgentRunRecord) -> None:
        self._ensure_open()
        previous = self._runs.get(record.agent_run_id)
        if previous is not None and previous != record:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._runs[record.agent_run_id] = record

    def get_agent_run_local(self, agent_run_id: str) -> AgentRunRecord | None:
        self._ensure_open()
        return self._runs.get(agent_run_id)

    def append_event_local(self, event: StepEvent) -> None:
        self._ensure_open()
        if event.agent_run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        values = self._events.setdefault(event.agent_run_id, [])
        if event not in values:
            values.append(event)

    def list_events_local(self, agent_run_id: str) -> list[StepEvent]:
        self._ensure_open()
        return list(self._events.get(agent_run_id, ()))

    def list_checkpoints_local(self, agent_run_id: str) -> list[AgentRunCheckpoint]:
        self._ensure_open()
        return list(self._checkpoints.get(agent_run_id, ()))

    def save_checkpoint_local(self, checkpoint: AgentRunCheckpoint) -> None:
        self._ensure_open()
        if checkpoint.agent_run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        values = self._checkpoints.setdefault(checkpoint.agent_run_id, [])
        if checkpoint not in values:
            values.append(checkpoint)

    def latest_checkpoint_local(
        self,
        agent_run_id: str,
        *,
        include_interrupted: bool = False,
    ) -> AgentRunCheckpoint | None:
        self._ensure_open()
        values = self._checkpoints.get(agent_run_id, ())
        if not values:
            return None
        latest = values[-1]
        return latest if include_interrupted or latest.state == "complete" else None

    def release_agent_run_local(self, agent_run_id: str) -> None:
        self._ensure_open()
        self._runs.pop(agent_run_id, None)
        self._events.pop(agent_run_id, None)
        self._checkpoints.pop(agent_run_id, None)
        self._interactions.pop(agent_run_id, None)
        self._payloads.pop(agent_run_id, None)

    def capture_projection_local(
        self,
        agent_run_id: str,
        offset: _ProjectionOffset,
    ) -> ExecutionProjectionBatch | None:
        self._ensure_open()
        run = self._runs.get(agent_run_id)
        if run is None:
            return None
        events = self._events.get(agent_run_id, ())
        checkpoints = self._checkpoints.get(agent_run_id, ())
        interactions = self._interactions.get(agent_run_id, ())
        return ExecutionProjectionBatch(
            run,
            tuple(events[offset.events:]),
            tuple(checkpoints[offset.checkpoints:]),
            offset.events,
            offset.checkpoints,
            len(events),
            len(checkpoints),
            offset.transcript_messages,
            offset.transcript_messages,
            tuple(interactions[offset.interactions:]),
            offset.interactions,
            len(interactions),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)


class InMemoryStepArchive(StagingAgentRunStore):
    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        super().__init__()
        self._runtime_domain = runtime_domain

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    async def sync_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None:
        del execution_id
        self._ensure_open()
        async with self._lock:
            previous = self._runs.get(run.agent_run_id)
            if previous is not None and previous != run:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._runs[run.agent_run_id] = run
            event_values = self._events.setdefault(run.agent_run_id, [])
            checkpoint_values = self._checkpoints.setdefault(run.agent_run_id, [])
            for event in events:
                if event not in event_values:
                    event_values.append(event)
            for checkpoint in checkpoints:
                if checkpoint not in checkpoint_values:
                    checkpoint_values.append(checkpoint)
            interaction_values = self._interactions.setdefault(run.agent_run_id, [])
            for interaction in interactions:
                if interaction not in interaction_values:
                    interaction_values.append(interaction)

    async def materialize_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self.sync_projection(
            run,
            events=(),
            checkpoints=(checkpoint,),
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

    async def iter_messages(self, *, agent_run_id: str) -> AsyncIterator[object]:
        checkpoint = await self.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        if checkpoint is not None:
            for message in checkpoint.messages:
                yield message

    async def transcript_message_count(self, agent_run_id: str) -> int:
        checkpoint = await self.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        return 0 if checkpoint is None else len(checkpoint.messages)

    async def iter_message_range(
        self,
        *,
        agent_run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        checkpoint = await self.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        total = 0 if checkpoint is None else len(checkpoint.messages)
        if end > total:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if checkpoint is not None:
            for message in checkpoint.messages[start:end]:
                yield message

    async def load_model_context(self, *, agent_run_id: str) -> tuple[object, ...]:
        checkpoint = await self.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        return () if checkpoint is None else tuple(checkpoint.messages)

    async def prepare_relocated_interactions(
        self,
        interactions: Sequence[ModelInteractionRecord],
        resolved: Sequence[
            tuple[
                tuple[ModelMessage, ...],
                tuple[ModelMessage, ...] | None,
                bytes,
            ]
        ],
    ) -> tuple[ModelInteractionRecord, ...]:
        if len(interactions) != len(resolved):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        def inline_context(messages: Sequence[ModelMessage]) -> ContextProjection:
            return ContextProjection(
                (
                    InlineContextBlock(
                        RuntimePayloadRef(
                            StoredPayload.inline_bytes(
                                encode_model_messages(tuple(messages))
                            ),
                            self._runtime_domain,
                        )
                    ),
                )
            )

        values: list[ModelInteractionRecord] = []
        for interaction, (request, response, envelope) in zip(
            interactions,
            resolved,
            strict=True,
        ):
            values.append(
                replace(
                    interaction,
                    request_context=inline_context(request),
                    request_envelope=RuntimePayloadRef(
                        StoredPayload.inline_bytes(envelope),
                        self._runtime_domain,
                    ),
                    response_context=(
                        None if response is None else inline_context(response)
                    ),
                )
            )
        return tuple(values)

    async def resolve_model_interaction(
        self,
        interaction: object,
    ) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...] | None, bytes]:
        if not isinstance(interaction, ModelInteractionRecord):
            raise TypeError("model interaction is invalid")
        checkpoint = await self.latest_checkpoint(
            agent_run_id=interaction.agent_run_id,
            include_interrupted=True,
        )
        if checkpoint is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        def resolve(projection: ContextProjection) -> tuple[ModelMessage, ...]:
            values: list[ModelMessage] = []
            for item in projection.items:
                if isinstance(item, TranscriptSpanRef):
                    if item.start < 0 or item.end > len(checkpoint.messages):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    values.extend(checkpoint.messages[item.start : item.end])
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


class StateStepArchive(AgentRunStore):
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
        self._history_lock = _AgentRunHistoryLock()
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
        agent_run_id: str,
    ) -> tuple[int, int, int, str]:
        values = await self.execution_history_heads((agent_run_id,))
        head = values.get(agent_run_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return (
            head.event_count,
            head.checkpoint_count,
            head.transcript_message_count,
            head.projection_digest,
        )

    async def execution_history_head_record(
        self,
        agent_run_id: str,
    ) -> ExecutionRunSealHead:
        values = await self.execution_history_heads((agent_run_id,))
        head = values.get(agent_run_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return head

    async def execution_history_heads(
        self,
        agent_run_ids: Sequence[str],
    ) -> Mapping[str, ExecutionRunSealHead]:
        require_no_run_history_lock("StateStepArchive.execution_history_heads")
        if self._runtime_domain is not RuntimeDomain.EXECUTION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        unique_agent_run_ids = tuple(dict.fromkeys(agent_run_ids))
        if not unique_agent_run_ids:
            return {}
        sequence_keys = tuple(
            key
            for agent_run_id in unique_agent_run_ids
            for key in (
                self._sequence(agent_run_id, "event"),
                self._sequence(agent_run_id, "checkpoint"),
                self._sequence(agent_run_id, "interaction"),
            )
        )
        run_keys = tuple(self._agent_run_key(agent_run_id) for agent_run_id in unique_agent_run_ids)
        projection_keys = tuple(
            self._history.projection_key(agent_run_id) for agent_run_id in unique_agent_run_ids
        )
        head_keys = tuple(self._history.head_key(agent_run_id) for agent_run_id in unique_agent_run_ids)
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
        for agent_run_id in unique_agent_run_ids:
            agent_run_record = records.get(self._agent_run_key(agent_run_id))
            head_record = records.get(self._history.head_key(agent_run_id))
            projection_record = records.get(self._history.projection_key(agent_run_id))
            event_count = sequences.get(self._sequence(agent_run_id, "event"), 0)
            checkpoint_count = sequences.get(self._sequence(agent_run_id, "checkpoint"), 0)
            interaction_count = sequences.get(
                self._sequence(agent_run_id, "interaction"), 0
            )
            if head_record is None:
                if (
                    agent_run_record is not None
                    or projection_record is not None
                    or event_count
                    or checkpoint_count
                    or interaction_count
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result[agent_run_id] = ExecutionRunSealHead(agent_run_id, 0, 0, 0, "empty")
                continue
            if agent_run_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = self._history.decode_head(head_record)
            projection_digest = "empty"
            if projection_record is not None:
                projection = _decode_enveloped_domain(
                    projection_record.data,
                    ContextProjection,
                )
                projection_digest = projection.digest
            result[agent_run_id] = ExecutionRunSealHead(
                agent_run_id,
                event_count,
                checkpoint_count,
                head.message_count,
                projection_digest,
                interaction_count,
            )
        return result

    async def prepare_relocated_interactions(
        self,
        interactions: Sequence[ModelInteractionRecord],
        resolved: Sequence[
            tuple[
                tuple[ModelMessage, ...],
                tuple[ModelMessage, ...] | None,
                bytes,
            ]
        ],
    ) -> tuple[ModelInteractionRecord, ...]:
        if len(interactions) != len(resolved):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def inline_context(
            agent_run_id: str,
            messages: Sequence[ModelMessage],
        ) -> ContextProjection:
            projection = ContextProjection(
                (
                    InlineContextBlock(
                        RuntimePayloadRef(
                            StoredPayload.inline_bytes(
                                encode_model_messages(tuple(messages))
                            ),
                            self._runtime_domain,
                        )
                    ),
                )
            )
            return await self._history.prepare_projection(agent_run_id, projection)

        values: list[ModelInteractionRecord] = []
        for interaction, (request, response, envelope) in zip(
            interactions,
            resolved,
            strict=True,
        ):
            values.append(
                replace(
                    interaction,
                    request_context=await inline_context(
                        interaction.agent_run_id,
                        request,
                    ),
                    request_envelope=await self._prepare_inline_payload(
                        interaction.agent_run_id,
                        RuntimePayloadRef(
                            StoredPayload.inline_bytes(envelope),
                            self._runtime_domain,
                        ),
                    ),
                    response_context=(
                        None
                        if response is None
                        else await inline_context(interaction.agent_run_id, response)
                    ),
                )
            )
        return tuple(values)

    async def resolve_model_interaction(
        self,
        interaction: object,
    ) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...] | None, bytes]:
        if not isinstance(interaction, ModelInteractionRecord):
            raise TypeError("model interaction is invalid")
        request = await self._history.load_projected_context(
            interaction.agent_run_id,
            interaction.request_context,
        )
        response = None
        if interaction.response_context is not None:
            response = (
                await self._history.load_projected_context(
                    interaction.agent_run_id,
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
            records[0].agent_run_id,
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
        run: AgentRunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        *,
        local_message_base: int = 0,
        local_message_count: int = 0,
    ) -> tuple[ModelInteractionRecord, ...]:
        values = tuple(interactions)
        if not values:
            return ()
        if local_message_base < 0 or local_message_count < 0:
            raise ValueError("local interaction transcript range is invalid")
        request_sequences: set[int] = set()
        external_refs: list[TranscriptMessageRef] = []
        for interaction in values:
            if (
                interaction.agent_run_id != run.agent_run_id
                or interaction.request_sequence in request_sequences
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request_sequences.add(interaction.request_sequence)
            for projection in (
                interaction.request_context,
                *(
                    ()
                    if interaction.response_context is None
                    else (interaction.response_context,)
                ),
            ):
                for item in projection.items:
                    if isinstance(item, StagedContextSpan):
                        if item.end > local_message_count:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    elif isinstance(item, TranscriptSpanRef):
                        external_refs.extend(
                            TranscriptMessageRef(
                                item.source_domain,
                                item.owner_id,
                                index,
                            )
                            for index in range(item.start, item.end)
                        )
        if external_refs:
            await self._history.resolve_transcript_message_refs(tuple(external_refs))

        result: list[ModelInteractionRecord] = []
        for staged in values:
            request_context = context_projection_to_durable(
                staged.request_context,
                owner_id=run.agent_run_id,
                source_domain=self._runtime_domain,
                payload=payload,
                local_message_base=local_message_base,
            )
            request_context = await self._history.prepare_projection(
                run.agent_run_id,
                request_context,
            )
            request_envelope = await self._prepare_inline_payload(
                run.agent_run_id,
                RuntimePayloadRef(
                    StoredPayload.inline_bytes(payload(staged.request_envelope_digest)),
                    self._runtime_domain,
                ),
            )
            response_context = None
            if staged.response_context is not None:
                response_context = context_projection_to_durable(
                    staged.response_context,
                    owner_id=run.agent_run_id,
                    source_domain=self._runtime_domain,
                    payload=payload,
                    local_message_base=local_message_base,
                )
                response_context = await self._history.prepare_projection(
                    run.agent_run_id,
                    response_context,
                )
            result.append(
                ModelInteractionRecord(
                    staged.agent_run_id,
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
                    staged.attachments,
                )
            )
        return tuple(result)

    async def _prepare_inline_payload(
        self,
        agent_run_id: str,
        content: RuntimePayloadRef,
    ) -> RuntimePayloadRef:
        projection = await self._history.prepare_projection(
            agent_run_id,
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
        if await self.get_agent_run(agent_run_id=projection.run.agent_run_id) != projection.run:
            return False
        head = await self.execution_history_head(projection.run.agent_run_id)
        record = await self.execution_history_head_record(projection.run.agent_run_id)
        return head == (
            projection.target_event_offset,
            projection.target_checkpoint_offset,
            projection.target_transcript_message_count,
            projection.projection_digest,
        ) and record.interaction_count == projection.target_interaction_offset

    def bind_history_lock(self, history_lock: _AgentRunHistoryLock) -> None:
        self._history_lock = history_lock

    def register_context_baseline(
        self,
        agent_run_id: str,
        context: LoadedModelContext,
    ) -> None:
        self._context_baselines[agent_run_id] = context

    async def transcript_message_count_for_run(
        self,
        run: AgentRunRecord,
    ) -> int:
        require_no_run_history_lock(
            "StateStepArchive.transcript_message_count_for_run"
        )
        owner_id = (
            self._history_id(run)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else run.agent_run_id
        )
        return await self._history.transcript_message_count(owner_id)

    async def relocate_conversation_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
    ) -> AgentRunCheckpoint:
        """Rebase one cumulative run checkpoint onto the conversation owner."""
        self._ensure_open()
        require_no_run_history_lock(
            "StateStepArchive.relocate_conversation_checkpoint"
        )
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing_run = await self.get_agent_run(agent_run_id=run.agent_run_id)
        before = 0
        if existing_run is not None:
            if existing_run != run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            observed = await self.latest_checkpoint(
                agent_run_id=run.agent_run_id,
                include_interrupted=True,
            )
            if not _conversation_relocated_checkpoint_matches(checkpoint, observed):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            before = len(checkpoint.messages)
        return replace(
            checkpoint,
            transcript_message_count_before=before,
        )

    async def relocate_run_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
    ) -> AgentRunCheckpoint:
        """Rebase one cumulative checkpoint onto its run-owned archive."""
        self._ensure_open()
        require_no_run_history_lock(
            "StateStepArchive.relocate_run_checkpoint"
        )
        if self._runtime_domain is RuntimeDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing_run = await self.get_agent_run(agent_run_id=run.agent_run_id)
        if existing_run is None:
            return replace(checkpoint, transcript_message_count_before=0)
        if existing_run != run:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        before = await self.transcript_message_count_for_run(run)
        source_messages = tuple(checkpoint.messages)
        if before > len(source_messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if before:
            observed = await self._history.load_message_span(
                run.agent_run_id,
                0,
                before,
            )
            if tuple(
                _exact_message_signature(message) for message in observed
            ) != tuple(
                _exact_message_signature(message)
                for message in source_messages[:before]
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replace(
            checkpoint,
            transcript_message_count_before=before,
        )

    async def prepare_checkpoints(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
    ) -> PreparedAgentRunCheckpointBatch:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.prepare_checkpoints")
        return await self._prepare_checkpoints(
            run,
            checkpoints,
        )

    async def prepare_checkpoints_after_seal(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
    ) -> PreparedAgentRunCheckpointBatch:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.prepare_checkpoints_after_seal")
        if active_state_scope() is not None or _held_run_history_locks.get():
            raise LockOrderError(
                "sealed checkpoint preparation requires no StateStore or run lock"
            )
        return await self._prepare_checkpoints(
            run,
            checkpoints,
        )

    async def initialize(self) -> None:
        self._closed = False
        self._context_baselines.clear()

    async def close(self) -> None:
        self._closed = True
        self._context_baselines.clear()

    def _agent_run_key(self, agent_run_id: str) -> bytes:
        return record_key_digest(self._namespace, self._tenant_id, self._runtime_domain.value, "agent_run", agent_run_id)

    def _stream(self, agent_run_id: str, family: str) -> bytes:
        return stream_digest(self._namespace, self._tenant_id, self._runtime_domain.value, family, agent_run_id)

    def _sequence(self, agent_run_id: str, family: str) -> bytes:
        return sequence_key(self._namespace, self._tenant_id, self._runtime_domain.value, family, agent_run_id)

    async def register_agent_run(
        self,
        record: AgentRunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.register_agent_run")

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

    def _stored_agent_run(self, record: AgentRunRecord) -> StoredRecord:
        return StoredRecord(
            self._agent_run_key(record.agent_run_id),
            None
            if record.agent_conversation_id is None
            else scope_digest(
                self._namespace,
                self._tenant_id,
                self._runtime_domain.value,
                "agent_run",
                "conversation",
                record.agent_conversation_id,
            ),
            None
            if record.parent_agent_run_id is None
            else parent_digest(
                self._namespace,
                self._tenant_id,
                self._runtime_domain.value,
                "agent_run",
                "parent",
                record.parent_agent_run_id,
            ),
            "agent_run",
            sortable_timestamp(record.started_at, record.agent_run_id),
            None,
            0,
            None,
            0,
            None,
            _encode_step(record),
        )

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        require_no_run_history_lock("StateStepArchive.get_agent_run")
        stored = await self._store.read(lambda transaction: transaction.get_record(self._agent_run_key(agent_run_id)))
        return None if stored is None else _decode_step(stored.data)

    async def list_agent_runs(
        self, *, parent_agent_run_id: str | None = None, agent_conversation_id: str | None = None
    ) -> list[AgentRunRecord]:
        require_no_run_history_lock("StateStepArchive.list_agent_runs")
        if parent_agent_run_id is not None:
            query = RecordQuery(
                kind="agent_run",
                parent_digest=parent_digest(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "agent_run",
                    "parent",
                    parent_agent_run_id,
                )
            )
        elif agent_conversation_id is not None:
            query = RecordQuery(
                kind="agent_run",
                scope_digest=scope_digest(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "agent_run",
                    "conversation",
                    agent_conversation_id,
                )
            )
        else:
            query = RecordQuery(
                kind="agent_run",
            )
        records = await self._store.read(lambda transaction: transaction.list_records(query))
        values = [_decode_step(record.data) for record in records]
        return [value for value in values if isinstance(value, AgentRunRecord)]

    async def _prepare_checkpoints(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
    ) -> PreparedAgentRunCheckpointBatch:
        values = tuple(checkpoints)
        if not values:
            head_owner = (
                self._history_id(run)
                if self._runtime_domain is RuntimeDomain.CONVERSATION
                else run.agent_run_id
            )
            head = await self._history.get_head(head_owner)
            return PreparedAgentRunCheckpointBatch(
                run.agent_run_id,
                (),
                0,
                0,
                0 if head is None else head.message_count,
            )
        explicit = tuple(
            checkpoint.transcript_message_count_before is not None
            for checkpoint in values
        )
        if all(explicit):
            return await self._prepare_explicit_checkpoints(run, values)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _prepare_explicit_checkpoints(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
    ) -> PreparedAgentRunCheckpointBatch:
        owner_id = (
            self._history_id(run)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else run.agent_run_id
        )
        head = await self._history.get_head(owner_id)
        if head is None:
            if await self.get_agent_run(agent_run_id=run.agent_run_id) is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = self._history.empty_head(owner_id)

        first_before = checkpoints[0].transcript_message_count_before
        if first_before is None or first_before > head.message_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive_base = head.message_count - first_before
        target_message_count = head.message_count
        baseline = self._context_baselines.get(run.agent_run_id, LoadedModelContext(()))
        baseline_messages = baseline.model_messages()
        baseline_sources = tuple(
            self._reusable_context_source(value.source)
            for value in baseline.messages
        )
        prepared: list[PreparedAgentRunCheckpoint] = []

        for checkpoint in checkpoints:
            before = checkpoint.transcript_message_count_before
            incoming = tuple(checkpoint.messages)
            if (
                before is None
                or before > len(incoming)
                or archive_base + before != target_message_count
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            delta = incoming[before:]
            chunks = await self._prepare_captured_chunks(
                owner_id,
                TranscriptCapture(
                    target_message_count,
                    delta,
                    (TranscriptOrigin.RAW,) * len(delta),
                    head.quality,
                ),
            )
            raw_sources = tuple(
                TranscriptMessageRef(
                    self._runtime_domain,
                    owner_id,
                    archive_base + index,
                )
                for index in range(len(incoming))
            )
            source_messages = (*baseline_messages, *incoming)
            source_refs = (*baseline_sources, *raw_sources)
            projection_messages = (
                source_messages
                if checkpoint.context_messages is None
                else tuple(checkpoint.context_messages)
            )
            projection_sources = self._projection_sources(
                projection_messages,
                source_messages,
                source_refs,
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
                PreparedAgentRunCheckpoint(
                    owner_id,
                    StoredAgentRunCheckpoint(
                        run.agent_run_id,
                        checkpoint.step_index,
                        checkpoint.timestamp,
                        checkpoint.state,
                        projection.digest,
                        checkpoint.context_messages is not None,
                        checkpoint.pending_request_index,
                    ),
                    chunks,
                    projection,
                    head.quality,
                )
            )
            target_message_count += len(delta)

        return PreparedAgentRunCheckpointBatch(
            run.agent_run_id,
            tuple(prepared),
            0,
            0,
            target_message_count,
        )

    def _reusable_context_source(
        self,
        source: TranscriptMessageRef | None,
    ) -> TranscriptMessageRef | None:
        if source is None:
            return None
        if source.source_domain is self._runtime_domain:
            return source
        if source.source_domain is RuntimeDomain.CONVERSATION:
            return source
        return None

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

    def _history_id(self, run: AgentRunRecord) -> str:
        history_id = run.metadata.get("history_id")
        return history_id or run.agent_run_id

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

    async def _normalize_checkpoints_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
    ) -> tuple[PreparedAgentRunCheckpoint, ...]:
        del transaction, run
        values = tuple(checkpoints)
        if any(not isinstance(checkpoint, PreparedAgentRunCheckpoint) for checkpoint in values):
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
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.sync_projection")
        prepared = await self._prepare_checkpoints(
            run,
            checkpoints,
        )
        await self._store.mutate(
            lambda transaction: self._sync_projection_in_transaction(
                transaction,
                run,
                events=events,
                checkpoints=prepared.checkpoints,
                interactions=interactions,
                execution_id=execution_id,
            )
        )

    async def sync_prepared_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None:
        """Commit a prepared projection without preparing its payload twice."""
        self._ensure_open()
        require_no_run_history_lock(
            "StateStepArchive.sync_prepared_projection"
        )
        values = tuple(checkpoints)
        if any(not isinstance(value, PreparedAgentRunCheckpoint) for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.mutate(
            lambda transaction: self._sync_projection_in_transaction(
                transaction,
                run,
                events=events,
                checkpoints=values,
                interactions=interactions,
                execution_id=execution_id,
            )
        )

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        self._ensure_open()
        checkpoints = await self._normalize_checkpoints_in_transaction(
            transaction,
            run,
            checkpoints,
        )
        facts = tuple(
            ("event", event, _step_event_kind(event)) for event in events
        ) + tuple(
            ("checkpoint", checkpoint.stored, checkpoint.stored.state)
            for checkpoint in checkpoints
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
        owner = self._agent_run_key(run.agent_run_id)
        owner_record = await self._ensure_run_in_transaction(transaction, run)
        grouped: dict[str, list[object]] = {
            "event": [],
            "checkpoint": [],
            "interaction": [],
        }
        kinds: dict[str, list[str]] = {
            "event": [],
            "checkpoint": [],
            "interaction": [],
        }
        for family, value, kind in facts:
            grouped[family].append(value)
            kinds[family].append(kind)
        stored_facts: list[StoredFact] = []
        reservation_requests = {
            self._sequence(run.agent_run_id, family): len(grouped[family])
            for family in ("event", "checkpoint", "interaction")
            if grouped[family]
        }
        high_waters = await transaction.reserve_sequences(reservation_requests)
        for family in ("event", "checkpoint", "interaction"):
            values = grouped[family]
            if not values:
                continue
            sequence_key_value = self._sequence(run.agent_run_id, family)
            final = high_waters[sequence_key_value]
            sequences = tuple(range(final - len(values) + 1, final + 1))
            stream = self._stream(run.agent_run_id, family)
            fact_kind = {
                "event": "step_event",
                "checkpoint": "step_checkpoint",
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
        if checkpoints:
            await self._history.append_chunks(
                transaction,
                checkpoints[0].owner_id,
                tuple(
                    chunk
                    for checkpoint in checkpoints
                    for chunk in checkpoint.chunks
                ),
                min(
                    (checkpoint.history_quality for checkpoint in checkpoints),
                    key=lambda value: value is HistoryQuality.COMPLETE,
                    default=HistoryQuality.COMPLETE,
                ),
            )
            await self._history.store_projection(
                transaction,
                checkpoints[-1].owner_id,
                checkpoints[-1].projection,
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

    async def materialize_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        interactions: Sequence[ModelInteractionRecord] = (),
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.materialize_checkpoint")
        prepared = await self._prepare_checkpoints(
            run,
            (checkpoint,),
        )
        async def mutate(transaction: StateTransaction) -> None:
            await self._materialize_checkpoint_in_transaction(
                transaction,
                run,
                prepared.checkpoints[0],
                execution_id=execution_id,
            )
            if interactions:
                await self._sync_projection_in_transaction(
                    transaction,
                    run,
                    events=(),
                    checkpoints=(),
                    interactions=interactions,
                    execution_id=execution_id,
                )

        await self._store.mutate(mutate)

    async def _materialize_checkpoint_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        checkpoint: PreparedAgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        if not isinstance(checkpoint, PreparedAgentRunCheckpoint):
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
            "checkpoint",
            checkpoint.stored,
            checkpoint.stored.state,
        ):
            return
        await self._ensure_run_in_transaction(transaction, run)
        await self._history.append_chunks(
            transaction,
            checkpoint.owner_id,
            checkpoint.chunks,
            checkpoint.history_quality,
        )
        await self._history.store_projection(
            transaction,
            checkpoint.owner_id,
            checkpoint.projection,
        )
        await self._materialize_fact_in_transaction(
            transaction,
            run,
            "checkpoint",
            checkpoint.stored,
            checkpoint.stored.state,
        )
        if history_head_guard is not None and not supplied_history_head_guard:
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

    async def materialize_checkpoint_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        checkpoint: PreparedAgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        require_no_run_history_lock(
            "StateStepArchive.materialize_checkpoint_in_transaction"
        )
        await self._materialize_checkpoint_in_transaction(
            transaction,
            run,
            checkpoint,
            execution_id=execution_id,
            history_head_guard=history_head_guard,
        )

    async def sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
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
            checkpoints=checkpoints,
            interactions=interactions,
            execution_id=execution_id,
            history_head_guard=history_head_guard,
        )

    async def _materialize_fact_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        family: str,
        value: object,
        kind: str,
    ) -> None:
        stream = self._stream(run.agent_run_id, family)
        owner = self._agent_run_key(run.agent_run_id)
        subject = _step_subject(value)
        fact_kind = {
            "checkpoint": "step_checkpoint",
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
        sequence = (await _reserve_sequences(transaction, self._sequence(run.agent_run_id, family), 1))[0]
        await _insert_facts(
            transaction,
            (StoredFact(stream, sequence, owner, fact_kind, subject, kind, data),),
        )

    async def _ensure_run_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
    ) -> StoredRecord:
        owner_record, _created = await self._ensure_run_with_head_in_transaction(
            transaction,
            run,
        )
        return owner_record

    async def _ensure_run_with_head_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
    ) -> tuple[StoredRecord, bool]:
        owner = self._agent_run_key(run.agent_run_id)
        history_owner = (
            self._history_id(run)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else run.agent_run_id
        )
        head_key = self._history.head_key(history_owner)
        records = await transaction.get_records((owner, head_key))
        owner_record = records.get(owner)
        head_record = records.get(head_key)
        if owner_record is None:
            stored_run = self._stored_agent_run(run)
            if head_record is None:
                await transaction.insert_records(
                    (
                        stored_run,
                        self._history.empty_head_record(history_owner),
                    )
                )
                _logger.debug(
                    "AgentRun admitted with transcript head: agent_run=%s",
                    run.agent_run_id,
                )
            else:
                self._history.decode_head(head_record)
                await transaction.insert_records((stored_run,))
                _logger.debug(
                    "AgentRun admitted using existing transcript head: agent_run=%s",
                    run.agent_run_id,
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
        run: AgentRunRecord,
        family: str,
        value: object,
        kind: str,
    ) -> bool:
        stream = self._stream(run.agent_run_id, family)
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
            event.agent_run_id,
            "event",
            event,
            _step_event_kind(event),
            execution_id=execution_id,
        )

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        require_no_run_history_lock("StateStepArchive.list_events")
        values = await self._facts(agent_run_id, "event")
        return [_decode_step(value.data) for value in values]

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        require_no_run_history_lock("StateStepArchive.list_model_interactions")
        _validate_interaction_page(after_request_sequence, limit)
        values = await self._facts(agent_run_id, "interaction")
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

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        require_no_run_history_lock(
            "StateStepArchive.model_interaction_count"
        )
        values = await self._facts(agent_run_id, "interaction", latest=True)
        if not values:
            return 0
        if len(values) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fact = values[0]
        interaction = _decode_step(fact.data)
        if (
            not isinstance(interaction, ModelInteractionRecord)
            or fact.sequence != interaction.request_sequence
            or fact.sequence < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return fact.sequence

    async def iter_messages(self, *, agent_run_id: str) -> AsyncIterator[object]:
        require_no_run_history_lock("StateStepArchive.iter_messages")
        async for message in self._history.iter_messages(agent_run_id):
            yield message

    async def transcript_message_count(self, agent_run_id: str) -> int:
        require_no_run_history_lock(
            "StateStepArchive.transcript_message_count"
        )
        return await self._history.transcript_message_count(agent_run_id)

    def iter_message_range(
        self,
        *,
        agent_run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._history.iter_message_range(
            agent_run_id,
            start=start,
            end=end,
        )

    async def iter_raw_messages(self, *, agent_run_id: str) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock("StateStepArchive.iter_raw_messages")
        async for message in self._history.iter_raw_messages(agent_run_id):
            yield message

    async def load_model_context(
        self,
        *,
        agent_run_id: str,
    ) -> tuple[object, ...]:
        require_no_run_history_lock("StateStepArchive.load_model_context")
        values = await self._history.load_model_context(
            agent_run_id,
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

    async def load_committed_session_model_context(
        self,
        history_id: str,
        *,
        agent_run_id: str,
        message_count: int,
        tenant_id: str,
    ) -> LoadedModelContext:
        require_no_run_history_lock(
            "StateStepArchive.load_committed_session_model_context"
        )
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if (
            isinstance(message_count, bool)
            or not isinstance(message_count, int)
            or message_count < 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        total = await self._history.history_message_count(
            history_id,
            tenant_id=tenant_id,
        )
        if message_count > total:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        values = await self._facts(agent_run_id, "checkpoint", latest=True)
        if not values:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        stored = _decode_step(values[0].data)
        if (
            not isinstance(stored, StoredAgentRunCheckpoint)
            or stored.agent_run_id != agent_run_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if stored.state != "complete":
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        projection = await self._history.load_projection(history_id)
        if (
            projection is not None
            and projection.digest == stored.projection_digest
        ):
            return await self._history.load_projected_context(
                history_id,
                projection,
            )
        return await self._history.load_session_raw_model_context(
            history_id,
            tenant_id=tenant_id,
            message_count=message_count,
        )

    async def verify_checkpoint_projection(
        self,
        *,
        agent_run_id: str,
        checkpoint: AgentRunCheckpoint,
    ) -> bool:
        require_no_run_history_lock(
            "StateStepArchive.verify_checkpoint_projection"
        )
        values = await self._facts(agent_run_id, "checkpoint", latest=True)
        if not values:
            return False
        stored = _decode_step(values[0].data)
        if not isinstance(stored, StoredAgentRunCheckpoint):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run = await self.get_agent_run(agent_run_id=agent_run_id)
        if run is None:
            return False
        owner_id = (
            self._history_id(run)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else agent_run_id
        )
        projection = await self._history.load_projection(owner_id)
        if projection is None or projection.digest != stored.projection_digest:
            return False
        context = await self._history.load_model_context(owner_id)
        expected_messages = (
            checkpoint.messages
            if checkpoint.context_messages is None
            else checkpoint.context_messages
        )
        return (
            stored.agent_run_id == checkpoint.agent_run_id
            and stored.step_index == checkpoint.step_index
            and stored.timestamp == checkpoint.timestamp
            and stored.state == checkpoint.state
            and stored.has_context_projection
            == (checkpoint.context_messages is not None)
            and stored.pending_request_index == checkpoint.pending_request_index
            and context.model_messages() == tuple(expected_messages)
        )

    async def save_checkpoint(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.save_checkpoint")
        run = await self.get_agent_run(agent_run_id=checkpoint.agent_run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        await self.materialize_checkpoint(
            run,
            checkpoint,
            execution_id=execution_id,
        )

    async def latest_checkpoint(self, *, agent_run_id: str, include_interrupted: bool = False) -> AgentRunCheckpoint | None:
        require_no_run_history_lock("StateStepArchive.latest_checkpoint")
        values = await self._facts(agent_run_id, "checkpoint", latest=True)
        if not values:
            return None
        latest = _decode_step(values[0].data)
        if not isinstance(latest, StoredAgentRunCheckpoint):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run = await self.get_agent_run(agent_run_id=agent_run_id)
        if run is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        owner_id = (
            self._history_id(run)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else agent_run_id
        )
        messages = (await self._history.load_model_context(owner_id)).model_messages()
        raw_messages = tuple(
            [message async for message in self._history.iter_raw_messages(owner_id)]
        )
        if not raw_messages and not latest.has_context_projection:
            raw_messages = tuple(messages)
        context_messages = (
            list(messages) if latest.has_context_projection else None
        )
        latest = AgentRunCheckpoint(
            agent_run_id=latest.agent_run_id,
            step_index=latest.step_index,
            messages=list(raw_messages),
            agent_conversation_id=run.agent_conversation_id,
            parent_agent_run_id=run.parent_agent_run_id,
            agent_name=run.agent_name,
            timestamp=latest.timestamp,
            state=latest.state,
            context_messages=context_messages,
            pending_request_index=latest.pending_request_index,
        )
        return latest if include_interrupted or latest.state == "complete" else None

    async def release_agent_run(
        self,
        agent_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.release_agent_run")

        async def mutate(transaction: StateTransaction) -> None:
            history_head_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
            await transaction.delete_record(self._agent_run_key(agent_run_id))
            await transaction.delete_sequences(
                tuple(
                    self._sequence(agent_run_id, family)
                    for family in ("event", "checkpoint", "interaction")
                )
            )
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

        await self._store.mutate(mutate)
        self._context_baselines.pop(agent_run_id, None)

    def release_runtime_cache(self, agent_run_id: str) -> None:
        self._context_baselines.pop(agent_run_id, None)

    async def _append(
        self,
        agent_run_id: str,
        family: str,
        value: object,
        kind: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive._append")
        stream = self._stream(agent_run_id, family)
        owner = self._agent_run_key(agent_run_id)
        subject = _step_subject(value)
        fact_kind = {
            "event": "step_event",
            "checkpoint": "step_checkpoint",
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
            sequence = await transaction.next_sequence(self._sequence(agent_run_id, family))
            await transaction.insert_fact(StoredFact(stream, sequence, owner, fact_kind, subject, kind, data))
            await self._advance_execution_history_head_in_transaction(
                transaction,
                history_head_guard,
            )

        await self._store.mutate(mutate)

    async def _facts(
        self,
        agent_run_id: str,
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
                    self._stream(agent_run_id, family),
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


def _conversation_relocated_checkpoint_matches(
    source: AgentRunCheckpoint,
    observed: AgentRunCheckpoint | None,
) -> bool:
    if observed is None:
        return False
    if (
        observed.agent_run_id != source.agent_run_id
        or observed.step_index != source.step_index
        or observed.agent_conversation_id != source.agent_conversation_id
        or observed.parent_agent_run_id != source.parent_agent_run_id
        or observed.agent_name != source.agent_name
        or observed.timestamp != source.timestamp
        or observed.state != source.state
        or observed.idempotency_key != source.idempotency_key
        or observed.pending_request_index != source.pending_request_index
        or observed.context_messages != source.context_messages
    ):
        return False
    source_messages = tuple(source.messages)
    observed_messages = tuple(observed.messages)
    if not source_messages:
        return True
    return (
        len(observed_messages) >= len(source_messages)
        and observed_messages[-len(source_messages) :] == source_messages
    )


async def _sync_projection(
    target: AgentRunStore,
    run: AgentRunRecord,
    events: tuple[StepEvent, ...],
    checkpoints: tuple[AgentRunCheckpoint, ...],
    interactions: tuple[ModelInteractionRecord, ...] = (),
    *,
    execution_id: str | None = None,
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.sync_projection(
            run,
            events=events,
            checkpoints=checkpoints,
            interactions=interactions,
            execution_id=execution_id,
        )
        return
    if await target.get_agent_run(agent_run_id=run.agent_run_id) is None:
        await target.register_agent_run(run, execution_id=execution_id)
    for event in events:
        await target.append_event(event, execution_id=execution_id)
    for checkpoint in checkpoints:
        await target.save_checkpoint(checkpoint, execution_id=execution_id)


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


async def _materialize_checkpoint(
    target: AgentRunStore,
    run: AgentRunRecord,
    checkpoint: AgentRunCheckpoint,
    *,
    execution_id: str | None = None,
) -> None:
    if isinstance(target, _StepArchiveBatch):
        await target.materialize_checkpoint(
            run,
            checkpoint,
            execution_id=execution_id,
        )
        return
    existing_run = await target.get_agent_run(agent_run_id=run.agent_run_id)
    existing_checkpoint = await target.latest_checkpoint(
        agent_run_id=run.agent_run_id,
        include_interrupted=True,
    )
    if existing_run == run and existing_checkpoint == checkpoint:
        return
    await target.register_agent_run(run, execution_id=execution_id)
    await target.save_checkpoint(checkpoint, execution_id=execution_id)


def _encode_step(value: object) -> dict[str, object]:
    return _encode_step_envelope(value)


def _step_subject(value: object) -> bytes | None:
    if isinstance(value, ModelInteractionRecord):
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "agent_run_id": value.agent_run_id,
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
    "PreparedAgentRunCheckpoint",
    "PreparedAgentRunCheckpointBatch",
    "StagingAgentRunStore",
    "StateStepArchive",
]
