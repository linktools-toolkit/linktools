#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step staging, archive, projection, and history-lock owners."""

import asyncio
from abc import abstractmethod
import hashlib
import json
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from enum import Enum
from datetime import datetime, timezone
from typing import Protocol, TypeVar, runtime_checkable

from linktools.core import environ
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent as NativeStepEvent

from ..core import ExecutionStatus, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelClientRequest
from ..storage import CacheBackend, ObjectRef, ObjectStore, StoredPayload
from ..model.messages import decode_model_messages, encode_model_messages
from ._transcript_staging import MessageArchive, MessageSnapshot, StagedTranscript
from ._model_interaction import (
    StagedContextSpan,
    StagedContextInline,
    StagedContextProjection,
    StagedModelInteraction,
    context_projection_to_durable,
)
from ._archive import ArchiveRoot, PreparedArchive
from .codec import (
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_step_envelope,
    encode_record,
    decode_record,
)
from .contracts import (
    ArchivedContextBlock,
    ContextProjection,
    ConversationHistoryRepository,
    ExecutionRecord,
    ExecutionTerminalPublication,
    ExecutionRecoveryState,
    ExecutionInsertionReceipt,
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
    TranscriptHeadRecord,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptSpanRef,
)
from ._execution_repositories import ExecutionRepositoryImpl
from ._history import (
    PreparedTranscriptObservation,
    TranscriptCapture,
    TranscriptRepository,
)
from ._model_client import capture_client_requests, prepare_client_requests, resolve_client_requests
from ._native_event_reference import NativeEventReferences
from ._native_run_reference import NativeRunReferences, is_native_run_reference
from ._sql import SqlStateStore
from ._step_store import SqlStepStore, StepWriteReceipt
from ._message_root import NativeMessageRoot
from ._plan import PersistenceDomain
from ._object_router import runtime_object_source_reads
from ._step_contracts import (
    TOOL_ERROR_CODE_METADATA_KEY,
    AgentRunCheckpoint,
    AgentRunHistoryCapture,
    AgentRunRecord,
    StepEvent,
    AgentRunStore,
)
from .state import (
    FactQuery,
    FactReader,
    GroupRecordReader,
    StateGroupTransaction,
    RecordQuery,
    StateLockOrderError,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    active_state_scope,
    active_state_transaction,
    enter_run_history_lock,
    exit_run_history_lock,
    parent_key,
    record_key_digest,
    require_no_run_history_lock,
    scope_key,
    sortable_timestamp,
    stream_digest,
)

_logger = environ.get_logger("ai.runtime.state.run_store")
_ValueT = TypeVar("_ValueT")


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
        archive: PreparedArchive | None = None,
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
    request_projection: ContextProjection | None = None
    observation_task: asyncio.Task[None] | None = None
    observation_batch_size: int | None = None
    observation_flush_interval: float | None = None
    observation_execution_id: str | None = None
    observation_producer_generation: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionProjectionBatch:
    run: AgentRunRecord
    events: tuple[StepEvent, ...]
    checkpoints: Sequence[AgentRunCheckpoint]
    base_event_offset: int
    base_checkpoint_offset: int
    target_event_offset: int
    target_checkpoint_offset: int
    base_message_index: int
    target_message_index: int
    interactions: tuple[StagedModelInteraction, ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0
    transcript: StagedTranscript | None = None
    payloads: tuple[tuple[str, bytes], ...] = ()
    local_messages: Sequence[ModelMessage] = ()


@dataclass(frozen=True, slots=True)
class PreparedAgentRunCheckpoint:
    owner_id: str
    stored: StoredAgentRunCheckpoint
    chunks: tuple[TranscriptChunk, ...]
    projection: ContextProjection
    history_quality: HistoryQuality = HistoryQuality.COMPLETE
    archive: PreparedArchive = field(default_factory=PreparedArchive)
    context_messages: Sequence[ModelMessage] = ()


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
    observation: PreparedTranscriptObservation | None = None
    producer_generation: int | None = None

    archive: PreparedArchive = field(default_factory=PreparedArchive)

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
    checkpoints: Sequence[AgentRunCheckpoint]
    base_event_offset: int
    base_checkpoint_offset: int
    target_event_offset: int
    target_checkpoint_offset: int
    interactions: tuple[StagedModelInteraction, ...] = ()
    base_interaction_offset: int = 0
    target_interaction_offset: int = 0
    transcript: StagedTranscript | None = None
    base_message_index: int = 0
    target_message_index: int = 0
    producer_generation: int | None = None
    payloads: tuple[tuple[str, bytes], ...] = ()
    local_messages: Sequence[ModelMessage] = ()


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


@dataclass(frozen=True, slots=True)
class _StagedCheckpoint:
    value: AgentRunCheckpoint

    @classmethod
    def freeze(cls, value: AgentRunCheckpoint, archive: MessageArchive) -> "_StagedCheckpoint":
        return cls(replace(
            value, messages=archive.snapshot(value.messages),
            context_messages=None if value.context_messages is None else archive.snapshot(value.context_messages),
        ))

    def materialize(self) -> AgentRunCheckpoint:
        return replace(
            self.value, messages=list(self.value.messages),
            context_messages=None if self.value.context_messages is None else list(self.value.context_messages),
        )


class _CheckpointSnapshot(Sequence[AgentRunCheckpoint]):
    def __init__(self, values: tuple[_StagedCheckpoint, ...]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int | slice) -> "AgentRunCheckpoint | _CheckpointSnapshot":
        if isinstance(index, slice):
            return _CheckpointSnapshot(self._values[index])
        # Internal publication retains frozen contents; public reads materialize SDK lists.
        return replace(self._values[index].value)


class StagingAgentRunStore(AgentRunStore):
    """Process-local facts collected before owner materialization."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRunRecord] = {}
        self._native_run_ids: set[str] = set()
        self._events: dict[str, list[StepEvent]] = {}
        self._checkpoints: dict[str, list[_StagedCheckpoint]] = {}
        self._checkpoint_bases: dict[str, int] = {}
        self._latest_checkpoints: dict[str, _StagedCheckpoint] = {}
        self._message_archives: dict[str, MessageArchive] = {}
        self._interactions: dict[str, list[StagedModelInteraction]] = {}
        self._interaction_dirty: dict[str, set[int]] = {}
        self._prepared_interactions: set[tuple[str, int]] = set()
        self._payloads: dict[str, dict[str, bytes]] = {}
        self._transcripts: dict[str, StagedTranscript] = {}
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

    def stage_transcript(self, agent_run_id: str, transcript: StagedTranscript) -> None:
        self._ensure_open()
        archive = self._message_archives.setdefault(agent_run_id, MessageArchive())
        self._transcripts[agent_run_id] = StagedTranscript(
            archive.snapshot(transcript.messages), transcript.pending, transcript.pending_keys,
        )

    def staged_transcript(self, agent_run_id: str) -> StagedTranscript | None:
        self._ensure_open()
        transcript = self._transcripts.get(agent_run_id)
        if transcript is not None:
            return transcript
        checkpoint = self._latest_checkpoints.get(agent_run_id)
        return None if checkpoint is None else StagedTranscript(checkpoint.value.messages)

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
        values = self._interactions.setdefault(interaction.agent_run_id, [])
        if not values or interaction.model_request_seq == values[-1].model_request_seq + 1:
            values.append(interaction)
            self._interaction_dirty.setdefault(interaction.agent_run_id, set()).add(interaction.model_request_seq)
            return
        index = interaction.model_request_seq - values[0].model_request_seq
        if (
            index < 0
            or index >= len(values)
            or values[index].model_request_seq != interaction.model_request_seq
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous = values[index]
        if previous == interaction:
            return
        stable_previous = (
            previous.agent_run_id,
            previous.step_index,
            previous.model_request_seq,
            previous.purpose,
            previous.output_retry_index,
            previous.model,
            previous.request_context,
            previous.request_envelope_digest,
            previous.attachments,
            previous.started_at,
        )
        stable_current = (
            interaction.agent_run_id,
            interaction.step_index,
            interaction.model_request_seq,
            interaction.purpose,
            interaction.output_retry_index,
            interaction.model,
            interaction.request_context,
            interaction.request_envelope_digest,
            interaction.attachments,
            interaction.started_at,
        )
        if (
            previous.status != "RUNNING"
            or interaction.status == "RUNNING"
            or stable_previous != stable_current
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values[index] = interaction
        self._interaction_dirty.setdefault(interaction.agent_run_id, set()).add(interaction.model_request_seq)

    def prepare_model_interaction(self, interaction: object) -> None:
        """Freeze a pending request's prepared SDK input exactly once."""
        self._ensure_open()
        if not isinstance(interaction, StagedModelInteraction):
            raise TypeError("staged model interaction is invalid")
        values = self._interactions.get(interaction.agent_run_id, ())
        key = (interaction.agent_run_id, interaction.model_request_seq)
        if (not values or key in self._prepared_interactions
                or interaction.request_context is None or interaction.request_envelope_digest is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        index = interaction.model_request_seq - values[0].model_request_seq
        if index < 0 or index >= len(values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous = values[index]
        expected = replace(
            previous,
            model=interaction.model,
            request_context=interaction.request_context,
            request_envelope_digest=interaction.request_envelope_digest,
            attachments=interaction.attachments,
        )
        if previous.status != "RUNNING" or expected != interaction:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values[index] = interaction
        self._interaction_dirty.setdefault(interaction.agent_run_id, set()).add(interaction.model_request_seq)
        self._prepared_interactions.add(key)

    def capture_interactions_local(self, agent_run_id: str) -> tuple[StagedModelInteraction, ...]:
        self._ensure_open()
        return tuple(self._interactions.get(agent_run_id, ()))

    def acknowledge_interactions_local(
        self, agent_run_id: str, interactions: Sequence[StagedModelInteraction],
    ) -> None:
        values = self._interactions.get(agent_run_id, ())
        dirty = self._interaction_dirty.get(agent_run_id)
        if not values or dirty is None:
            return
        first = values[0].model_request_seq
        for interaction in interactions:
            index = interaction.model_request_seq - first
            if 0 <= index < len(values) and values[index] == interaction:
                dirty.discard(interaction.model_request_seq)

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        self._ensure_open()
        _validate_interaction_page(after_model_request_seq, limit)
        values = self._interactions.get(agent_run_id, ())
        if not values:
            return []
        start = (
            0
            if after_model_request_seq is None
            else max(
                0,
                after_model_request_seq - values[0].model_request_seq + 1,
            )
        )
        end = None if limit is None else start + limit
        return list(values[start:end])

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        self._ensure_open()
        values = self._interactions.get(agent_run_id, ())
        return values[-1].model_request_seq if values else 0

    async def resolve_model_interaction(
        self, interaction: object, *, include_client_requests: bool = False, include_content: bool = True,
    ) -> object:
        del interaction
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> list[object]:
        del interactions
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def load_loaded_model_context(
        self,
        *,
        owner_id: str,
    ) -> LoadedModelContext:
        self._ensure_open()
        frozen = self._latest_checkpoints.get(owner_id)
        if frozen is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        checkpoint = frozen.value
        messages = checkpoint.messages if checkpoint.context_messages is None else checkpoint.context_messages
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

    def adopt_native_run_local(self, record: AgentRunRecord) -> None:
        """Replace a staged candidate with its acknowledged native registration."""
        self._ensure_open()
        previous = self._runs.get(record.agent_run_id)
        if previous is None:
            return
        if (previous.agent_conversation_id, previous.parent_agent_run_id, previous.agent_id) != (
                record.agent_conversation_id, record.parent_agent_run_id, record.agent_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._runs[record.agent_run_id] = record
        self._native_run_ids.add(record.agent_run_id)

    def native_run_local(self, agent_run_id: str) -> AgentRunRecord | None:
        """Read an acknowledged registration retained until this staging run drains."""
        self._ensure_open()
        return self._runs.get(agent_run_id) if agent_run_id in self._native_run_ids else None

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
        return [value.materialize() for value in self._checkpoints.get(agent_run_id, ())]

    def save_checkpoint_local(self, checkpoint: AgentRunCheckpoint) -> AgentRunCheckpoint:
        self._ensure_open()
        if checkpoint.agent_run_id not in self._runs:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        archive = self._message_archives.setdefault(checkpoint.agent_run_id, MessageArchive())
        frozen = _StagedCheckpoint.freeze(checkpoint, archive)
        values = self._checkpoints.setdefault(checkpoint.agent_run_id, [])
        if frozen != self._latest_checkpoints.get(checkpoint.agent_run_id) and frozen not in values:
            values.append(frozen)
            self._latest_checkpoints[checkpoint.agent_run_id] = frozen
        return replace(frozen.value)

    def latest_checkpoint_local(
        self,
        agent_run_id: str,
        *,
        include_interrupted: bool = False,
    ) -> AgentRunCheckpoint | None:
        self._ensure_open()
        latest = self._latest_checkpoints.get(agent_run_id)
        if latest is None or not include_interrupted and latest.value.state != "complete":
            return None
        return latest.materialize()

    def checkpoint_count_local(self, agent_run_id: str) -> int:
        return self._checkpoint_bases.get(agent_run_id, 0) + len(self._checkpoints.get(agent_run_id, ()))

    def trim_checkpoints_local(self, agent_run_id: str, *, checkpoint_offset: int) -> None:
        base = self._checkpoint_bases.get(agent_run_id, 0)
        values = self._checkpoints.get(agent_run_id, [])
        if not base <= checkpoint_offset <= base + len(values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        del values[:checkpoint_offset - base]
        self._checkpoint_bases[agent_run_id] = checkpoint_offset

    def release_agent_run_local(self, agent_run_id: str) -> None:
        self._ensure_open()
        self._runs.pop(agent_run_id, None)
        self._native_run_ids.discard(agent_run_id)
        self._events.pop(agent_run_id, None)
        self._checkpoints.pop(agent_run_id, None)
        self._checkpoint_bases.pop(agent_run_id, None)
        self._latest_checkpoints.pop(agent_run_id, None)
        self._message_archives.pop(agent_run_id, None)
        self._interactions.pop(agent_run_id, None)
        self._interaction_dirty.pop(agent_run_id, None)
        self._prepared_interactions = {
            key for key in self._prepared_interactions if key[0] != agent_run_id
        }
        self._payloads.pop(agent_run_id, None)
        self._transcripts.pop(agent_run_id, None)

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
        if offset.checkpoints < self._checkpoint_bases.get(agent_run_id, 0):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        interactions = self._interactions.get(agent_run_id, ())
        dirty = self._interaction_dirty.get(agent_run_id, ())
        captured_interactions = tuple(
            interactions[sequence - interactions[0].model_request_seq]
            for sequence in sorted(dirty)
        ) if interactions else ()
        transcript = self.staged_transcript(agent_run_id)
        local_messages = () if transcript is None else transcript.messages
        payload_digests = {
            digest for interaction in captured_interactions for digest, _ in interaction.payload_dependencies()
        }
        payloads = tuple((digest, self.staged_payload(agent_run_id, digest)) for digest in sorted(payload_digests))
        target_message_index = offset.transcript_messages
        if transcript is not None:
            target_message_index = len(transcript.messages)
            if target_message_index < offset.transcript_messages:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            transcript = StagedTranscript(
                transcript.messages[offset.transcript_messages:],
                transcript.pending,
                transcript.pending_keys,
            )
        return ExecutionProjectionBatch(
            run,
            tuple(events[offset.events:]),
            _CheckpointSnapshot(tuple(checkpoints[offset.checkpoints - self._checkpoint_bases.get(agent_run_id, 0):])),
            offset.events,
            offset.checkpoints,
            len(events),
            self.checkpoint_count_local(agent_run_id),
            offset.transcript_messages,
            target_message_index,
            captured_interactions,
            offset.interactions,
            interactions[-1].model_request_seq if interactions else offset.interactions,
            transcript,
            payloads,
            local_messages,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)


class InMemoryStepArchive(StagingAgentRunStore):
    def __init__(self, runtime_domain: PersistenceDomain) -> None:
        super().__init__()
        self._runtime_domain = runtime_domain

    @property
    def runtime_domain(self) -> PersistenceDomain:
        return self._runtime_domain

    async def sync_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
        archive: PreparedArchive | None = None,
    ) -> None:
        del execution_id, archive
        self._ensure_open()
        async with self._lock:
            previous = self._runs.get(run.agent_run_id)
            if previous is not None and previous != run:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            interaction_values = self._interactions.get(run.agent_run_id, [])
            first_sequence = interaction_values[0].model_request_seq if interaction_values else 1
            pending_updates: dict[int, ModelInteractionRecord] = {}
            count = len(interaction_values)
            for interaction in interactions:
                index = interaction.model_request_seq - first_sequence
                if index < 0 or index > count:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                previous_interaction = pending_updates.get(index)
                if previous_interaction is None and index < len(interaction_values):
                    previous_interaction = interaction_values[index]
                if previous_interaction is not None:
                    previous_interaction.validate_successor(interaction)
                elif index == count:
                    count += 1
                pending_updates[index] = interaction
            self._runs[run.agent_run_id] = run
            event_values = self._events.setdefault(run.agent_run_id, [])
            for event in events:
                if event not in event_values:
                    event_values.append(event)
            for checkpoint in checkpoints:
                self.save_checkpoint_local(checkpoint)
            interaction_values = self._interactions.setdefault(run.agent_run_id, [])
            for index, interaction in pending_updates.items():
                if index == len(interaction_values):
                    interaction_values.append(interaction)
                else:
                    interaction_values[index] = interaction

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
        self._ensure_open()
        checkpoint = self._latest_checkpoints.get(agent_run_id)
        if checkpoint is not None:
            for message in checkpoint.value.messages:
                yield message

    async def transcript_message_count(self, agent_run_id: str) -> int:
        self._ensure_open()
        checkpoint = self._latest_checkpoints.get(agent_run_id)
        return 0 if checkpoint is None else len(checkpoint.value.messages)

    async def iter_message_range(
        self,
        *,
        agent_run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._ensure_open()
        checkpoint = self._latest_checkpoints.get(agent_run_id)
        total = 0 if checkpoint is None else len(checkpoint.value.messages)
        if end > total:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if checkpoint is not None:
            for message in checkpoint.value.messages[start:end]:
                yield message

    async def load_model_context(self, *, agent_run_id: str) -> tuple[object, ...]:
        checkpoint = await self.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        return () if checkpoint is None else tuple(checkpoint.messages)

    async def prepare_relocated_interactions(
        self,
        interactions: Sequence[ModelInteractionRecord],
        resolved: Sequence[
            tuple[
                tuple[ModelMessage, ...] | None,
                tuple[ModelMessage, ...] | None,
                bytes | None,
                tuple[ModelClientRequest, ...] | None,
                tuple[tuple[ModelMessage, ...] | None, ...] | None,
            ]
        ],
        *, prepared: PreparedArchive | None = None,
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
        for interaction, (request, response, envelope, client_requests, stream_responses) in zip(
            interactions,
            resolved,
            strict=True,
        ):
            if (interaction.client_requests is None) != (client_requests is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if ((interaction.stream_attempts is None) != (stream_responses is None)
                or stream_responses is not None and len(stream_responses) != len(interaction.stream_attempts)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(
                replace(
                    interaction,
                    request_context=None if request is None else inline_context(request),
                    request_envelope=None if envelope is None else RuntimePayloadRef(
                        StoredPayload.inline_bytes(envelope),
                        self._runtime_domain,
                    ),
                    response_context=(
                        None if response is None or (
                            self._runtime_domain is not PersistenceDomain.TRACE
                            and interaction.stream_attempts and interaction.status != "SUCCEEDED"
                        ) else inline_context(response)
                    ),
                    client_requests=await prepare_client_requests(
                        client_requests, self._runtime_domain,
                    ),
                    stream_attempts=None if interaction.stream_attempts is None else tuple(
                        replace(attempt, response_context=(
                            inline_context(partial)
                            if self._runtime_domain is PersistenceDomain.TRACE and partial is not None else None
                        )) for attempt, partial in zip(interaction.stream_attempts, stream_responses, strict=True)
                    ),
                )
            )
        return tuple(values)

    async def resolve_model_interaction(
        self,
        interaction: object,
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> tuple[
        tuple[ModelMessage, ...] | None, tuple[ModelMessage, ...] | None,
        bytes | None, tuple[ModelClientRequest, ...] | None,
        tuple[tuple[ModelMessage, ...] | None, ...] | None,
    ]:
        if not isinstance(interaction, ModelInteractionRecord):
            raise TypeError("model interaction is invalid")
        interaction.validate_storage_domain(self._runtime_domain)
        client_requests = await resolve_client_requests(
            interaction.client_requests, self._runtime_domain,
        ) if include_client_requests else None
        if not include_content:
            return None, None, None, client_requests, (
                None if interaction.stream_attempts is None else tuple(None for _ in interaction.stream_attempts)
            )
        projections = (
            interaction.request_context, interaction.response_context,
            *(attempt.response_context for attempt in interaction.stream_attempts or ()),
        )
        checkpoint = None
        if (interaction.request_context is not None
                or any(isinstance(item, TranscriptSpanRef) for projection in projections
                       if projection is not None for item in projection.items)):
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
                    if checkpoint is None or item.start < 0 or item.end > len(checkpoint.messages):
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

        envelope = None if interaction.request_envelope is None else interaction.request_envelope.payload.decode()
        if envelope is not None and not isinstance(envelope, bytes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        response = (
            None
            if interaction.response_context is None
            else resolve(interaction.response_context)
        )
        stream_responses = None if interaction.stream_attempts is None else tuple(
            None if attempt.response_context is None else resolve(attempt.response_context)
            for attempt in interaction.stream_attempts
        )
        request = None if interaction.request_context is None else resolve(interaction.request_context)
        return request, response, envelope, client_requests, stream_responses

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> list[object]:
        return [
            await self.resolve_model_interaction(
                interaction, include_client_requests=include_client_requests, include_content=include_content,
            )
            for interaction in interactions
        ]


class StateStepArchive(AgentRunStore):
    """Durable Step owner archive using StateStore Record and Fact primitives."""

    def __init__(
        self,
        store: StateStore,
        *,
        object_store: "ObjectStore | None",
        archive_object_store: "ObjectStore | None" = None,
        archive_object_reader: "Callable[[ObjectRef], Awaitable[bytes]] | None" = None,
        namespace: str,
        tenant_id: str,
        runtime_domain: PersistenceDomain,
        context_sources: Mapping[PersistenceDomain, TranscriptRepository] | None = None,
        history_repository: "ConversationHistoryRepository | None" = None,
        execution_repository: "ExecutionRepository | None" = None,
        object_reader: "Callable[[PersistenceDomain, ObjectRef], Awaitable[bytes]] | None" = None,
        cache: CacheBackend | None = None,
    ) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self._history = TranscriptRepository(
            store,
            object_store=object_store,
            archive_object_store=archive_object_store,
            archive_object_reader=archive_object_reader,
            namespace=namespace,
            tenant_id=tenant_id,
            runtime_domain=runtime_domain,
            context_sources=context_sources,
            history_repository=history_repository,
            object_reader=object_reader,
        )
        self._execution_repository = execution_repository
        self._context_baselines: dict[str, LoadedModelContext] = {}
        self._context_cache = cache if isinstance(cache, CacheBackend) else None
        self._context_cache_keys: OrderedDict[str, str] = OrderedDict()
        self._history_lock = _AgentRunHistoryLock()
        self._closed = False
        self._native_runs: NativeRunReferences | None = None
        self._native_events: NativeEventReferences | None = None

    def enable_native_run_reads(self) -> None:
        """Use the co-located provisioned native schema for Trace source joins."""
        if self._runtime_domain is PersistenceDomain.TRACE and isinstance(self._store, SqlStateStore):
            if self._native_runs is None:
                self._native_runs = NativeRunReferences(self._store, namespace=self._namespace, tenant_id=self._tenant_id)
                self._native_events = NativeEventReferences(self._store, namespace=self._namespace, tenant_id=self._tenant_id)

    def bind_native_run_store(self, native: SqlStepStore) -> None:
        if (type(native) is SqlStepStore and isinstance(self._store, SqlStateStore)
                and native.storage_context.engine is self._store.context.engine
                and native.store_digest == SqlStepStore.scope_digest(
                    namespace=self._namespace, tenant_id=self._tenant_id,
                )):
            self._history.bind_native_message_store(native)
        if self._native_runs is not None:
            self._native_runs.bind(native)
            self._native_events.bind(native)

    def can_share_native_start(self, native: object) -> bool:
        return self._native_events is not None and self._native_events.accepts(native)

    async def stage_native_event(self, run: RunRecord, source: NativeStepEvent, event: StepEvent) -> None:
        if self._native_events is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        await self._native_events.stage(run, source, event)

    def confirm_native_event_source(self, receipt: StepWriteReceipt) -> None:
        if self._native_events is not None:
            self._native_events.confirm_native(receipt)

    @asynccontextmanager
    async def native_event_publication(self, run_id: str) -> AsyncIterator[None]:
        if self._native_events is None:
            yield
        else:
            async with self._native_events.publication(run_id):
                yield

    def has_pending_native_start(self, run_id: str) -> bool:
        return self._native_events is not None and self._native_events.has_pending(run_id)

    def native_event_receipt(self, run_id: str) -> StepWriteReceipt | None:
        return None if self._native_events is None else self._native_events.confirmed_receipt(run_id)

    def release_native_event_receipt(self, run_id: str) -> None:
        if self._native_events is not None:
            self._native_events.release_settled(run_id)

    def confirm_native_event_projection(self, events: Sequence[StepEvent]) -> tuple[StepWriteReceipt, ...]:
        return () if self._native_events is None else self._native_events.confirm(events)

    async def read_event_facts(self, transaction: StateTransaction, query: FactQuery,
                               *, sequences: Sequence[int] | None = None,
                               full_source: bool = False) -> tuple[StoredFact, ...]:
        if self._native_events is not None:
            return await self._native_events.read(transaction, query=query, sequences=sequences, full_source=full_source)
        if sequences is not None and isinstance(transaction, FactReader):
            return await transaction.get_facts(query.stream_digest, sequences)
        return await transaction.list_facts(query)

    def confirm_native_run(self, record: RunRecord) -> None:
        if self._native_runs is not None:
            self._native_runs.confirm(record)

    def invalidate_native_run(self, run_id: str) -> None:
        if self._native_runs is not None:
            self._native_runs.invalidate(run_id)

    async def read_run_records(
        self, transaction: StateTransaction, keys: Sequence[bytes],
    ) -> Mapping[bytes, StoredRecord]:
        """Read DTO projections without replacing the transaction's raw records."""
        if self._native_runs is None:
            return await transaction.get_records(keys)
        return {record.key_digest: record for record in await self._native_runs.read(transaction, keys=keys)}

    @property
    def runtime_domain(self) -> PersistenceDomain:
        return self._runtime_domain

    @property
    def state_store(self) -> StateStore:
        return self._store

    @property
    def transaction_stores(self) -> tuple[StateStore, ...]:
        """Enlist evidence and its authoritative producer head atomically."""
        repository = self._execution_repository
        if repository is not None:
            return tuple(dict.fromkeys((self._store, repository.state_store)))
        return (self._store,)

    async def _mutate(
        self, callback: Callable[[StateTransaction], Awaitable[_ValueT]],
    ) -> _ValueT:
        stores = self.transaction_stores
        if len(stores) == 1:
            return await self._store.mutate(callback)
        group = self._store.storage_group
        if any(store.storage_group is not group for store in stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                          "trace evidence and execution authority require one StateStorageGroup")
        return await group.mutate(
            stores, lambda transaction: callback(transaction.transaction(self._store)),
        )

    def _execution_transaction(self, transaction: StateTransaction) -> StateTransaction:
        repository = self._execution_repository
        if repository is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if repository.state_store is self._store:
            return transaction
        if active_state_transaction(self._store, writable=True) is not transaction:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        core_transaction = active_state_transaction(repository.state_store, writable=True)
        if core_transaction is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return core_transaction

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
        if self._runtime_domain is not PersistenceDomain.TRACE:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        unique_agent_run_ids = tuple(dict.fromkeys(agent_run_ids))
        if not unique_agent_run_ids:
            return {}
        sequence_keys = tuple(
            key
            for agent_run_id in unique_agent_run_ids
            for key in (
                self._stream(agent_run_id, "event"),
                self._stream(agent_run_id, "checkpoint"),
                self._stream(agent_run_id, "interaction"),
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
            records = await self.read_run_records(
                transaction, (*run_keys, *projection_keys, *head_keys)
            )
            sequences = await transaction.get_sequences(sequence_keys)
            return records, sequences

        records, sequences = await self._store.read(read)
        result: dict[str, ExecutionRunSealHead] = {}
        for agent_run_id in unique_agent_run_ids:
            agent_run_record = records.get(self._agent_run_key(agent_run_id))
            head_record = records.get(self._history.head_key(agent_run_id))
            projection_record = records.get(self._history.projection_key(agent_run_id))
            event_count = sequences.get(self._stream(agent_run_id, "event"), 0)
            checkpoint_count = sequences.get(self._stream(agent_run_id, "checkpoint"), 0)
            interaction_count = sequences.get(
                self._stream(agent_run_id, "interaction"), 0
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
                tuple[ModelMessage, ...] | None,
                tuple[ModelMessage, ...] | None,
                bytes | None,
                tuple[ModelClientRequest, ...] | None,
                tuple[tuple[ModelMessage, ...] | None, ...] | None,
            ]
        ],
        *, prepared: PreparedArchive,
    ) -> tuple[ModelInteractionRecord, ...]:
        if len(interactions) != len(resolved):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def archived_context(messages: Sequence[ModelMessage]) -> ContextProjection:
            return await self._history.prepare_model_projection(messages, prepared)

        values: list[ModelInteractionRecord] = []
        for interaction, (request, response, envelope, client_requests, stream_responses) in zip(
            interactions,
            resolved,
            strict=True,
        ):
            if (interaction.client_requests is None) != (client_requests is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if ((interaction.stream_attempts is None) != (stream_responses is None)
                or stream_responses is not None and len(stream_responses) != len(interaction.stream_attempts)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(
                replace(
                    interaction,
                    request_context=None if request is None else await archived_context(request),
                    request_envelope=None if envelope is None else RuntimePayloadRef(
                        StoredPayload.inline_json(json.loads(envelope)),
                        self._runtime_domain,
                    ),
                    response_context=(
                        None
                        if response is None or (
                            self._runtime_domain is not PersistenceDomain.TRACE
                            and interaction.stream_attempts and interaction.status != "SUCCEEDED"
                        )
                        else await archived_context(response)
                    ),
                    client_requests=await prepare_client_requests(
                        client_requests, self._runtime_domain, self._history, prepared=prepared,
                    ),
                    stream_attempts=None if interaction.stream_attempts is None else tuple([
                        replace(attempt, response_context=(
                            await archived_context(partial)
                            if self._runtime_domain is PersistenceDomain.TRACE and partial is not None else None
                        )) for attempt, partial in zip(interaction.stream_attempts, stream_responses, strict=True)
                    ]),
                )
            )
        return tuple(values)

    async def resolve_model_interaction(
        self,
        interaction: object,
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> tuple[
        tuple[ModelMessage, ...] | None, tuple[ModelMessage, ...] | None,
        bytes | None, tuple[ModelClientRequest, ...] | None,
        tuple[tuple[ModelMessage, ...] | None, ...] | None,
    ]:
        if not isinstance(interaction, ModelInteractionRecord):
            raise TypeError("model interaction is invalid")
        interaction.validate_storage_domain(self._runtime_domain)
        client_requests = await resolve_client_requests(
            interaction.client_requests, self._runtime_domain, self._history,
        ) if include_client_requests else None
        if not include_content:
            return None, None, None, client_requests, (
                None if interaction.stream_attempts is None else tuple(None for _ in interaction.stream_attempts)
            )
        request = None if interaction.request_context is None else await self._history.load_projected_context(
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
        envelope = None if interaction.request_envelope is None else await self._history.read_payload(
            interaction.request_envelope.payload
        )
        stream_responses = None if interaction.stream_attempts is None else tuple([
            None if attempt.response_context is None else (await self._history.load_projected_context(
                interaction.agent_run_id, attempt.response_context,
            )).model_messages() for attempt in interaction.stream_attempts
        ])
        return None if request is None else request.model_messages(), response, envelope, client_requests, stream_responses

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> list[object]:
        load = await self._history.read_content(lambda: self.capture_model_interactions(
            interactions, include_client_requests=include_client_requests, include_content=include_content,
        ))
        return await load()

    async def capture_model_interactions(
        self,
        interactions: Sequence[object],
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> Callable[[], Awaitable[list[object]]]:
        require_no_run_history_lock(
            "StateStepArchive.resolve_model_interactions"
        )
        values = tuple(interactions)
        if not values:
            async def empty() -> list[object]:
                return []
            return empty
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise TypeError("model interaction is invalid")
        records = tuple(value for value in values if isinstance(value, ModelInteractionRecord))
        for record in records:
            record.validate_storage_domain(self._runtime_domain)
        projections = tuple(
            projection
            for record in records
            for projection in (
                *((record.request_context,) if record.request_context is not None else ()),
                *(
                    ()
                    if record.response_context is None
                    else (record.response_context,)
                ),
                *(attempt.response_context for attempt in record.stream_attempts or ()
                  if attempt.response_context is not None),
            )
        )
        load_contexts = await self._history.capture_projected_contexts(
            records[0].agent_run_id, projections,
        ) if include_content else None

        client_loader = await capture_client_requests(
            tuple(request for record in records for request in record.client_requests or ()),
            self._runtime_domain, self._history,
        ) if include_client_requests else None

        async def load() -> list[object]:
            contexts = await load_contexts() if load_contexts is not None else ()
            envelopes: dict[str, bytes] = {}
            result: list[object] = []
            context_index = client_index = 0
            clients = await client_loader() if client_loader is not None else ()
            for record in records:
                client_requests = None
                if include_client_requests and record.client_requests is not None:
                    count = len(record.client_requests)
                    client_requests = clients[client_index:client_index + count]
                    client_index += count
                if not include_content:
                    result.append((None, None, None, client_requests,
                                   None if record.stream_attempts is None else tuple(None for _ in record.stream_attempts)))
                    continue
                request = None
                if record.request_context is not None:
                    request = contexts[context_index].model_messages()
                    context_index += 1
                response = None
                if record.response_context is not None:
                    response = contexts[context_index].model_messages()
                    context_index += 1
                envelope = None
                if record.request_envelope is not None:
                    digest = record.request_envelope.payload.digest
                    if digest not in envelopes:
                        envelopes[digest] = await self._history.read_payload(
                            record.request_envelope.payload
                        )
                    envelope = envelopes[digest]
                stream_responses = None
                if record.stream_attempts is not None:
                    partials = []
                    for attempt in record.stream_attempts:
                        partial = None
                        if attempt.response_context is not None:
                            partial = contexts[context_index].model_messages()
                            context_index += 1
                        partials.append(partial)
                    stream_responses = tuple(partials)
                result.append((request, response, envelope, client_requests, stream_responses))
            return result

        return load

    async def prepare_interactions(
        self,
        run: AgentRunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        *,
        local_message_base: int = 0,
        local_message_count: int = 0,
        local_messages: Sequence[ModelMessage] = (),
        prepared: PreparedArchive,
        replay: bool = False,
        baseline: ContextProjection | None = None,
    ) -> tuple[ModelInteractionRecord, ...]:
        values = tuple(
            replace(value, request_context=None, request_envelope_digest=None)
            if self._runtime_domain is PersistenceDomain.TRACE and value.status == "RUNNING"
            else replace(value, response_context=None)
            if self._runtime_domain is not PersistenceDomain.TRACE and value.stream_attempts and value.status != "SUCCEEDED"
            else value
            for value in interactions
        )
        if not values:
            return ()
        if local_message_base < 0 or local_message_count < 0:
            raise ValueError("local interaction transcript range is invalid")
        model_request_seqs: set[int] = set()
        external_refs: list[TranscriptMessageRef] = []
        for interaction in values:
            if (
                interaction.agent_run_id != run.agent_run_id
                or interaction.model_request_seq in model_request_seqs
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            model_request_seqs.add(interaction.model_request_seq)
            for projection in (
                *((interaction.request_context,) if interaction.request_context is not None else ()),
                *(
                    ()
                    if interaction.response_context is None
                    else (interaction.response_context,)
                ),
                *(attempt.response_context for attempt in interaction.stream_attempts or ()
                  if attempt.response_context is not None),
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
        resolved_external = dict(zip(
            external_refs,
            await self._history.resolve_transcript_message_refs(tuple(external_refs)),
            strict=True,
        )) if external_refs else {}

        stored_interactions: dict[int, ModelInteractionRecord] = {}
        if replay:
            first, last = min(model_request_seqs), max(model_request_seqs)
            existing = await self.list_model_interactions(
                agent_run_id=run.agent_run_id,
                after_model_request_seq=first - 1, limit=last - first + 1,
            )
            if any(not isinstance(value, ModelInteractionRecord) for value in existing):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            # Cold replay must verify retained bytes before reusing their immutable roots.
            await self._history.load_projected_contexts(run.agent_run_id, tuple(
                projection for value in existing
                for projection in (value.request_context, value.response_context,
                                   *(attempt.response_context for attempt in value.stream_attempts or ()))
                if projection is not None
            ))
            stored_interactions = {value.model_request_seq: value for value in existing}
        previous_projection = (baseline if baseline is not None
                               else await self._history.load_projection(run.agent_run_id))
        if previous_projection is None:
            context = self._context_baselines.get(run.agent_run_id)
            if (context is not None and isinstance(context.archive_root, NativeMessageRoot)
                    and context.archive_digest is not None):
                previous_projection = ContextProjection((ArchivedContextBlock(
                    context.archive_root, context.archive_digest, len(encode_model_messages(context.model_messages())),
                ),))

        async def prepare_context(
            projection: StagedContextProjection, *, extend: bool,
            stored: ContextProjection | None = None,
        ) -> ContextProjection:
            nonlocal previous_projection
            messages: list[ModelMessage] = []
            for item in projection.items:
                if isinstance(item, StagedContextSpan):
                    if item.end > len(local_messages):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    messages.extend(local_messages[item.start:item.end])
                elif isinstance(item, StagedContextInline):
                    messages.extend(decode_model_messages(payload(item.payload_digest)))
                elif isinstance(item, TranscriptSpanRef):
                    messages.extend(resolved_external[TranscriptMessageRef(
                        item.source_domain, item.owner_id, index,
                    )].message for index in range(item.start, item.end))
                else:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await self._history.prepare_model_projection(messages, prepared,
                baseline=stored if stored is not None else previous_projection if extend else None)
            if extend:
                previous_projection = result
            return result

        result: list[ModelInteractionRecord] = []
        for staged in values:
            previous = stored_interactions.get(staged.model_request_seq)
            request_context = None if staged.request_context is None else await prepare_context(
                staged.request_context, extend=True,
                stored=None if previous is None else previous.request_context,
            )
            request_envelope = None if staged.request_envelope_digest is None else RuntimePayloadRef(
                StoredPayload.inline_json(json.loads(payload(staged.request_envelope_digest))),
                self._runtime_domain,
            )
            response_context = None if staged.response_context is None else await prepare_context(
                staged.response_context, extend=False,
                stored=None if previous is None else previous.response_context,
            )
            result.append(
                ModelInteractionRecord(
                    staged.agent_run_id,
                    staged.step_index,
                    staged.model_request_seq,
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
                    staged.started_at,
                    staged.finished_at,
                    staged.attachments,
                    await prepare_client_requests(
                        staged.client_requests, self._runtime_domain, self._history, prepared=prepared,
                    ),
                    None if staged.stream_attempts is None else tuple([
                        attempt.to_durable(
                            await prepare_context(attempt.response_context, extend=False)
                            if self._runtime_domain is PersistenceDomain.TRACE and attempt.response_context is not None
                            else None
                        ) for attempt in staged.stream_attempts
                    ]),
                )
            )
        return tuple(result)

    async def checkpoint_count(self, *, agent_run_id: str) -> int:
        require_no_run_history_lock("StateStepArchive.checkpoint_count")
        key = self._stream(agent_run_id, "checkpoint")
        return (await self._store.read(lambda transaction: transaction.get_sequences((key,)))).get(key, 0)

    async def verify_checkpoint_range(
        self, *, agent_run_id: str, after_sequence: int,
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
    ) -> bool:
        require_no_run_history_lock("StateStepArchive.verify_checkpoint_range")
        if not checkpoints:
            return True
        facts = await self._store.read(lambda transaction: transaction.list_facts(FactQuery(
            self._stream(agent_run_id, "checkpoint"),
            after_sequence=after_sequence, limit=len(checkpoints),
        )))
        return len(facts) == len(checkpoints) and all(
            fact.sequence == after_sequence + index + 1
            and _decode_step(fact.data) == checkpoint.stored
            for index, (fact, checkpoint) in enumerate(zip(facts, checkpoints, strict=True))
        )

    async def verify_execution_projection_head(
        self,
        projection: PreparedExecutionProjection,
    ) -> bool:
        require_no_run_history_lock(
            "StateStepArchive.verify_execution_projection_head"
        )
        expected_run = (projection.run if self._native_events is None
                        else self._native_events.resolve_run(projection.run))
        source = None if self._native_events is None else self._native_events.pending_source(projection.run.agent_run_id)
        if source is None:
            observed_run = await self.get_agent_run(agent_run_id=projection.run.agent_run_id)
        else:
            records = await self._store.read(lambda transaction: self._native_runs.read(transaction,
                keys=(self._agent_run_key(projection.run.agent_run_id),), expected_run=source[0],
                expected_start=source[1], expected_terminal=source[2]))
            observed_run = None if not records else _decode_step(records[0].data)
        if observed_run != expected_run:
            return False
        head = await self.execution_history_head_record(projection.run.agent_run_id)
        return (
            head.event_count,
            head.checkpoint_count,
            head.transcript_message_count,
            head.projection_digest,
            head.interaction_count,
        ) == (
            projection.target_event_offset,
            projection.target_checkpoint_offset,
            projection.target_transcript_message_count,
            projection.projection_digest,
            projection.target_interaction_offset,
        )

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
            if self._runtime_domain is PersistenceDomain.CONVERSATION
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
        if self._runtime_domain is not PersistenceDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing_run = await self.get_agent_run(agent_run_id=run.agent_run_id)
        return await self._relocate_conversation_checkpoint(run, checkpoint, existing_run)

    async def conversation_preparation_record_keys(
        self, run: AgentRunRecord, checkpoint: AgentRunCheckpoint,
    ) -> Mapping[StateStore, tuple[bytes, ...]] | None:
        """Name the records a built-in conversation preparation can capture together."""
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.conversation_preparation_record_keys")
        if (
            type(self).relocate_conversation_checkpoint is not StateStepArchive.relocate_conversation_checkpoint
            or type(self).prepare_checkpoints is not StateStepArchive.prepare_checkpoints
            or not isinstance(self._store.storage_group, GroupRecordReader)
        ):
            return None
        if self._runtime_domain is not PersistenceDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = await self._history.content_archive(create=True)
        if content.store.storage_group is not self._store.storage_group:
            return None
        keys = {self._store: (self._agent_run_key(run.agent_run_id),
                             self._history.head_key(self._history_id(run)))}
        values = tuple(("model-message", encode_model_messages((message,))) for message in checkpoint.messages)
        keys[content.store] = tuple(dict.fromkeys((*keys.get(content.store, ()),
                                                 *content.content_record_keys(values))))
        return keys

    async def prepare_conversation_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        *,
        archive: PreparedArchive | None = None,
        observed_records: Mapping[StateStore, Mapping[bytes, StoredRecord | None]] | None = None,
    ) -> tuple[AgentRunCheckpoint, PreparedAgentRunCheckpointBatch]:
        """Relocate and prepare one conversation checkpoint from one metadata capture."""
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.prepare_conversation_checkpoint")
        if observed_records is not None and active_state_scope() is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            type(self).relocate_conversation_checkpoint is not StateStepArchive.relocate_conversation_checkpoint
            or type(self).prepare_checkpoints is not StateStepArchive.prepare_checkpoints
        ):
            if observed_records is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            relocated = await self.relocate_conversation_checkpoint(run, checkpoint)
            return relocated, await self.prepare_checkpoints(run, (relocated,))
        if self._runtime_domain is not PersistenceDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run_key = self._agent_run_key(run.agent_run_id)
        head_key = self._history.head_key(self._history_id(run))
        requests = (None if observed_records is not None
                    else await self.conversation_preparation_record_keys(run, checkpoint))
        content = None
        content_keys: tuple[bytes, ...] = ()
        observed_content = None
        if observed_records is not None or requests is not None:
            content = await self._history.content_archive(create=True)
            if observed_records is not None:
                owner_keys = {run_key, head_key}
                if (not isinstance(self._store.storage_group, GroupRecordReader)
                        or content.store.storage_group is not self._store.storage_group
                        or set(observed_records) != {self._store, content.store}
                        or not owner_keys.issubset(observed_records[self._store])
                        or (content.store is not self._store and set(observed_records[self._store]) != owner_keys)
                        or any(not isinstance(key, bytes) or len(key) != 32
                               or (record is not None and record.key_digest != key)
                               for values in observed_records.values() for key, record in values.items())):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                captured = {store: {key: value for key, value in values.items() if value is not None}
                            for store, values in observed_records.items()}
                content_keys = tuple(key for key in observed_records[content.store]
                                     if content.store is not self._store or key not in owner_keys)
            else:
                assert requests is not None
                content_keys = tuple(key for key in requests[content.store] if key not in (run_key, head_key))
                async def capture(group: StateGroupTransaction):
                    if not isinstance(group, GroupRecordReader):
                        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                    return await group.get_records(requests)

                captured = await self._store.storage_group.read_many(tuple(requests), capture)
            records = captured[self._store]
            observed_content = {key: captured[content.store][key] for key in content_keys
                                if key in captured[content.store]}
        else:
            records = await self._store.read(lambda transaction: transaction.get_records((run_key, head_key)))
        record = records.get(run_key)
        existing_run = None if record is None else _decode_step(record.data)
        relocated = await self._relocate_conversation_checkpoint(run, checkpoint, existing_run)
        # Existing runs validate checkpoint bodies and then recapture metadata.
        # A new run can consume this observation; publication still checks fresh records.
        archive = PreparedArchive() if archive is None else archive
        if existing_run is None and observed_content is not None:
            assert content is not None
            content.observe_content(archive, content_keys, observed_content)
        try:
            return relocated, await self._prepare_explicit_checkpoints(
                run, (relocated,), archive=archive,
                observed_records=records if existing_run is None else None,
            )
        finally:
            archive.clear_content_observation()

    async def checkpoint_publication_record_keys(
        self, run: AgentRunRecord, checkpoint: PreparedAgentRunCheckpoint,
    ) -> Mapping[StateStore, tuple[bytes, ...]]:
        content = await self._history.content_archive(create=True)
        keys = {self._store: (self._agent_run_key(run.agent_run_id),
                              *self._history.checkpoint_record_keys(checkpoint.owner_id, checkpoint.chunks))}
        keys[content.store] = tuple(dict.fromkeys((*keys.get(content.store, ()),
                                                *content.publication_record_keys(checkpoint.archive))))
        return keys

    async def _relocate_conversation_checkpoint(
        self,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        existing_run: object | None,
    ) -> AgentRunCheckpoint:
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

    async def prepare_checkpoints(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
        *,
        observed_message_count: int | None = None,
        archive: PreparedArchive | None = None,
    ) -> PreparedAgentRunCheckpointBatch:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.prepare_checkpoints")
        return await self._prepare_checkpoints(
            run,
            checkpoints,
            observed_message_count=observed_message_count,
            archive=archive,
        )

    async def prepare_checkpoints_after_seal(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
        *,
        observed_message_count: int | None = None,
        archive: PreparedArchive | None = None,
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
            observed_message_count=observed_message_count,
            archive=archive,
        )

    async def initialize(self) -> None:
        self._closed = False
        if self._native_runs is not None:
            self._native_runs.clear()
        if self._native_events is not None:
            self._native_events.clear()
        self._context_baselines.clear()
        self._context_cache_keys.clear()

    async def close(self) -> None:
        self._closed = True
        if self._native_runs is not None:
            self._native_runs.clear()
        if self._native_events is not None:
            self._native_events.clear()
        self._context_baselines.clear()
        self._context_cache_keys.clear()

    def _agent_run_key(self, agent_run_id: str) -> bytes:
        return record_key_digest(self._namespace, self._tenant_id, self._runtime_domain.value, "agent_run", agent_run_id)

    def _stream(self, agent_run_id: str, family: str) -> bytes:
        return stream_digest(self._namespace, self._tenant_id, self._runtime_domain.value, family, agent_run_id)

    async def register_agent_run(
        self,
        record: AgentRunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.register_agent_run")

        async def mutate(transaction: StateTransaction) -> None:
            execution_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
            _owner_record, created = await self._ensure_run_with_head_in_transaction(
                transaction,
                record,
            )

        await self._mutate(mutate)

    def _stored_agent_run(self, record: AgentRunRecord, *, source_receipt: StepWriteReceipt | None = None) -> StoredRecord:
        key = self._agent_run_key(record.agent_run_id)
        return StoredRecord(
            key,
            None
            if record.agent_conversation_id is None
            else scope_key(
                self._namespace,
                self._tenant_id,
                self._runtime_domain.value,
                "agent_run",
                "conversation",
                (record.agent_conversation_id,),
            ),
            None
            if record.parent_agent_run_id is None
            else parent_key(
                self._namespace,
                self._tenant_id,
                self._runtime_domain.value,
                "agent_run",
                "parent",
                (record.parent_agent_run_id,),
            ),
            "agent_run",
            sortable_timestamp(record.started_at, key),
            None,
            0,
            None,
            0,
            None,
            (None if self._native_runs is None else self._native_runs.encode(record, source_receipt=source_receipt)) or _encode_step(record),
        )

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        require_no_run_history_lock("StateStepArchive.get_agent_run")
        key = self._agent_run_key(agent_run_id)
        records = await self._store.read(lambda transaction: self.read_run_records(transaction, (key,)))
        stored = records.get(key)
        return None if stored is None else _decode_step(stored.data)

    async def list_agent_runs(
        self, *, parent_agent_run_id: str | None = None, agent_conversation_id: str | None = None
    ) -> list[AgentRunRecord]:
        require_no_run_history_lock("StateStepArchive.list_agent_runs")
        if parent_agent_run_id is not None:
            query = RecordQuery(
                kind="agent_run",
                parent_key=parent_key(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "agent_run",
                    "parent",
                    (parent_agent_run_id,),
                )
            )
        elif agent_conversation_id is not None:
            query = RecordQuery(
                kind="agent_run",
                scope_key=scope_key(
                    self._namespace,
                    self._tenant_id,
                    self._runtime_domain.value,
                    "agent_run",
                    "conversation",
                    (agent_conversation_id,),
                )
            )
        else:
            query = RecordQuery(
                kind="agent_run",
            )
        records = await self._store.read(
            lambda transaction: transaction.list_records(query) if self._native_runs is None
            else self._native_runs.read(transaction, query=query)
        )
        values = [_decode_step(record.data) for record in records]
        return [value for value in values if isinstance(value, AgentRunRecord)]

    async def _prepare_checkpoints(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
        *,
        observed_message_count: int | None = None,
        archive: PreparedArchive | None = None,
    ) -> PreparedAgentRunCheckpointBatch:
        values = checkpoints
        if not values:
            if observed_message_count is not None:
                return PreparedAgentRunCheckpointBatch(
                    run.agent_run_id, (), 0, 0, observed_message_count,
                )
            head_owner = (
                self._history_id(run)
                if self._runtime_domain is PersistenceDomain.CONVERSATION
                else run.agent_run_id
            )
            head = await self._history.get_head(head_owner)
            return PreparedAgentRunCheckpointBatch(
                run.agent_run_id,
                (),
                0,
                0,
                observed_message_count if observed_message_count is not None else (0 if head is None else head.message_count),
            )
        return await self._prepare_explicit_checkpoints(
            run, values, observed_message_count=observed_message_count, archive=archive,
        )

    async def _prepare_explicit_checkpoints(
        self,
        run: AgentRunRecord,
        checkpoints: Sequence[AgentRunCheckpoint],
        *,
        observed_message_count: int | None = None,
        archive: PreparedArchive | None = None,
        observed_records: Mapping[bytes, StoredRecord] | None = None,
    ) -> PreparedAgentRunCheckpointBatch:
        owner_id = (
            self._history_id(run)
            if self._runtime_domain is PersistenceDomain.CONVERSATION
            else run.agent_run_id
        )
        head_key = self._history.head_key(owner_id)
        run_key = self._agent_run_key(run.agent_run_id)
        projection_key = self._history.projection_key(owner_id)
        records = observed_records
        if records is None:
            records = await self._store.read(lambda transaction: transaction.get_records((head_key, run_key, projection_key)))
        head_record = records.get(head_key)
        if head_record is not None:
            head = self._history.decode_head(head_record)
        else:
            run_record = records.get(run_key)
            if run_record is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = self._history.empty_head(owner_id)

        if observed_message_count is not None:
            if observed_message_count < head.message_count:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = replace(head, message_count=observed_message_count)
        first_before = checkpoints[0].transcript_message_count_before
        if first_before is None or first_before > head.message_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive_base = (head.message_count - first_before
                        if self._runtime_domain is PersistenceDomain.CONVERSATION else 0)
        target_message_count = head.message_count
        baseline = self._context_baselines.get(run.agent_run_id, LoadedModelContext(()))
        baseline_messages = baseline.model_messages()
        message_archive = MessageArchive()
        baseline_snapshot = message_archive.snapshot(baseline_messages)
        prepared: list[PreparedAgentRunCheckpoint] = []
        archive = PreparedArchive() if archive is None else archive
        previous_projection = None
        previous_record = records.get(projection_key)
        if previous_record is not None:
            previous_projection = _decode_enveloped_domain(previous_record.data, ContextProjection)
        elif baseline.archive_root is not None and baseline.archive_digest is not None:
            previous_projection = ContextProjection((ArchivedContextBlock(
                baseline.archive_root, baseline.archive_digest, len(encode_model_messages(baseline_messages)),
            ),))

        for checkpoint in checkpoints:
            before = checkpoint.transcript_message_count_before
            incoming = message_archive.snapshot(checkpoint.messages)
            if (
                before is None
                or before > len(incoming)
                or archive_base + before > target_message_count
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            delta = incoming[max(before, target_message_count - archive_base):]
            projection_messages = (
                message_archive.concatenate(baseline_snapshot, incoming)
                if checkpoint.context_messages is None
                else message_archive.snapshot(checkpoint.context_messages)
            )
            frontier = ()
            if previous_projection is not None and len(previous_projection.items) == 1:
                block = previous_projection.items[0]
                if (isinstance(block, ArchivedContextBlock) and block.root == baseline.archive_root
                        and block.raw_digest == baseline.archive_digest):
                    frontier = baseline.archive_frontier
            projection = await self._history.prepare_model_projection(
                projection_messages, archive, baseline=previous_projection, baseline_frontier=frontier,
            )
            previous_projection = projection
            chunks = await self._prepare_captured_chunks(
                owner_id,
                TranscriptCapture(
                    target_message_count,
                    tuple(delta),
                    (TranscriptOrigin.RAW,) * len(delta),
                    head.quality,
                ),
                prepared=archive,
            )
            prepared.append(
                PreparedAgentRunCheckpoint(
                    owner_id,
                    StoredAgentRunCheckpoint(
                        run.agent_run_id,
                        checkpoint.step_index,
                        checkpoint.timestamp,
                        checkpoint.state,
                        projection,
                        archive_base + len(incoming),
                        checkpoint.context_messages is not None,
                        checkpoint.pending_request_index,
                        checkpoint.output_retries_used,
                    ),
                    chunks,
                    projection,
                    head.quality,
                    archive,
                    projection_messages,
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

    def _history_id(self, run: AgentRunRecord) -> str:
        history_id = run.metadata.get("history_id")
        return history_id or run.agent_run_id

    async def _prepare_captured_chunks(
        self,
        owner_id: str,
        capture: TranscriptCapture,
        *,
        message_index_offset: int = 0,
        prepared: PreparedArchive,
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
                    prepared=prepared,
                )
            )
            offset += end - start
            start = end
        return tuple(result)

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
        if values and any(checkpoint.owner_id != values[0].owner_id for checkpoint in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def _execution_history_guard_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str | None,
        execution_guard: tuple[ExecutionRecord, StoredRecord] | None,
        producer_generation: int | None = None,
        *,
        allow_cancelling: bool = True,
        inserted_terminal: ExecutionInsertionReceipt | None = None,
    ) -> tuple[ExecutionRecord, StoredRecord] | None:
        if self._execution_repository is None:
            if self._runtime_domain is PersistenceDomain.TRACE:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if execution_guard is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
        if not execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        core_transaction = self._execution_transaction(transaction)
        if execution_guard is not None:
            execution, record = execution_guard
            if inserted_terminal is not None:
                # The expected-absent terminal insertion owns this atomic publication.
                if (inserted_terminal.transaction is not core_transaction
                        or execution != inserted_terminal.execution or record != inserted_terminal.record
                        or execution.execution_id != execution_id
                        or execution.history_seal_digest is None or not execution.retention_closed
                        or execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                        or record.storage_revision != 0
                        or (producer_generation is not None and execution.producer_generation != producer_generation)
                        or await core_transaction.get_record(record.key_digest) != record):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return execution_guard
            if (execution.execution_id != execution_id or execution.history_seal_digest is not None
                    or execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                    or (producer_generation is not None and execution.producer_generation != producer_generation)):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return execution_guard
        if inserted_terminal is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if producer_generation is not None:
            return await self._execution_repository.require_execution_producer_in_transaction(
                core_transaction, execution_id,
                expected_producer_generation=producer_generation,
                allow_cancelling=allow_cancelling,
            )
        return await self._execution_repository.require_execution_history_write_in_transaction(
            core_transaction, execution_id,
        )

    async def guard_execution_producer(
        self, transaction: StateTransaction, *, execution_id: str,
        producer_generation: int, allow_cancelling: bool = True,
    ) -> None:
        if self._execution_repository is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        await self._execution_repository.require_execution_producer_in_transaction(
            self._execution_transaction(transaction), execution_id,
            expected_producer_generation=producer_generation, allow_cancelling=allow_cancelling,
        )

    async def require_revoked_execution_producer(
        self, *, execution_id: str, producer_generation: int,
        terminal_execution: ExecutionRecord | None = None,
        terminal_publication: ExecutionTerminalPublication | None = None,
    ) -> None:
        """Prove source control permanently revoked this producer's publication."""
        require_no_run_history_lock("StateStepArchive.require_revoked_execution_producer")
        repository = self._execution_repository
        if repository is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if (terminal_publication is not None and terminal_execution is not None
                and terminal_publication.execution == terminal_execution
                and terminal_execution.execution_id == execution_id
                and terminal_execution.producer_generation == producer_generation
                and not terminal_execution.durable_trace
                and terminal_execution.recovery_state is not ExecutionRecoveryState.HANDOFF
                and terminal_execution.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}):
            return
        await repository.require_revoked_producer(
            execution_id, tenant_id=self._tenant_id, producer_generation=producer_generation,
        )

    async def sync_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
        archive: PreparedArchive | None = None,
    ) -> None:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.sync_projection")
        prepared = await self._prepare_checkpoints(
            run,
            checkpoints,
        )
        async def commit(transaction: StateTransaction) -> None:
            if archive is not None:
                await self._history.publish_archive(archive)
            await self._sync_projection_in_transaction(
                transaction, run, events=events, checkpoints=prepared.checkpoints,
                interactions=interactions, execution_id=execution_id,
            )
            if archive is not None:
                await self._history.publish_native_archive(transaction, archive, owner_written=True)
        await self._mutate(commit)

    def combine_projection_archives(
        self,
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        observation: PreparedTranscriptObservation | None,
        archive: PreparedArchive,
    ) -> tuple[tuple[PreparedAgentRunCheckpoint, ...], PreparedTranscriptObservation | None, PreparedArchive]:
        """Publish one immutable-content batch for one projection transaction."""
        values = (archive, *(checkpoint.archive for checkpoint in checkpoints),
                  *((observation.archive,) if observation is not None else ()))
        combined = PreparedArchive.combine(values)
        checkpoints = tuple(replace(checkpoint, archive=PreparedArchive()) for checkpoint in checkpoints)
        observation = None if observation is None else replace(observation, archive=PreparedArchive())
        return checkpoints, observation, combined

    async def sync_prepared_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        observation: PreparedTranscriptObservation | None = None,
        producer_generation: int | None = None,
        execution_id: str | None = None,
        archive: PreparedArchive | None = None,
    ) -> None:
        """Commit a prepared projection without preparing its payload twice."""
        self._ensure_open()
        require_no_run_history_lock(
            "StateStepArchive.sync_prepared_projection"
        )
        values = tuple(checkpoints)
        if any(not isinstance(value, PreparedAgentRunCheckpoint) for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def commit(transaction: StateTransaction) -> None:
            if archive is not None:
                await self._history.publish_archive(archive)
            await self._sync_projection_in_transaction(
                transaction,
                run,
                events=events,
                checkpoints=values,
                interactions=interactions,
                observation=observation,
                producer_generation=producer_generation,
                execution_id=execution_id,
            )
            if archive is not None:
                await self._history.publish_native_archive(transaction, archive, owner_written=True)
        await self._mutate(commit)

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        observation: PreparedTranscriptObservation | None = None,
        producer_generation: int | None = None,
        execution_id: str | None = None,
        execution_guard: tuple[ExecutionRecord, StoredRecord] | None = None,
        inserted_terminal: ExecutionInsertionReceipt | None = None,
        prefetch_keys: Sequence[bytes] = (),
    ) -> bool:
        """Commit projection metadata and report whether its AgentRun was admitted."""
        self._ensure_open()
        if interactions:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        checkpoints = await self._normalize_checkpoints_in_transaction(
            transaction,
            run,
            checkpoints,
        )
        facts = tuple(
            ("event", event, _step_event_type(event)) for event in events
        ) + tuple(
            ("checkpoint", checkpoint.stored, checkpoint.stored.state)
            for checkpoint in checkpoints
        )
        execution_guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            execution_guard,
            producer_generation,
            inserted_terminal=inserted_terminal,
        )
        source_receipt = None
        if self._native_events is not None:
            source_receipt = await self._native_events.publish(transaction, events)
            run = self._native_events.resolve_run(run)
        owner_record, created = await self._ensure_run_with_head_in_transaction(
            transaction,
            run,
            include_projection=bool(checkpoints),
            source_receipt=source_receipt,
            prefetch_keys=(
                *prefetch_keys,
                *(key for checkpoint in checkpoints for key in
                  self._history.checkpoint_record_keys(checkpoint.owner_id, checkpoint.chunks)),
            ),
        )
        if observation is not None:
            await self._history.commit_observation(transaction, observation)
        if not facts:
            return created
        owner = self._agent_run_key(run.agent_run_id)
        grouped: dict[str, list[object]] = {
            "event": [],
            "checkpoint": [],
        }
        kinds: dict[str, list[str]] = {
            "event": [],
            "checkpoint": [],
        }
        for family, value, kind in facts:
            grouped[family].append(value)
            kinds[family].append(kind)
        stored_facts: list[StoredFact] = []
        streams = {
            family: self._stream(run.agent_run_id, family)
            for family in ("event", "checkpoint") if grouped[family]
        }
        high_waters = await transaction.reserve_sequences({
            stream: len(grouped[family]) for family, stream in streams.items()
        })
        for family in ("event", "checkpoint"):
            values = grouped[family]
            if not values:
                continue
            stream = streams[family]
            final = high_waters[stream]
            sequences = tuple(range(final - len(values) + 1, final + 1))
            if family == "event":
                await self.index_history_events_in_transaction(
                    transaction, run, tuple(zip(sequences, events)),
                )
            fact_kind = {
                "event": "step_event",
                "checkpoint": "step_checkpoint",
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
                        ((self._native_events.encode(value) if self._native_events is not None and family == "event" else None)
                         or _encode_step(value)),
                    )
                )
        if checkpoints:
            published_archives: set[int] = set()
            for checkpoint in checkpoints:
                if id(checkpoint.archive) not in published_archives:
                    await self._history.publish_archive(checkpoint.archive)
                    published_archives.add(id(checkpoint.archive))
            await self._history.append_checkpoint(
                transaction,
                checkpoints[0].owner_id,
                tuple(
                    chunk
                    for checkpoint in checkpoints
                    for chunk in checkpoint.chunks
                ),
                checkpoints[-1].projection,
                min(
                    (checkpoint.history_quality for checkpoint in checkpoints),
                    key=lambda value: value is HistoryQuality.COMPLETE,
                    default=HistoryQuality.COMPLETE,
                ),
            )
        if await transaction.guard_record(
            owner,
            expected_storage_revision=owner_record.storage_revision,
        ) is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await _insert_facts(transaction, tuple(stored_facts))
        for checkpoint in checkpoints:
            await self._history.publish_native_archive(transaction, checkpoint.archive, owner_written=True)
        return created

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

        await self._mutate(mutate)

    async def _materialize_checkpoint_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        checkpoint: PreparedAgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        execution_guard: tuple[ExecutionRecord, StoredRecord] | None = None,
    ) -> int:
        if not isinstance(checkpoint, PreparedAgentRunCheckpoint):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            execution_guard,
        )
        # Pin ownership before the fact read so READ COMMITTED races fail the CAS.
        owner_record, created = await self._ensure_run_with_head_in_transaction(
            transaction, run, include_projection=True,
            prefetch_keys=self._history.checkpoint_record_keys(
                checkpoint.owner_id, checkpoint.chunks,
            ),
        )
        stream = self._stream(run.agent_run_id, "checkpoint")
        data = _encode_step(checkpoint.stored)
        existing = () if created else await transaction.list_facts(
            FactQuery(stream, latest=True)
        )
        for fact in existing:
            if fact.data == data and fact.state == checkpoint.stored.state:
                return fact.sequence
        owner = self._agent_run_key(run.agent_run_id)
        checkpoint_fact = StoredFact(
            stream, 0, owner, "step_checkpoint", None,
            checkpoint.stored.state, data,
        )
        await self._history.publish_archive(checkpoint.archive)
        batch_new_owner = created and bool(checkpoint.chunks) and type(self._history) is TranscriptRepository
        if batch_new_owner:
            # Inserting the run already pins its owner; existing runs retain
            # their transcript-first then run-guard write order below.
            await self._history.append_checkpoint(
                transaction, checkpoint.owner_id, checkpoint.chunks,
                checkpoint.projection, checkpoint.history_quality,
                checkpoint_fact=checkpoint_fact,
            )
        else:
            await self._history.append_checkpoint(
                transaction, checkpoint.owner_id, checkpoint.chunks,
                checkpoint.projection, checkpoint.history_quality,
            )
        if await transaction.guard_record(
            owner,
            expected_storage_revision=owner_record.storage_revision,
        ) is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if not batch_new_owner:
            sequence = (await _reserve_sequences(
                transaction, stream, 1,
            ))[0]
            await _insert_facts(transaction, (replace(checkpoint_fact, sequence=sequence),))
        else:
            sequence = await self.checkpoint_count(agent_run_id=run.agent_run_id)
        await self._history.publish_native_archive(transaction, checkpoint.archive, owner_written=True)
        return sequence

    async def materialize_checkpoint_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        checkpoint: PreparedAgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        execution_guard: tuple[ExecutionRecord, StoredRecord] | None = None,
    ) -> int | None:
        """Materialize and return its exact position; legacy overrides may return None."""
        require_no_run_history_lock(
            "StateStepArchive.materialize_checkpoint_in_transaction"
        )
        return await self._materialize_checkpoint_in_transaction(
            transaction,
            run,
            checkpoint,
            execution_id=execution_id,
            execution_guard=execution_guard,
        )

    async def sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        observation: PreparedTranscriptObservation | None = None,
        producer_generation: int | None = None,
        execution_id: str | None = None,
        execution_guard: tuple[ExecutionRecord, StoredRecord] | None = None,
        inserted_terminal: ExecutionInsertionReceipt | None = None,
        archive: PreparedArchive | None = None,
    ) -> None:
        require_no_run_history_lock(
            "StateStepArchive.sync_projection_in_transaction"
        )
        if archive is not None:
            await self._history.publish_archive(archive)
        await self._sync_projection_in_transaction(
            transaction,
            run,
            events=events,
            checkpoints=checkpoints,
            interactions=interactions,
            observation=observation,
            producer_generation=producer_generation,
            execution_id=execution_id,
            execution_guard=execution_guard,
            inserted_terminal=inserted_terminal,
        )
        if archive is not None:
            await self._history.publish_native_archive(transaction, archive, owner_written=True)

    async def _ensure_run_with_head_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        include_projection: bool = False,
        prefetch_keys: Sequence[bytes] = (),
        source_receipt: StepWriteReceipt | None = None,
    ) -> tuple[StoredRecord, bool]:
        owner = self._agent_run_key(run.agent_run_id)
        history_owner = (
            self._history_id(run)
            if self._runtime_domain is PersistenceDomain.CONVERSATION
            else run.agent_run_id
        )
        head_key = self._history.head_key(history_owner)
        keys = (owner, head_key)
        if include_projection:
            keys += (self._history.projection_key(history_owner),)
        records = (await transaction.get_records((*keys, *prefetch_keys)) if self._native_runs is None else
                   await self._native_runs.read_physical(transaction, (*keys, *prefetch_keys)))
        owner_record = records.get(owner)
        head_record = records.get(head_key)
        if owner_record is None:
            stored_run = self._stored_agent_run(run, source_receipt=source_receipt)
            if self._native_runs is not None:
                self._native_runs.remember_prepared(transaction, stored_run, run, source_receipt=source_receipt)
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
        elif (self._native_runs is None or not self._native_runs.matches(owner_record.data, run)
              if is_native_run_reference(owner_record.data) else _decode_step(owner_record.data) != run):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif head_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._history.decode_head(head_record)
        if self._native_runs is not None and not is_native_run_reference(owner_record.data):
            reference = self._native_runs.encode(run, source_receipt=source_receipt)
            if reference is not None:
                updated = replace(owner_record, data=reference, storage_revision=owner_record.storage_revision + 1)
                self._native_runs.remember_prepared(transaction, updated, run, source_receipt=source_receipt)
                if not await transaction.replace_record(updated, expected_storage_revision=owner_record.storage_revision):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                owner_record = updated
        return owner_record, False

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
            _step_event_type(event),
            execution_id=execution_id,
        )

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        require_no_run_history_lock("StateStepArchive.list_events")
        values = await self._facts(agent_run_id, "event")
        return [_decode_step(value.data) for value in values]

    async def interrupt_model_interactions(
        self, *, agent_run_id: str, execution_id: str, producer_generation: int,
    ) -> None:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    @abstractmethod
    async def list_model_interactions(
        self, *, agent_run_id: str,
        after_model_request_seq: int | None = None, limit: int | None = None,
    ) -> list[object]:
        raise NotImplementedError

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        require_no_run_history_lock("StateStepArchive.model_interaction_count")
        key = self._stream(agent_run_id, "interaction")
        return (await self._store.read(lambda transaction: transaction.get_sequences((key,)))).get(key, 0)

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
        if self._runtime_domain is PersistenceDomain.CONVERSATION:
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

    async def committed_session_message_count(
        self, history_id: str, *, agent_run_id: str, tenant_id: str,
    ) -> int:
        """Read the selected committed run boundary without loading its body."""
        run = await self.get_agent_run(agent_run_id=agent_run_id)
        if run is None or self._history_id(run) != history_id:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        values = await self._facts(agent_run_id, "checkpoint", latest=True)
        if not values:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        checkpoint = _decode_step(values[0].data)
        if not isinstance(checkpoint, StoredAgentRunCheckpoint):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        inherited = await self._history.inherited_message_count(history_id, tenant_id=tenant_id)
        return inherited + checkpoint.transcript_message_count

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

    async def latest_checkpoint_sequence(self, *, agent_run_id: str) -> int:
        """Resolve the current immutable fact for an explicitly publishing caller."""
        values = await self._facts(agent_run_id, "checkpoint", latest=True)
        if not values:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        return values[0].sequence

    async def checkpoint_at_sequence(
        self, *, agent_run_id: str, checkpoint_sequence: int,
    ) -> StoredAgentRunCheckpoint:
        self._ensure_open()
        if isinstance(checkpoint_sequence, bool) or not isinstance(checkpoint_sequence, int) or checkpoint_sequence < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values = await self._store.read(lambda transaction: transaction.list_facts(FactQuery(
            self._stream(agent_run_id, "checkpoint"), after_sequence=checkpoint_sequence - 1, limit=1,
        )))
        if not values or values[0].sequence != checkpoint_sequence:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        stored = _decode_step(values[0].data)
        if (not isinstance(stored, StoredAgentRunCheckpoint) or stored.agent_run_id != agent_run_id
                or stored.state != values[0].state):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def confirm_session_context(
        self, *, checkpoint_sequence: int,
        prepared: PreparedAgentRunCheckpoint, archive_records: PreparedArchive | None = None,
    ) -> None:
        """Publish a prepared context only after its Session/fact commit is confirmed."""
        self._ensure_open()
        if active_state_scope() is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (self._runtime_domain is not PersistenceDomain.CONVERSATION
                or prepared.stored.state != "complete"
                or isinstance(checkpoint_sequence, bool) or not isinstance(checkpoint_sequence, int)
                or checkpoint_sequence < 1):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._context_cache is None:
            return
        blocks = prepared.stored.projection.items
        if len(blocks) != 1 or not isinstance(blocks[0], ArchivedContextBlock):
            return
        block = blocks[0]
        frozen = MessageArchive().snapshot(prepared.context_messages)
        if frozen.encoded_digest() != (block.raw_digest, block.raw_size):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive = await self._history.content_archive()
        frontier = (() if isinstance(block.root, NativeMessageRoot) else
                    archive.prepared_frontier(block.root, prepared.archive if archive_records is None else archive_records))
        context = LoadedModelContext((), block.root, block.raw_digest, frontier)
        key = await self._session_context_cache_key(prepared.stored.agent_run_id,
            checkpoint_sequence, prepared.stored.projection_digest)
        await self._remember_session_context(key, prepared.stored.agent_run_id, frozen, context)

    async def _session_context_cache_key(
        self, agent_run_id: str, sequence: int, digest: str,
    ) -> str:
        archive = await self._history.content_archive()
        return "conversation-context:" + canonical_sha256([
            archive.scope_id, self._namespace, self._tenant_id,
            self._runtime_domain.value, agent_run_id, sequence, digest,
        ])

    async def _remember_session_context(
        self, key: str, agent_run_id: str, frozen: MessageSnapshot, context: LoadedModelContext,
    ) -> None:
        if self._context_cache is None or context.archive_root is None:
            return
        content = canonical_json_bytes({
            "root": context.archive_root.to_json(), "digest": context.archive_digest,
            "frontier": [encode_record(value) for value in context.archive_frontier],
            "messages": json.loads(encode_model_messages(frozen)),
        })
        try:
            await self._context_cache.put(key, content)
        except Exception:
            return
        self._context_cache_keys[key] = agent_run_id
        self._context_cache_keys.move_to_end(key)
        while len(self._context_cache_keys) > 4096:
            self._context_cache_keys.popitem(last=False)

    async def _cached_session_context(self, content: bytes | None, projection: ContextProjection) -> LoadedModelContext | None:
        if content is None:
            return None
        try:
            value = json.loads(content)
            wire = value["root"]
            root = (NativeMessageRoot.from_json(wire) if wire.get("type") == "native_message_root"
                    else ArchiveRoot.from_json(wire))
            raw = canonical_json_bytes(value["messages"])
            messages = decode_model_messages(raw)
            block = ArchivedContextBlock(root, value["digest"], len(raw))
            archive = await self._history.content_archive()
            valid_scope = ((root.group_id == archive.scope_id and root.store_digest == SqlStepStore.scope_digest(
                namespace=self._namespace, tenant_id=self._tenant_id)) if isinstance(root, NativeMessageRoot)
                else root.scope_id == archive.scope_id and root.content_scope == archive.content_scope)
            if (not valid_scope
                    or root.count != len(messages) or hashlib.sha256(raw).hexdigest() != block.raw_digest
                    or projection != ContextProjection((block,))):
                return None
            frontier = tuple(decode_record(record) for record in value["frontier"])
            if frontier:
                if isinstance(root, NativeMessageRoot):
                    return None
                archive.reuse_frontier(root, frontier, PreparedArchive())
            return LoadedModelContext(tuple(LoadedContextMessage(message, None) for message in messages),
                                      root, block.raw_digest, frontier)
        except (AIError, KeyError, TypeError, ValueError):
            return None

    async def load_committed_session_model_context(
        self, *, agent_run_id: str,
        checkpoint_sequence: int, projection_digest: str, tenant_id: str,
        include_content: bool = True,
    ) -> LoadedModelContext:
        require_no_run_history_lock("StateStepArchive.load_committed_session_model_context")
        self._ensure_open()
        if self._runtime_domain is not PersistenceDomain.CONVERSATION:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        if (isinstance(checkpoint_sequence, bool) or not isinstance(checkpoint_sequence, int)
                or checkpoint_sequence < 1):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = None
        cached_bytes = None
        if include_content and self._context_cache is not None:
            key = await self._session_context_cache_key(agent_run_id, checkpoint_sequence, projection_digest)
            try:
                cached_bytes = await self._context_cache.get(key)
            except Exception:
                pass

        async def capture(*, use_cache: bool = True):
            stored = await self.checkpoint_at_sequence(
                agent_run_id=agent_run_id, checkpoint_sequence=checkpoint_sequence,
            )
            if stored.state != "complete":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if stored.projection_digest != projection_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            load = None
            if include_content and (not use_cache or cached_bytes is None):
                load = await self._history.capture_projected_contexts(agent_run_id, (stored.projection,))
            return stored, load

        stored, load = await self._history.read_content(capture)
        if not include_content:
            return LoadedModelContext(())
        if cached_bytes is not None:
            cached = await self._cached_session_context(cached_bytes, stored.projection)
            if cached is not None:
                return cached
            stored, load = await self._history.read_content(lambda: capture(use_cache=False))
        # The selected immutable fact and native rows share one read snapshot;
        # no mutable latest head can redefine this committed context revision.
        context = (await load())[0]
        if key is not None and all(message.source is None for message in context.messages):
            await self._remember_session_context(key, agent_run_id, MessageArchive().snapshot(context.model_messages()), context)
        return context

    async def verify_checkpoint_projection(
        self,
        *,
        agent_run_id: str,
        checkpoint: AgentRunCheckpoint,
    ) -> bool:
        require_no_run_history_lock(
            "StateStepArchive.verify_checkpoint_projection"
        )
        async def capture():
            values = await self._facts(agent_run_id, "checkpoint", latest=True)
            if not values:
                return None
            stored = _decode_step(values[0].data)
            if not isinstance(stored, StoredAgentRunCheckpoint):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            run = await self.get_agent_run(agent_run_id=agent_run_id)
            if run is None:
                return None
            owner_id = (
                self._history_id(run)
                if self._runtime_domain is PersistenceDomain.CONVERSATION
                else agent_run_id
            )
            return stored, await self._history.capture_projected_contexts(owner_id, (stored.projection,))

        captured = await self._history.read_content(capture)
        if captured is None:
            return False
        stored, load = captured
        with runtime_object_source_reads():
            context = (await load())[0]
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
            and stored.output_retries_used == checkpoint.output_retries_used
            and (self._runtime_domain is PersistenceDomain.CONVERSATION
                 or stored.transcript_message_count == len(checkpoint.messages))
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
        async def capture():
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
                if self._runtime_domain is PersistenceDomain.CONVERSATION
                else agent_run_id
            )
            load_context = await self._history.capture_projected_contexts(owner_id, (latest.projection,))
            load_messages = await self._history.capture_message_spans(owner_id, ((0, latest.transcript_message_count),))
            return latest, run, load_context, load_messages

        captured = await self._history.read_content(capture)
        if captured is None:
            return None
        latest, run, load_context, load_messages = captured
        with runtime_object_source_reads():
            messages = (await load_context())[0].model_messages()
            raw_messages = (await load_messages())[0]
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
            agent_id=run.agent_id,
            timestamp=latest.timestamp,
            state=latest.state,
            context_messages=context_messages,
            pending_request_index=latest.pending_request_index,
            output_retries_used=latest.output_retries_used,
        )
        return latest if include_interrupted or latest.state == "complete" else None

    async def release_agent_run(
        self,
        agent_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        require_no_run_history_lock("StateStepArchive.release_agent_run")

        for key, cached_run in tuple(self._context_cache_keys.items()):
            if cached_run == agent_run_id:
                self._context_cache_keys.pop(key, None)
                if self._context_cache is not None:
                    try:
                        await self._context_cache.delete(key)
                    except Exception:
                        pass

        async def mutate(transaction: StateTransaction) -> None:
            execution_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
            interactions = await transaction.list_records(RecordQuery(
                parent_key=self._agent_run_key(agent_run_id),
            ))
            await transaction.delete_records((
                *(interaction.key_digest for interaction in interactions),
                self._agent_run_key(agent_run_id),
            ))
            await transaction.delete_sequences(
                tuple(
                    self._stream(agent_run_id, family)
                    for family in ("event", "checkpoint", "interaction")
                )
            )

        await self._mutate(mutate)
        self._context_baselines.pop(agent_run_id, None)

    def release_runtime_cache(self, agent_run_id: str) -> None:
        self._context_baselines.pop(agent_run_id, None)
        self.invalidate_native_run(agent_run_id)
        if self._native_events is not None:
            self._native_events.clear(agent_run_id)

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
        fact_kind = {
            "event": "step_event",
            "checkpoint": "step_checkpoint",
        }[family]
        data = _encode_step(value)

        async def mutate(transaction: StateTransaction) -> None:
            owner_record = (await self.read_run_records(transaction, (owner,))).get(owner)
            if owner_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            execution_guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
            )
            if await transaction.guard_record(
                owner,
                expected_storage_revision=owner_record.storage_revision,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            sequence = await transaction.next_sequence(stream)
            await transaction.insert_fact(StoredFact(stream, sequence, owner, fact_kind, None, kind, data))
            if isinstance(value, StepEvent):
                run = _decode_step(owner_record.data)
                if not isinstance(run, AgentRunRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self.index_history_events_in_transaction(transaction, run, ((sequence, value),))

        await self._mutate(mutate)

    async def _facts(
        self,
        agent_run_id: str,
        family: str,
        *,
        subject: str | None = None,
        latest: bool = False,
        latest_per_subject: bool = False,
    ) -> tuple[StoredFact, ...]:
        require_no_run_history_lock("StateStepArchive._facts")
        if latest and latest_per_subject:
            raise ValueError("latest and latest_per_subject cannot both be set")
        return await self._store.read(
            lambda transaction: (self.read_event_facts if family == "event" else lambda tx, query: tx.list_facts(query))(
                transaction, FactQuery(
                    self._stream(agent_run_id, family), subject_id=subject,
                    latest=latest, latest_per_subject=latest_per_subject,
                ),
            )
        )

    def _history_association_key(self, agent_run_id: str, family: str, identity: str) -> bytes:
        return record_key_digest(
            self._namespace, self._tenant_id, self._runtime_domain.value,
            "history_association", [agent_run_id, family, identity],
        )

    def _history_association_records(
        self, agent_run_id: str, events: Sequence[tuple[int, StepEvent]],
    ) -> dict[bytes, StoredRecord]:
        derived: dict[bytes, StoredRecord] = {}
        for sequence, event in events:
            locators: list[tuple[str, str]] = (
                [("trace", str(sequence))]
                if event.event_type.startswith(("MODEL_REQUEST_", "TOOL_CALL_")) else []
            )
            request = event.metadata.get("linktools.ai.model_request_seq")
            message = event.metadata.get("linktools.ai.message_seq")
            if event.event_type.startswith("MODEL_REQUEST_") and request is not None:
                locators.append((f"request:{event.event_type}", request))
            if event.event_type == "MODEL_REQUEST_SUCCEEDED" and message is not None:
                locators.append(("response", message))
            if (event.event_type in {"TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED"}
                    and event.tool_call_id is not None
                    and not (event.event_type == "TOOL_CALL_FAILED"
                             and event.metadata.get(TOOL_ERROR_CODE_METADATA_KEY) == ErrorCode.TOOL_EFFECT_UNKNOWN.value)):
                locators.append((event.event_type, event.tool_call_id))
            for family, identity in locators:
                key = self._history_association_key(agent_run_id, family, identity)
                record = StoredRecord(
                    key, None, self._agent_run_key(agent_run_id), "history_association",
                    (f"trace:{event.timestamp.astimezone(timezone.utc).isoformat(timespec='microseconds')}:{sequence:020d}"
                     if family == "trace" else f"{sequence:020d}").encode("ascii"),
                    None, 1, None, 0, None, {"sequence": sequence},
                )
                previous = derived.get(key)
                if previous is not None and previous.data != record.data:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                derived[key] = record
        return derived

    def _history_association_coverage(self, agent_run_id: str, record: StoredRecord | None) -> int:
        if record is None:
            return 0
        sequence = record.data.get("sequence")
        if (record.key_digest != self._history_association_key(agent_run_id, "coverage", "event")
                or record.kind != "history_association" or record.sort_key != b"coverage:event"
                or record.parent_key != self._agent_run_key(agent_run_id)
                or set(record.data) != {"sequence"} or isinstance(sequence, bool)
                or not isinstance(sequence, int) or sequence < 1):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return sequence

    async def index_history_events_in_transaction(
        self, transaction: StateTransaction, run: AgentRunRecord,
        events: Sequence[tuple[int, StepEvent]],
    ) -> None:
        """Derive event locators and certify only a continuously indexed prefix."""
        if self._runtime_domain is not PersistenceDomain.TRACE or not events:
            return
        derived = self._history_association_records(run.agent_run_id, events)
        coverage_key = self._history_association_key(run.agent_run_id, "coverage", "event")
        existing = await transaction.get_records((*derived, coverage_key))
        coverage_record = existing.get(coverage_key)
        coverage = self._history_association_coverage(run.agent_run_id, coverage_record)
        inserts: list[StoredRecord] = []
        for key, record in derived.items():
            previous = existing.get(key)
            if previous is None:
                inserts.append(record)
            elif previous.data != record.data:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sequences = tuple(sequence for sequence, _ in events)
        if sequences != tuple(range(coverage + 1, coverage + 1 + len(events))):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if coverage_record is None:
            inserts.append(StoredRecord(
                coverage_key, None, self._agent_run_key(run.agent_run_id), "history_association",
                b"coverage:event", None, 1, None, 0, None, {"sequence": sequences[-1]},
            ))
        elif not await transaction.replace_record(
            replace(coverage_record, storage_revision=coverage_record.storage_revision + 1,
                    data={"sequence": sequences[-1]}),
            expected_storage_revision=coverage_record.storage_revision,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if inserts:
            await transaction.insert_records(tuple(inserts))

    async def _captured_history_events(
        self, transaction: StateTransaction, agent_run_id: str, event_high_water: int,
    ) -> list[tuple[int, StepEvent]]:
        facts = await self.read_event_facts(transaction, FactQuery(
            self._stream(agent_run_id, "event"), after_sequence=0, limit=event_high_water,
        )) if event_high_water else ()
        if len(facts) != event_high_water:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events: list[tuple[int, StepEvent]] = []
        for sequence, fact in enumerate(facts, 1):
            event = _decode_step(fact.data)
            if fact.sequence != sequence or not isinstance(event, StepEvent) or event.agent_run_id != agent_run_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            events.append((sequence, event))
        return events

    async def read_history_associations(
        self, *, agent_run_id: str, message_seqs: Sequence[int],
        tool_call_ids: Sequence[str], event_high_water: int,
    ) -> list[StepEvent]:
        """Resolve bounded immutable event references at a captured event cutoff."""
        require_no_run_history_lock("StateStepArchive.read_history_associations")
        keys = tuple(dict.fromkeys((
            *(self._history_association_key(agent_run_id, "response", str(value)) for value in message_seqs),
            *(self._history_association_key(agent_run_id, kind, value)
              for value in tool_call_ids
              for kind in ("TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED")),
        )))
        if not keys:
            return []

        async def read(transaction: StateTransaction) -> list[StepEvent]:
            selected: dict[int, StepEvent] = {}
            coverage_key = self._history_association_key(agent_run_id, "coverage", "event")
            records = await transaction.get_records((*keys, coverage_key))
            coverage = self._history_association_coverage(agent_run_id, records.get(coverage_key))
            if coverage < event_high_water:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            async def resolve(
                index_keys: Sequence[bytes], indexed: Mapping[bytes, StoredRecord] | None = None,
            ) -> None:
                if indexed is None:
                    indexed = await transaction.get_records(index_keys)
                locators: list[tuple[bytes, StoredRecord, int]] = []
                for key in index_keys:
                    record = indexed.get(key)
                    if record is None:
                        continue
                    sequence = record.data.get("sequence")
                    if record.kind != "history_association" or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if sequence > event_high_water:
                        continue
                    locators.append((key, record, sequence))
                missing = sorted({sequence for _, _, sequence in locators if sequence not in selected})
                for offset in range(0, len(missing), 32):
                    facts = await self._read_selected_history_facts(transaction, agent_run_id, missing[offset:offset + 32])
                    for sequence, fact in facts.items():
                        event = _decode_step(fact.data)
                        if not isinstance(event, StepEvent) or event.agent_run_id != agent_run_id:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        selected[sequence] = event
                    del facts, fact
                for key, record, sequence in locators:
                    event = selected[sequence]
                    if self._history_association_records(agent_run_id, ((sequence, event),)).get(key) != record:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            await resolve(keys, records)
            requests = tuple(dict.fromkeys(
                event.metadata["linktools.ai.model_request_seq"]
                for event in selected.values()
                if "linktools.ai.model_request_seq" in event.metadata
            ))
            await resolve(tuple(
                self._history_association_key(agent_run_id, f"request:{kind}", value)
                for value in requests
                for kind in ("MODEL_REQUEST_STARTED", "MODEL_REQUEST_SUCCEEDED", "MODEL_REQUEST_FAILED", "MODEL_REQUEST_CANCELLED")
            ))
            return [selected[sequence] for sequence in sorted(selected)]

        return await self._store.read(read)

    async def _read_selected_history_facts(
        self, transaction: StateTransaction, agent_run_id: str, ordered: Sequence[int],
    ) -> dict[int, StoredFact]:
        """Read a sorted window of selected identities without including gaps."""
        if isinstance(transaction, FactReader):
            facts = await self.read_event_facts(transaction, FactQuery(self._stream(agent_run_id, "event")), sequences=ordered)
            if [fact.sequence for fact in facts] != list(ordered):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return {fact.sequence: fact for fact in facts}
        selected: dict[int, StoredFact] = {}
        offset = 0
        while offset < len(ordered):
            end = offset + 1
            while end < len(ordered) and ordered[end] == ordered[end - 1] + 1:
                end += 1
            facts = await transaction.list_facts(FactQuery(
                self._stream(agent_run_id, "event"),
                after_sequence=ordered[offset] - 1, limit=end - offset,
            ))
            if [fact.sequence for fact in facts] != list(ordered[offset:end]):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            selected.update((fact.sequence, fact) for fact in facts)
            offset = end
        return selected

    async def list_trace_events(
        self, *, agent_run_id: str, event_high_water: int,
        after_timestamp: datetime | None, after_sequence: int, limit: int,
        model_request_seq: int | None = None, step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> list[tuple[int, StepEvent]]:
        """Read immutable events in timestamp order using their derived locators."""
        require_no_run_history_lock("StateStepArchive.list_trace_events")
        if event_high_water < 0 or after_sequence < 0 or limit < 1:
            raise AIError(ErrorCode.CURSOR_INVALID)
        after_sort_key = (
            None if after_timestamp is None else
            f"trace:{after_timestamp.astimezone(timezone.utc).isoformat(timespec='microseconds')}:{after_sequence:020d}".encode("ascii")
        )

        if event_high_water == 0:
            return []
        filtered = any(value is not None for value in (model_request_seq, step_index, tool_call_id))

        def matches(sequence: int, event: StepEvent) -> bool:
            if not event.event_type.startswith(("MODEL_REQUEST_", "TOOL_CALL_")):
                return False
            request = event.metadata.get("linktools.ai.model_request_seq")
            if request is not None and (not request.isdigit() or int(request) < 1):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return not (
                model_request_seq is not None and (request is None or int(request) != model_request_seq)
                or step_index is not None and event.step_index != step_index
                or tool_call_id is not None and event.tool_call_id != tool_call_id
                or after_timestamp is not None and (event.timestamp, sequence) <= (after_timestamp, after_sequence)
            )

        async def scan_captured(transaction: StateTransaction) -> list[tuple[int, StepEvent]]:
            events = await self._captured_history_events(transaction, agent_run_id, event_high_water)
            expected = self._history_association_records(agent_run_id, events)
            records = await transaction.get_records(tuple(
                key for key, record in expected.items() if record.sort_key.startswith(b"trace:")
            ))
            for key, record in expected.items():
                if record.sort_key.startswith(b"trace:") and records.get(key) != record:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result: list[tuple[int, StepEvent]] = []
            for sequence, event in events:
                if matches(sequence, event):
                    result.append((sequence, event))
            result.sort(key=lambda value: (value[1].timestamp, value[0]))
            return result[:limit]

        async def read(transaction: StateTransaction) -> list[tuple[int, StepEvent]]:
            coverage_record = await transaction.get_record(
                self._history_association_key(agent_run_id, "coverage", "event"),
            )
            coverage = self._history_association_coverage(agent_run_id, coverage_record)
            if coverage < event_high_water:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if filtered:
                return await scan_captured(transaction)
            records = await transaction.list_records(RecordQuery(
                parent_key=self._agent_run_key(agent_run_id), kind="history_association",
                sort_key_prefix=b"trace:", after_sort_key=after_sort_key,
                limit=limit,
            ))
            sequences: list[int] = []
            for record in records:
                sequence = record.data.get("sequence")
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if sequence > event_high_water:
                    return await scan_captured(transaction)
                sequences.append(sequence)
            result: list[tuple[int, StepEvent]] = []
            start = 0
            while start < len(sequences):
                window_end = start
                while window_end < len(sequences):
                    end = window_end + 1
                    while (end < len(sequences) and end - window_end < 32
                           and sequences[end] == sequences[end - 1] + 1):
                        end += 1
                    # Do not split an already contiguous read to fill a window.
                    if end - start > 32:
                        break
                    window_end = end
                window = sequences[start:window_end]
                ordered = sorted(set(window))
                selected = await self._read_selected_history_facts(transaction, agent_run_id, ordered)
                for index, sequence in enumerate(window, start):
                    record, fact = records[index], selected[sequence]
                    event = _decode_step(fact.data)
                    if (not isinstance(event, StepEvent) or event.agent_run_id != agent_run_id
                            or record.sort_key != f"trace:{event.timestamp.astimezone(timezone.utc).isoformat(timespec='microseconds')}:{sequence:020d}".encode("ascii")
                            or not matches(sequence, event)):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    result.append((sequence, event))
                del selected, fact
                start = window_end
            return result

        return await self._store.read(read)

    async def capture_history(
        self, agent_run_ids: Sequence[str], *, include_pending: bool = False,
    ) -> Mapping[str, AgentRunHistoryCapture]:
        """Read run identities and all observable high waters in one snapshot."""
        require_no_run_history_lock("StateStepArchive.capture_history")
        self._ensure_open()
        if self._runtime_domain is not PersistenceDomain.TRACE:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        run_ids = tuple(dict.fromkeys(agent_run_ids))
        if not run_ids:
            return {}
        keys = tuple(key for run_id in run_ids for key in (
            self._agent_run_key(run_id), self._history.head_key(run_id),
        ))
        sequence_keys = tuple(key for run_id in run_ids for key in (
            self._stream(run_id, "event"), self._stream(run_id, "interaction"),
        ))

        pending_loaders = {}

        async def read(transaction: StateTransaction) -> Mapping[str, AgentRunHistoryCapture]:
            records = await self.read_run_records(transaction, keys)
            sequences = await transaction.get_sequences(sequence_keys)
            captured: dict[str, AgentRunHistoryCapture] = {}
            for run_id in run_ids:
                owner = records.get(self._agent_run_key(run_id))
                head_record = records.get(self._history.head_key(run_id))
                event_count = sequences.get(self._stream(run_id, "event"), 0)
                interaction_count = sequences.get(self._stream(run_id, "interaction"), 0)
                if owner is None:
                    if head_record is not None or event_count or interaction_count:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    captured[run_id] = AgentRunHistoryCapture(None, 0, 0, 0)
                    continue
                run = _decode_step(owner.data)
                if not isinstance(run, AgentRunRecord) or run.agent_run_id != run_id or head_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                head = self._history.decode_head(head_record)
                pending_parts, pending_keys = (
                    await self._history.capture_pending_in_transaction(transaction, head)
                    if include_pending else ((), ())
                )
                captured[run_id] = AgentRunHistoryCapture(
                    run, event_count, head.message_count, interaction_count,
                    head.pending, pending_keys, pending_parts,
                )
                if include_pending and head.pending is not None:
                    pending_loaders[run_id] = await self._history.capture_pending_message(head.pending, pending_parts)
            return captured

        captured = await self._history.read_content(lambda: self._store.read(read)) if include_pending else await self._store.read(read)
        return {run_id: (replace(value, pending_message=await pending_loaders[run_id]())
                         if run_id in pending_loaders else value) for run_id, value in captured.items()}

    async def read_pending_message(self, capture: AgentRunHistoryCapture) -> ModelMessage | None:
        """Resolve the immutable tail body retained by a committed capture."""
        require_no_run_history_lock("StateStepArchive.read_pending_message")
        if capture.pending_message is not None or capture.pending is None:
            return capture.pending_message
        return await self._history.read_pending_message(capture.pending, capture.pending_parts)

    async def list_event_range(
        self, *, agent_run_id: str, start: int, end: int, verify_native_source: bool = False,
    ) -> list[StepEvent]:
        """Read only the requested zero-based half-open committed event range."""
        require_no_run_history_lock("StateStepArchive.list_event_range")
        if start < 0 or end < start:
            raise AIError(ErrorCode.CURSOR_INVALID)
        if start == end:
            return []
        facts = await self._store.read(lambda transaction: self.read_event_facts(transaction, FactQuery(
            self._stream(agent_run_id, "event"), after_sequence=start, limit=end - start,
        ), full_source=verify_native_source))
        events: list[StepEvent] = []
        for sequence, fact in enumerate(facts, start + 1):
            event = _decode_step(fact.data)
            if fact.sequence != sequence or not isinstance(event, StepEvent) or event.agent_run_id != agent_run_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            events.append(event)
        if len(events) != end - start:
            raise AIError(ErrorCode.CURSOR_INVALID)
        return events

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)


def _validate_interaction_page(
    after_model_request_seq: int | None,
    limit: int | None,
) -> None:
    if after_model_request_seq is not None and (
        isinstance(after_model_request_seq, bool)
        or not isinstance(after_model_request_seq, int)
        or after_model_request_seq < 0
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
        or observed.agent_id != source.agent_id
        or observed.timestamp != source.timestamp
        or observed.state != source.state
        or observed.idempotency_key != source.idempotency_key
        or observed.pending_request_index != source.pending_request_index
        or observed.output_retries_used != source.output_retries_used
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
    checkpoints: Sequence[AgentRunCheckpoint],
    interactions: tuple[ModelInteractionRecord, ...] = (),
    *,
    execution_id: str | None = None,
) -> None:
    if type(target) is not StateStepArchive:
        checkpoints = tuple(replace(value, messages=list(value.messages), context_messages=(
            None if value.context_messages is None else list(value.context_messages))) for value in checkpoints)
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
    if type(target) is not StateStepArchive:
        checkpoint = replace(checkpoint, messages=list(checkpoint.messages), context_messages=(
            None if checkpoint.context_messages is None else list(checkpoint.context_messages)))
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


def _step_event_type(value: StepEvent) -> str:
    return str(value.event_type)


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
