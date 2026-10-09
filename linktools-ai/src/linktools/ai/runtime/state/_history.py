#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical transcript chunks and bounded context projections."""

import hashlib
import zlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from linktools.core import environ
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, StoredPayload, read_object
from .._storage_keys import runtime_object_key
from .._message import decode_model_messages, encode_model_messages
from ._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_envelope,
    encode_envelope,
)
from ._contracts import (
    ContextProjection,
    ConversationHistoryRepository,
    HistoryQuality,
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    RuntimePayloadRef,
    TranscriptChunk,
    TranscriptHeadRecord,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptOwnerDomain,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
    TranscriptSpanRef,
)
from ._history_index import (
    resolve_history_range_lazy,
)
from ._plan import RuntimeDomain
from ._store import (
    FactQuery,
    RecordQuery,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    active_state_scope,
    record_key_digest,
    require_no_run_history_lock,
    sequence_key,
    sortable_identity,
    stream_digest,
)

_logger = environ.get_logger("ai.runtime.state.history")
_CHUNK_TARGET = 256 * 1024
_COMPRESS_MINIMUM = 16 * 1024
_COMPRESS_RATIO = 0.9
_TRANSCRIPT_PAGE_SIZE = 64
_TRANSCRIPT_CHUNK_MAX_MESSAGES = 64
_TRANSCRIPT_SEEK_BLOCK = 128


def _exact_message_signature(message: ModelMessage) -> bytes:
    """Full canonical serialization including timestamp, for exact-content proof."""
    return encode_model_messages((message,))


@dataclass(frozen=True, slots=True)
class TranscriptCapture:
    first_message_index: int
    messages: tuple[ModelMessage, ...]
    origins: tuple[TranscriptOrigin, ...]
    quality: HistoryQuality


@dataclass(frozen=True, slots=True)
class _PendingPart:
    key: str
    source_part: object
    chunk: TranscriptChunk


@dataclass(frozen=True, slots=True)
class PreparedTranscriptObservation:
    owner_id: str
    base_message_count: int
    target_message_count: int
    chunks: tuple[TranscriptChunk, ...]
    base_storage_version: int
    pending: RuntimePayloadRef | None
    pending_parts: tuple[_PendingPart, ...]
    new_pending_parts: tuple[StoredFact, ...]


@dataclass(frozen=True, slots=True)
class _PendingCache:
    message_index: int
    storage_version: int
    pending: RuntimePayloadRef | None
    parts: tuple[_PendingPart, ...]


@dataclass(frozen=True, slots=True)
class _HistorySegment:
    history_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _HistoryResolution:
    segments: tuple[_HistorySegment, ...]


class _ContextProjector:
    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        self._runtime_domain = runtime_domain

    def project(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        *,
        origins: Sequence[TranscriptOrigin] = (),
        sources: Sequence[TranscriptMessageRef | None] = (),
    ) -> ContextProjection:
        values = tuple(messages)
        origin_values = tuple(origins)
        source_values = tuple(sources)
        items: list[TranscriptSpanRef | InlineContextBlock] = []
        index = 0
        while index < len(values):
            origin = (
                origin_values[index]
                if index < len(origin_values)
                else TranscriptOrigin.RAW
            )
            source = source_values[index] if index < len(source_values) else None
            end = index + 1
            while end < len(values):
                next_origin = (
                    origin_values[end]
                    if end < len(origin_values)
                    else TranscriptOrigin.RAW
                )
                next_source = source_values[end] if end < len(source_values) else None
                if (
                    next_origin is not TranscriptOrigin.RAW
                    or source is None
                    or next_source is None
                    or next_source.source_domain is not source.source_domain
                    or next_source.owner_id != source.owner_id
                    or next_source.message_index != source.message_index + (end - index)
                ):
                    break
                end += 1
            if source is not None and origin is TranscriptOrigin.RAW:
                items.append(
                    TranscriptSpanRef(
                        source.source_domain,
                        source.owner_id,
                        source.message_index,
                        source.message_index + (end - index),
                    )
                )
            else:
                raw = encode_model_messages(values[index:end])
                items.append(
                    InlineContextBlock(
                        RuntimePayloadRef(
                            StoredPayload.inline_bytes(raw),
                            self._runtime_domain,
                        )
                    )
                )
            index = end
        return ContextProjection(tuple(items))


class TranscriptRepository:
    """Persist lossless transcript chunks and read the active context view."""

    def __init__(
        self,
        store: StateStore,
        *,
        object_store: ObjectStore | None,
        namespace: str,
        tenant_id: str,
        runtime_domain: RuntimeDomain,
        context_sources: Mapping[RuntimeDomain, "TranscriptRepository"] | None = None,
        history_repository: "ConversationHistoryRepository | None" = None,
    ) -> None:
        self._store = store
        self._object_store = object_store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self._tenant_digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        self._runtime_domain = runtime_domain
        self._context_sources = dict(context_sources or {})
        self._history_repository = history_repository
        self._projector = _ContextProjector(runtime_domain)
        self._pending_cache: dict[str, _PendingCache] = {}

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    @property
    def _owner_domain(self) -> TranscriptOwnerDomain:
        try:
            return TranscriptOwnerDomain(self._runtime_domain.value)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def get_head(self, owner_id: str) -> TranscriptHeadRecord | None:
        """Read the typed transcript head without legacy fallback."""
        require_no_run_history_lock("TranscriptRepository.get_head")
        stored = await self._store.read(
            lambda transaction: transaction.get_record(self._head_key(owner_id))
        )
        return None if stored is None else self._decode_head(stored)

    async def create_head(self, owner_id: str) -> TranscriptHeadRecord:
        """Create an empty transcript head as part of owner admission."""
        require_no_run_history_lock("TranscriptRepository.create_head")
        return await self._store.mutate(
            lambda transaction: self.create_head_in_transaction(transaction, owner_id)
        )

    def empty_head(self, owner_id: str) -> TranscriptHeadRecord:
        """Return the empty baseline used by a first-write prepare."""
        return TranscriptHeadRecord(
            self._owner_domain,
            owner_id,
            0,
            0,
            HistoryQuality.COMPLETE,
        )

    def empty_head_record(self, owner_id: str) -> StoredRecord:
        """Return the canonical stored record for an empty transcript head."""
        return self._new_head_record(self.empty_head(owner_id))

    async def create_head_in_transaction(
        self,
        transaction: StateTransaction,
        owner_id: str,
    ) -> TranscriptHeadRecord:
        key = self._head_key(owner_id)
        existing = await transaction.get_record(key)
        if existing is not None:
            return self._decode_head(existing)
        head = self.empty_head(owner_id)
        await transaction.insert_record(self.empty_head_record(owner_id))
        return head

    async def get_head_in_transaction(
        self,
        transaction: StateTransaction,
        owner_id: str,
    ) -> tuple[TranscriptHeadRecord, StoredRecord] | None:
        stored = await transaction.get_record(self._head_key(owner_id))
        return None if stored is None else (self._decode_head(stored), stored)

    def _decode_head(self, record: StoredRecord) -> TranscriptHeadRecord:
        if record.kind != "transcript_head":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            head = _decode_enveloped_domain(record.data, TranscriptHeadRecord)
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if head.owner_domain is not self._owner_domain:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return head

    def decode_head(self, record: StoredRecord) -> TranscriptHeadRecord:
        """Decode one stored typed head for batched metadata reads."""
        return self._decode_head(record)

    def _new_head_record(self, head: TranscriptHeadRecord) -> StoredRecord:
        key = self._head_key(head.owner_id)
        return StoredRecord(
            key,
            None,
            None,
            "transcript_head",
            sortable_identity(head.owner_id),
            None,
            0,
            None,
            0,
            None,
            encode_envelope(
                {"type": "transcript_head", "payload": _encode_persisted_domain(head)}
            ),
        )

    async def capture_pending_in_transaction(
        self,
        transaction: StateTransaction,
        head: TranscriptHeadRecord,
    ) -> tuple[tuple[TranscriptChunk, ...], tuple[str, ...]]:
        return await self._capture_pending_suffix(transaction, head, ())

    async def _capture_pending_suffix(
        self,
        transaction: StateTransaction,
        head: TranscriptHeadRecord,
        known_keys: tuple[str, ...],
    ) -> tuple[tuple[TranscriptChunk, ...], tuple[str, ...]]:
        chunks: list[TranscriptChunk] = []
        keys: list[str] = []
        seen_keys = set(known_keys)
        start = len(known_keys)
        while start + len(chunks) < head.pending_part_count:
            facts = await transaction.list_facts(FactQuery(
                self._pending_stream(head.owner_id, head.message_count),
                after_sequence=start + len(chunks),
                limit=min(_TRANSCRIPT_PAGE_SIZE, head.pending_part_count - start - len(chunks)),
            ))
            if not facts:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for fact in facts:
                if (
                    fact.kind != "transcript_pending_part"
                    or fact.owner_key_digest != self._head_key(head.owner_id)
                    or fact.sequence != start + len(chunks) + 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                key = decode_envelope(fact.data).value.get("pending_key")
                self._require_pending_key(key)
                chunk = self.decode_chunk(fact)
                if (
                    not isinstance(key, str) or not key or key in seen_keys
                    or chunk.owner_id != head.owner_id
                    or chunk.first_message_index != head.message_count
                    or chunk.message_count != 1
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                chunks.append(chunk)
                keys.append(key)
                seen_keys.add(key)
        return tuple(chunks), tuple(keys)

    async def read_pending_message(
        self,
        pending: RuntimePayloadRef,
        parts: Sequence[TranscriptChunk],
    ) -> ModelMessage:
        if pending.source_domain is not self._runtime_domain:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        shells = decode_model_messages(await self._read_payload(pending.payload))
        if len(shells) != 1 or shells[0].parts:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        shell = shells[0]
        values = []
        for chunk in parts:
            messages = await self._decode_chunk_messages(chunk)
            if len(messages) != 1 or type(messages[0]) is not type(shell) or len(messages[0].parts) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.extend(messages[0].parts)
        return replace(shell, parts=values)

    async def load_pending(
        self,
        head: TranscriptHeadRecord,
    ) -> tuple[ModelMessage | None, tuple[str, ...]]:
        if head.pending is None:
            return None, ()
        parts, keys = await self._store.read(
            lambda transaction: self.capture_pending_in_transaction(transaction, head)
        )
        return await self.read_pending_message(head.pending, parts), keys

    @classmethod
    def _part_message(cls, message: ModelMessage, part: object) -> ModelMessage:
        timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(message, ModelRequest):
            return ModelRequest(parts=[part], timestamp=timestamp)
        return ModelResponse(parts=[part], timestamp=timestamp)

    @classmethod
    def _require_pending_key(cls, key: object) -> str:
        if isinstance(key, str):
            kind, separator, suffix = key.partition(":")
            if separator and suffix and (
                kind in {"tool_result", "retry"}
                or kind == "part" and suffix.isascii() and suffix.isdigit()
                and (suffix == "0" or not suffix.startswith("0"))
            ):
                return key
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @classmethod
    def _completed_parts(cls, message: ModelMessage) -> dict[str, object]:
        if isinstance(message, ModelResponse):
            return {f"part:{index}": part for index, part in enumerate(message.parts)}
        result: dict[str, object] = {}
        for part in message.parts:
            kind = "tool_result" if part.part_kind == "tool-return" else "retry" if part.part_kind == "retry-prompt" else None
            if kind is not None and part.tool_call_id is not None:
                key = f"{kind}:{part.tool_call_id}"
                if key in result:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result[key] = part
        return result

    async def prepare_observation(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        *,
        first_message_index: int,
        pending: ModelMessage | None,
        pending_keys: tuple[str, ...],
    ) -> PreparedTranscriptObservation:
        require_no_run_history_lock("TranscriptRepository.prepare_observation")
        scope = active_state_scope()
        committed_read = scope is None or not scope.writable

        async def capture(transaction: StateTransaction):
            entry = await self.get_head_in_transaction(transaction, owner_id)
            head = self.empty_head(owner_id) if entry is None else entry[0]
            version = -1 if entry is None else entry[1].storage_version
            cache = self._pending_cache.get(owner_id)
            previous = ()
            if (cache is not None and cache.message_index == head.message_count
                    and cache.storage_version <= version and cache.pending == head.pending
                    and len(cache.parts) <= head.pending_part_count):
                previous = cache.parts
            chunks, keys = await self._capture_pending_suffix(
                transaction, head, tuple(part.key for part in previous),
            )
            return head, version, previous, chunks, keys

        head, base_version, cached, previous_chunks, previous_keys = await self._store.read(capture)
        if head.message_count != first_message_index:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        loaded = list(cached)
        for key, chunk in zip(previous_keys, previous_chunks, strict=True):
            values = await self._decode_chunk_messages(chunk)
            if len(values) != 1 or len(values[0].parts) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            loaded.append(_PendingPart(key, values[0].parts[0], chunk))
        previous = tuple(loaded)
        values = tuple(messages)
        if values and previous:
            completed_parts = self._completed_parts(values[0])
            for part in previous:
                if part.key not in completed_parts:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                final = self._part_message(values[0], completed_parts[part.key])
                encoded = _exact_message_signature(final)
                if len(encoded) != part.chunk.raw_size or hashlib.sha256(encoded).hexdigest() != part.chunk.raw_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous = ()
        if pending is None:
            if pending_keys or previous:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            for key in pending_keys:
                self._require_pending_key(key)
            if len(pending.parts) != len(pending_keys) or len(set(pending_keys)) != len(pending_keys) or not pending_keys:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        chunks = await self.prepare_chunks(owner_id, values, first_message_index=first_message_index)
        target = first_message_index + len(values)
        shell = None
        next_parts: list[_PendingPart] = []
        additions: list[StoredFact] = []
        if pending is not None:
            if len(pending_keys) < len(previous):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            shell = head.pending if previous else RuntimePayloadRef(
                StoredPayload.inline_bytes(encode_model_messages((replace(pending, parts=[]),))),
                self._runtime_domain,
            )
            for index, (key, part) in enumerate(zip(pending_keys, pending.parts, strict=True)):
                if key.startswith("part:"):
                    if not isinstance(pending, ModelResponse):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                else:
                    kind, _, call_id = key.partition(":")
                    expected = "tool-return" if kind == "tool_result" else "retry-prompt"
                    if not isinstance(pending, ModelRequest) or part.part_kind != expected or part.tool_call_id != call_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if index < len(previous):
                    existing = previous[index]
                    if existing.key != key:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if existing.source_part is not part:
                        candidate = self._part_message(pending, part)
                        encoded = _exact_message_signature(candidate)
                        if len(encoded) != existing.chunk.raw_size or hashlib.sha256(encoded).hexdigest() != existing.chunk.raw_digest:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    next_parts.append(_PendingPart(key, part, existing.chunk))
                    continue
                message = self._part_message(pending, part)
                chunk = await self._make_chunk(owner_id, target, (message,), TranscriptOrigin.RAW)
                next_parts.append(_PendingPart(key, part, chunk))
                additions.append(StoredFact(
                    self._pending_stream(owner_id, target), index + 1,
                    self._head_key(owner_id), "transcript_pending_part", None, None,
                    encode_envelope({
                        "type": "transcript_chunk", "payload": _encode_persisted_domain(chunk),
                        "pending_key": key,
                    }),
                ))
        if committed_read:
            # Only the prefix read from durable facts is safe to reuse after rollback.
            if previous:
                self._pending_cache[owner_id] = _PendingCache(
                    head.message_count, base_version, head.pending,
                    tuple(next_parts[:len(previous)]),
                )
            else:
                self._pending_cache.pop(owner_id, None)
        return PreparedTranscriptObservation(
            owner_id, first_message_index, target, chunks, base_version,
            shell, tuple(next_parts), tuple(additions),
        )

    async def commit_observation(
        self,
        transaction: StateTransaction,
        prepared: PreparedTranscriptObservation,
    ) -> None:
        entry = await self.get_head_in_transaction(transaction, prepared.owner_id)
        if entry is None:
            if prepared.base_storage_version != -1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await self.create_head_in_transaction(transaction, prepared.owner_id)
            entry = await self.get_head_in_transaction(transaction, prepared.owner_id)
        if entry is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head, record = entry
        if (
            record.storage_version != max(0, prepared.base_storage_version)
            or head.message_count != prepared.base_message_count
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if (
            not prepared.chunks and not prepared.new_pending_parts
            and head.pending == prepared.pending
            and head.pending_part_count == len(prepared.pending_parts)
        ):
            guarded = await transaction.guard_record(
                record.key_digest, expected_storage_version=record.storage_version,
            )
            if guarded is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return
        await self.append_chunks(transaction, prepared.owner_id, prepared.chunks)
        entry = await self.get_head_in_transaction(transaction, prepared.owner_id)
        if entry is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head, record = entry
        next_head = replace(head, pending=prepared.pending, pending_part_count=len(prepared.pending_parts))
        upgraded = replace(
            record,
            data=encode_envelope({"type": "transcript_head", "payload": _encode_persisted_domain(next_head)}),
            storage_version=record.storage_version + 1,
        )
        if not await transaction.replace_record(upgraded, expected_storage_version=record.storage_version):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if prepared.new_pending_parts:
            await transaction.insert_facts(prepared.new_pending_parts)

    async def verify_observation(self, prepared: PreparedTranscriptObservation) -> bool:
        head = await self.get_head(prepared.owner_id)
        if head is None or head.message_count < prepared.target_message_count:
            return False
        if head.message_count == prepared.target_message_count and prepared.pending_parts:
            if head.pending is None or head.pending_part_count < len(prepared.pending_parts):
                return False
            if head.pending != prepared.pending:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for chunk in prepared.chunks:
            expected = await self._decode_chunk_messages(chunk)
            actual = tuple([value async for value in self.iter_message_range(
                prepared.owner_id, start=chunk.first_message_index,
                end=chunk.first_message_index + chunk.message_count,
            )])
            if encode_model_messages(actual) != encode_model_messages(expected):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if prepared.pending_parts and head.message_count > prepared.target_message_count:
            messages = tuple([value async for value in self.iter_message_range(
                prepared.owner_id, start=prepared.target_message_count,
                end=prepared.target_message_count + 1,
            )])
            if len(messages) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            completed = self._completed_parts(messages[0])
            for part in prepared.pending_parts:
                if part.key not in completed:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raw = _exact_message_signature(self._part_message(messages[0], completed[part.key]))
                if len(raw) != part.chunk.raw_size or hashlib.sha256(raw).hexdigest() != part.chunk.raw_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif prepared.pending_parts:
            cut = replace(head, pending_part_count=len(prepared.pending_parts))
            chunks, keys = await self._store.read(
                lambda transaction: self.capture_pending_in_transaction(transaction, cut)
            )
            if keys != tuple(part.key for part in prepared.pending_parts) or chunks != tuple(part.chunk for part in prepared.pending_parts):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return True

    async def prepare_chunks(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        *,
        first_message_index: int,
        origin: TranscriptOrigin = TranscriptOrigin.RAW,
    ) -> tuple[TranscriptChunk, ...]:
        require_no_run_history_lock("TranscriptRepository.prepare_chunks")
        values = tuple(messages)
        chunks: list[TranscriptChunk] = []
        current: list[ModelMessage] = []
        current_start = first_message_index
        current_size = 2
        for message in values:
            encoded = encode_model_messages((message,))
            part = encoded[1:-1]
            candidate_size = current_size + len(part) + (1 if current else 0)
            split = (
                current
                and (
                    candidate_size > _CHUNK_TARGET
                    or len(current) >= _TRANSCRIPT_CHUNK_MAX_MESSAGES
                )
            )
            if split:
                chunks.append(
                    await self._make_chunk(owner_id, current_start, current, origin)
                )
                current_start += len(current)
                current = [message]
                current_size = len(part) + 2
            else:
                current.append(message)
                current_size = candidate_size
        if current:
            chunks.append(await self._make_chunk(owner_id, current_start, current, origin))
        return tuple(chunks)

    async def _make_chunk(
        self,
        owner_id: str,
        first_message_index: int,
        messages: Sequence[ModelMessage],
        origin: TranscriptOrigin,
    ) -> TranscriptChunk:
        raw = encode_model_messages(messages)
        raw_digest = hashlib.sha256(raw).hexdigest()
        content = raw
        codec = "raw"
        if len(raw) >= _COMPRESS_MINIMUM:
            compressed = zlib.compress(raw)
            if len(compressed) <= len(raw) * _COMPRESS_RATIO:
                content = compressed
                codec = "zlib"
        payload = await self._store_payload(
            content,
            raw_size=len(raw),
        )
        return TranscriptChunk(
            owner_id,
            first_message_index,
            len(messages),
            origin,
            codec,
            raw_digest,
            len(raw),
            RuntimePayloadRef(payload, self._runtime_domain),
        )

    async def _store_payload(
        self,
        value: bytes,
        *,
        raw_size: int,
    ) -> StoredPayload:
        require_no_run_history_lock("TranscriptRepository._store_payload")
        if self._object_store is None or raw_size < _COMPRESS_MINIMUM:
            return StoredPayload.inline_bytes(value)
        key = runtime_object_key(
            namespace_digest=self._namespace_digest,
            tenant_digest=self._tenant_digest,
            stored_digest=hashlib.sha256(value).hexdigest(),
        )

        async def chunks() -> AsyncIterator[bytes]:
            yield value

        stat = await self._object_store.put(
            key,
            chunks(),
            expected_size=len(value),
            expected_digest=hashlib.sha256(value).hexdigest(),
        )
        return StoredPayload.object(
            ObjectRef("runtime", key, stat.digest, stat.size)
        )

    async def append_chunks(
        self,
        transaction: StateTransaction,
        owner_id: str,
        chunks: Sequence[TranscriptChunk],
        quality: HistoryQuality | None = None,
    ) -> None:
        if not chunks:
            return
        head_entry = await self.get_head_in_transaction(transaction, owner_id)
        if head_entry is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        base_head, head_record = head_entry
        stream = self._transcript_stream(owner_id)
        owner = self._head_key(owner_id)
        current_count = base_head.message_count
        expected = current_count
        for chunk in chunks:
            if (
                chunk.owner_id != owner_id
                or chunk.message_count <= 0
                or chunk.first_message_index != expected
            ):
                _logger.info(
                    "transcript append conflict: domain=%s owner=%s "
                    "expected_index=%s actual_index=%s",
                    self._runtime_domain.value,
                    owner_id,
                    expected,
                    chunk.first_message_index,
                )
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            expected += chunk.message_count
        next_head = replace(
            base_head,
            message_count=expected,
            chunk_count=base_head.chunk_count + len(chunks),
            quality=base_head.quality if quality is None else quality,
        )
        upgraded = replace(
            head_record,
            data=encode_envelope(
                {"type": "transcript_head", "payload": _encode_persisted_domain(next_head)}
            ),
            storage_version=head_record.storage_version + 1,
        )
        if not await transaction.replace_record(
            upgraded,
            expected_storage_version=head_record.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        final = await transaction.reserve_sequence(
            self._transcript_sequence(owner_id),
            len(chunks),
        )
        sequences = tuple(range(final - len(chunks) + 1, final + 1))
        await transaction.insert_facts(
            tuple(
                StoredFact(
                    stream,
                    sequence,
                    owner,
                    "transcript_chunk",
                    None,
                    chunk.origin.value,
                    encode_envelope(
                        {
                            "type": "transcript_chunk",
                            "payload": _encode_persisted_domain(chunk),
                        }
                    ),
                )
                for sequence, chunk in zip(sequences, chunks, strict=True)
            )
        )
        await self._insert_message_seek_boundaries(
            transaction,
            owner_id,
            chunks,
            sequences,
            base_head.message_count,
        )
        _logger.debug(
            "transcript chunks appended: domain=%s owner=%s "
            "first_index=%s message_count=%s chunks=%s",
            self._runtime_domain.value,
            owner_id,
            current_count,
            expected - current_count,
            len(chunks),
        )

    async def _insert_message_seek_boundaries(
        self,
        transaction: StateTransaction,
        owner_id: str,
        chunks: Sequence[TranscriptChunk],
        sequences: Sequence[int],
        base_message_count: int,
    ) -> None:
        candidates: dict[bytes, StoredRecord] = {}
        for chunk, sequence in zip(chunks, sequences, strict=True):
            start = chunk.first_message_index
            end = start + chunk.message_count
            boundary = (
                (max(start, base_message_count) + _TRANSCRIPT_SEEK_BLOCK - 1)
                // _TRANSCRIPT_SEEK_BLOCK
            ) * _TRANSCRIPT_SEEK_BLOCK
            for block_start in range(boundary, end, _TRANSCRIPT_SEEK_BLOCK):
                seek = TranscriptSeekRecord(
                    owner_id,
                    TranscriptSeekDimension.MESSAGE,
                    block_start,
                    sequence,
                    start,
                )
                key = self._seek_key(owner_id, block_start)
                candidates[key] = StoredRecord(
                    key,
                    None,
                    self._head_key(owner_id),
                    "transcript_seek",
                    f"b:{block_start:020d}",
                    None,
                    0,
                    None,
                    0,
                    None,
                    encode_envelope(
                        {
                            "type": "transcript_seek",
                            "payload": _encode_persisted_domain(seek),
                        }
                    ),
                )
        if not candidates:
            return
        existing = await transaction.get_records(tuple(candidates))
        fresh = tuple(
            record for key, record in candidates.items() if key not in existing
        )
        if fresh:
            await transaction.insert_records(fresh)

    async def prepare_projection(
        self,
        agent_run_id: str,
        projection: ContextProjection,
    ) -> ContextProjection:
        require_no_run_history_lock("TranscriptRepository.prepare_projection")
        items: list[TranscriptSpanRef | InlineContextBlock] = []
        changed = False
        for item in projection.items:
            if (
                not isinstance(item, InlineContextBlock)
                or item.content.payload.kind != "inline"
                or item.content.payload.size < _COMPRESS_MINIMUM
                or self._object_store is None
            ):
                items.append(item)
                continue
            raw = await self._read_payload(item.content.payload)
            stat = await self._object_store.put(
                runtime_object_key(
                    namespace_digest=self._namespace_digest,
                    tenant_digest=self._tenant_digest,
                    stored_digest=hashlib.sha256(raw).hexdigest(),
                ),
                _one_chunk(raw),
                expected_size=len(raw),
                expected_digest=item.content.payload.digest,
            )
            items.append(
                InlineContextBlock(
                    RuntimePayloadRef(
                        StoredPayload.object(
                            ObjectRef(
                                "runtime",
                                stat.key,
                                stat.digest,
                                stat.size,
                            )
                        ),
                        self._runtime_domain,
                    )
                )
            )
            changed = True
        if not changed:
            return projection
        return ContextProjection(tuple(items))

    def history_stream(self, history_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "history_transcript",
            history_id,
        )

    def agent_run_stream(self, agent_run_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "run_transcript",
            agent_run_id,
        )

    def transcript_stream(self, agent_run_id: str) -> bytes:
        """Return the owner stream used by this archive's domain."""
        if self._runtime_domain is RuntimeDomain.CONVERSATION:
            return self.history_stream(agent_run_id)
        return self.agent_run_stream(agent_run_id)

    async def latest_chunk(self, owner_id: str) -> TranscriptChunk | None:
        require_no_run_history_lock("TranscriptRepository.latest_chunk")
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(self._transcript_stream(owner_id), latest=True)
            )
        )
        if not values:
            return None
        return self.decode_chunk(values[0])

    async def iter_messages(self, owner_id: str) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock("TranscriptRepository.iter_messages")
        stream = self._transcript_stream(owner_id)
        after_sequence: int | None = None
        while True:
            values = await self._store.read(
                lambda transaction, sequence=after_sequence: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=sequence,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not values:
                return
            after_sequence = values[-1].sequence
            for fact in values:
                chunk = self.decode_chunk(fact)
                messages = await self._decode_chunk_messages(chunk)
                for message in messages:
                    yield message

    async def iter_message_range(
        self,
        owner_id: str,
        *,
        start: int,
        end: int,
    ) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock("TranscriptRepository.iter_message_range")
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        head = await self._resolve_observed_head(owner_id, None)
        if end > head.message_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async for message in self._iter_range(
            self._transcript_stream(owner_id),
            start=start,
            end=end,
            seek_owner_id=owner_id,
        ):
            yield message

    async def iter_raw_messages(self, owner_id: str) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock("TranscriptRepository.iter_raw_messages")
        stream = self._transcript_stream(owner_id)
        after_sequence: int | None = None
        while True:
            values = await self._store.read(
                lambda transaction, sequence=after_sequence: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=sequence,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not values:
                return
            after_sequence = values[-1].sequence
            for fact in values:
                chunk = self.decode_chunk(fact)
                if chunk.origin is not TranscriptOrigin.RAW:
                    continue
                messages = await self._decode_chunk_messages(chunk)
                for message in messages:
                    yield message

    async def _decode_chunk_messages(self, chunk: TranscriptChunk) -> tuple[ModelMessage, ...]:
        raw = await self._read_payload(chunk.content.payload)
        if chunk.codec == "zlib":
            try:
                raw = zlib.decompress(raw)
            except zlib.error as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        elif chunk.codec != "raw":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if hashlib.sha256(raw).hexdigest() != chunk.raw_digest or len(raw) != chunk.raw_size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return decode_model_messages(raw)

    async def load_messages(self, owner_id: str) -> tuple[ModelMessage, ...]:
        require_no_run_history_lock("TranscriptRepository.load_messages")
        return tuple([message async for message in self.iter_messages(owner_id)])

    async def validate_integrity(self) -> None:
        require_no_run_history_lock("TranscriptRepository.validate_integrity")
        after_sort: str | None = None
        after_key: bytes | None = None
        while True:
            page = await self._store.read(
                lambda transaction, sort_key=after_sort, key_digest=after_key: transaction.list_records(
                    RecordQuery(
                        kind="transcript_head",
                        after_sort_key=sort_key,
                        after_key_digest=key_digest,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not page:
                break
            for record in page:
                await self._validate_head_record(record)
            last = page[-1]
            after_sort = last.sort_key
            after_key = last.key_digest
        facts = await self._store.read(lambda transaction: transaction.scan_facts())
        for fact in facts:
            if fact.kind != "transcript_pending_part":
                continue
            chunk = self.decode_chunk(fact)
            messages = await self._decode_chunk_messages(chunk)
            if (
                len(messages) != 1 or len(messages[0].parts) != 1
                or _exact_message_signature(messages[0]) != _exact_message_signature(
                    self._part_message(messages[0], messages[0].parts[0])
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _validate_head_record(self, record: StoredRecord) -> None:
        head = self._decode_head(record)
        expected_message_index = 0
        expected_chunks = 0
        after_sequence: int | None = None
        while True:
            facts = await self._store.read(
                lambda transaction, sequence=after_sequence: transaction.list_facts(
                    FactQuery(
                        self._transcript_stream(head.owner_id),
                        after_sequence=sequence,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not facts:
                break
            after_sequence = facts[-1].sequence
            for fact in facts:
                if (
                    fact.kind != "transcript_chunk"
                    or fact.owner_key_digest != record.key_digest
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                chunk = self.decode_chunk(fact)
                if (
                    chunk.owner_id != head.owner_id
                    or chunk.first_message_index != expected_message_index
                    or chunk.message_count <= 0
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                messages = await self._decode_chunk_messages(chunk)
                if len(messages) != chunk.message_count:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                expected_message_index += chunk.message_count
                expected_chunks += 1
        if (
            expected_message_index != head.message_count
            or expected_chunks != head.chunk_count
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if head.pending is not None:
            await self.load_pending(head)
        current = await self._store.read(
            lambda transaction: transaction.get_record(record.key_digest)
        )
        if current is None or current.storage_version != record.storage_version:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def history_message_count(self, history_id: str, *, tenant_id: str) -> int:
        require_no_run_history_lock("TranscriptRepository.history_message_count")
        if self._history_repository is None:
            return 0
        record = await self._history_repository.get(
            history_id,
            tenant_id=tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        if record.inherited_message_count < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = await self.get_head(history_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return record.inherited_message_count + head.message_count

    async def transcript_message_count(self, owner_id: str) -> int:
        require_no_run_history_lock("TranscriptRepository.transcript_message_count")
        head = await self.get_head(owner_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return head.message_count

    async def load_session_model_context(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> LoadedModelContext:
        require_no_run_history_lock("TranscriptRepository.load_session_model_context")
        projection = await self.load_projection(history_id)
        if projection is not None:
            return await self._load_model_context_from_projection(
                history_id,
                projection,
            )
        head = await self.get_head(history_id)
        if head is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history = (
            None
            if self._history_repository is None
            else await self._history_repository.get(
                history_id,
                tenant_id=tenant_id,
            )
        )
        if history is None:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        if head.message_count or history.inherited_message_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return LoadedModelContext(())

    async def load_session_raw_model_context(
        self,
        history_id: str,
        *,
        tenant_id: str,
        message_count: int,
    ) -> LoadedModelContext:
        require_no_run_history_lock(
            "TranscriptRepository.load_session_raw_model_context"
        )
        if (
            isinstance(message_count, bool)
            or not isinstance(message_count, int)
            or message_count < 0
        ):
            raise ValueError("session message count must be a non-negative integer")
        resolution = await self._history_message_segments(
            history_id,
            tenant_id=tenant_id,
            start=0,
            end=message_count,
        )
        values: list[LoadedContextMessage] = []
        for segment in resolution.segments:
            index = segment.start
            async for message in self._iter_range(
                self.history_stream(segment.history_id),
                start=segment.start,
                end=segment.end,
                seek_owner_id=segment.history_id,
            ):
                values.append(
                    LoadedContextMessage(
                        message,
                        TranscriptMessageRef(
                            RuntimeDomain.CONVERSATION,
                            segment.history_id,
                            index,
                        ),
                    )
                )
                index += 1
            if index != segment.end:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if len(values) != message_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return LoadedModelContext(tuple(values))

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock("TranscriptRepository.iter_session_messages")
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            raise ValueError("session messages require the conversation archive")
        total = await self.history_message_count(
            history_id,
            tenant_id=tenant_id,
        )
        async for message in self.iter_session_message_range(
            history_id,
            tenant_id=tenant_id,
            start=0,
            end=total,
        ):
            yield message

    async def iter_session_message_range(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[ModelMessage]:
        require_no_run_history_lock(
            "TranscriptRepository.iter_session_message_range"
        )
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        resolution = await self._history_message_segments(
            history_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )
        expected = end - start
        emitted = 0
        for segment in resolution.segments:
            async for message in self._iter_range(
                self.history_stream(segment.history_id),
                start=segment.start,
                end=segment.end,
                seek_owner_id=segment.history_id,
            ):
                emitted += 1
                yield message
        if emitted != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def store_projection(
        self,
        transaction: StateTransaction,
        agent_run_id: str,
        projection: ContextProjection,
    ) -> None:
        key = self._projection_key(agent_run_id)
        value = StoredRecord(
            key,
            None,
            self._owner_key(agent_run_id),
            "context_projection",
            agent_run_id,
            None,
            0,
            None,
            0,
            None,
            encode_envelope(
                {
                    "type": "context_projection",
                    "payload": _encode_persisted_domain(projection),
                }
            ),
        )
        current = await transaction.get_record(key)
        if current is None:
            await transaction.insert_record(value)
            return
        if not await transaction.replace_record(
            replace(
                value,
                scope_digest=current.scope_digest,
                parent_digest=current.parent_digest,
                storage_version=current.storage_version + 1,
            ),
            expected_storage_version=current.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def load_projection(self, owner_id: str) -> ContextProjection | None:
        require_no_run_history_lock("TranscriptRepository.load_projection")
        value = await self._store.read(
            lambda transaction: transaction.get_record(self._projection_key(owner_id))
        )
        if value is None:
            return None
        return _decode_enveloped_domain(value.data, ContextProjection)

    async def load_projected_context(
        self,
        owner_id: str,
        projection: ContextProjection,
    ) -> LoadedModelContext:
        return (await self.load_projected_contexts(owner_id, (projection,)))[0]

    async def load_projected_contexts(
        self,
        owner_id: str,
        projections: Sequence[ContextProjection],
    ) -> tuple[LoadedModelContext, ...]:
        """Resolve several projections with one grouped transcript read."""
        del owner_id
        require_no_run_history_lock("TranscriptRepository.load_projected_contexts")
        projection_items: list[list[tuple[object, tuple[TranscriptMessageRef, ...]]]] = []
        refs: list[TranscriptMessageRef] = []
        payloads: dict[str, StoredPayload] = {}
        for projection in projections:
            items: list[tuple[object, tuple[TranscriptMessageRef, ...]]] = []
            for item in projection.items:
                if isinstance(item, TranscriptSpanRef):
                    item_refs = tuple(
                        TranscriptMessageRef(
                            item.source_domain,
                            item.owner_id,
                            index,
                        )
                        for index in range(item.start, item.end)
                    )
                    refs.extend(item_refs)
                    items.append((item, item_refs))
                else:
                    payloads[item.content.payload.digest] = item.content.payload
                    items.append((item, ()))
            projection_items.append(items)
        resolved = await self.resolve_transcript_message_refs(tuple(refs))
        loaded_payloads = {
            digest: decode_model_messages(await self._read_payload(payload))
            for digest, payload in payloads.items()
        }
        resolved_index = 0
        results: list[LoadedModelContext] = []
        for items in projection_items:
            values: list[LoadedContextMessage] = []
            for item, item_refs in items:
                if isinstance(item, TranscriptSpanRef):
                    values.extend(
                        resolved[resolved_index : resolved_index + len(item_refs)]
                    )
                    resolved_index += len(item_refs)
                    continue
                messages = loaded_payloads[item.content.payload.digest]  # type: ignore[union-attr]
                values.extend(
                    LoadedContextMessage(message, None) for message in messages
                )
            results.append(LoadedModelContext(tuple(values)))
        return tuple(results)

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
        """Resolve canonical raw transcript references in caller order."""
        require_no_run_history_lock(
            "TranscriptRepository.resolve_transcript_message_refs"
        )
        if not refs:
            return ()
        grouped: dict[
            tuple[TranscriptRepository, str],
            list[int],
        ] = {}
        ordered_sources: list[tuple[TranscriptRepository, TranscriptMessageRef]] = []
        for ref in refs:
            source = self if ref.source_domain is self._runtime_domain else (
                self._context_sources.get(ref.source_domain)
            )
            if source is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            ordered_sources.append((source, ref))
            grouped.setdefault((source, ref.owner_id), []).append(ref.message_index)
        resolved: dict[tuple[TranscriptRepository, str], dict[int, ModelMessage]] = {}
        for (source, owner_id), indexes in grouped.items():
            head = await source.get_head(owner_id)
            if head is None or any(index >= head.message_count for index in indexes):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            unique = sorted(set(indexes))
            windows: list[tuple[int, int]] = []
            window_start = unique[0]
            window_end = window_start + 1
            for index in unique[1:]:
                if index == window_end:
                    window_end += 1
                else:
                    windows.append((window_start, window_end))
                    window_start = index
                    window_end = index + 1
            windows.append((window_start, window_end))
            loaded = await source.load_message_spans(
                owner_id,
                windows,
                observed_head=head,
            )
            mapping: dict[int, ModelMessage] = {}
            for (start, end), messages in zip(windows, loaded, strict=True):
                if len(messages) != end - start:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                mapping.update(
                    (start + offset, message)
                    for offset, message in enumerate(messages)
                )
            resolved[(source, owner_id)] = mapping
        result: list[LoadedContextMessage] = []
        for source, ref in ordered_sources:
            try:
                message = resolved[(source, ref.owner_id)][ref.message_index]
            except KeyError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            result.append(LoadedContextMessage(message, ref))
        return tuple(result)

    async def load_model_context(
        self,
        owner_id: str,
    ) -> LoadedModelContext:
        require_no_run_history_lock("TranscriptRepository.load_model_context")
        projection = await self.load_projection(owner_id)
        if projection is None:
            head = await self.get_head(owner_id)
            if head is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if head.message_count:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return LoadedModelContext(())
        return await self._load_model_context_from_projection(owner_id, projection)

    async def _load_model_context_from_projection(
        self,
        owner_id: str,
        projection: ContextProjection,
    ) -> LoadedModelContext:
        return (await self.load_projected_contexts(owner_id, (projection,)))[0]

    async def load_message_span(
        self,
        owner_id: str,
        start: int,
        end: int,
        *,
        observed_head: TranscriptHeadRecord | None = None,
    ) -> tuple[ModelMessage, ...]:
        return (
            await self.load_message_spans(
                owner_id,
                ((start, end),),
                observed_head=observed_head,
            )
        )[0]

    async def load_message_spans(
        self,
        owner_id: str,
        windows: Sequence[tuple[int, int]],
        *,
        observed_head: TranscriptHeadRecord | None = None,
    ) -> tuple[tuple[ModelMessage, ...], ...]:
        require_no_run_history_lock("TranscriptRepository.load_message_spans")
        if not windows:
            return ()
        if any(start < 0 or end < start for start, end in windows):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        head = await self._resolve_observed_head(owner_id, observed_head)
        if any(end > head.message_count for _start, end in windows):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if all(start == end for start, end in windows):
            return tuple(() for _window in windows)
        values: list[list[ModelMessage]] = [[] for _ in windows]
        stop_at = max(end for _, end in windows)
        read_from = min(start for start, _ in windows)
        after_sequence = await self._seek_fact_sequence(
            owner_id,
            read_from,
            observed_head=head,
        )
        while True:
            facts = await self._store.read(
                lambda transaction, sequence=after_sequence: transaction.list_facts(
                    FactQuery(
                        self._transcript_stream(owner_id),
                        after_sequence=sequence,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not facts:
                break
            after_sequence = facts[-1].sequence
            for fact in facts:
                chunk = self.decode_chunk(fact)
                chunk_start = chunk.first_message_index
                chunk_end = chunk_start + chunk.message_count
                if chunk_start >= stop_at:
                    return self._validate_spans(
                        windows,
                        tuple(tuple(value) for value in values),
                    )
                if all(
                    chunk_end <= start or chunk_start >= end
                    for start, end in windows
                ):
                    continue
                messages = await self._decode_chunk_messages(chunk)
                for index, (start, end) in enumerate(windows):
                    if chunk_end <= start or chunk_start >= end:
                        continue
                    left = max(start - chunk_start, 0)
                    right = min(end - chunk_start, len(messages))
                    values[index].extend(messages[left:right])
        return self._validate_spans(
            windows,
            tuple(tuple(value) for value in values),
        )

    async def _seek_fact_sequence(
        self,
        owner_id: str,
        message_index: int,
        *,
        observed_head: TranscriptHeadRecord | None = None,
    ) -> int | None:
        if message_index < 0:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._resolve_observed_head(owner_id, observed_head)
        block = (message_index // _TRANSCRIPT_SEEK_BLOCK) * _TRANSCRIPT_SEEK_BLOCK
        record = await self._store.read(
            lambda transaction: transaction.get_record(
                self._seek_key(owner_id, block)
            )
        )
        if record is None:
            return None
        try:
            seek = _decode_enveloped_domain(record.data, TranscriptSeekRecord)
        except (AIError, TypeError, ValueError):
            return None
        if (
            record.kind != "transcript_seek"
            or seek.block_start != block
            or seek.dimension is not TranscriptSeekDimension.MESSAGE
            or seek.chunk_first_message_index > message_index
        ):
            return None
        return seek.fact_sequence - 1 if seek.fact_sequence > 0 else None

    async def _resolve_observed_head(
        self,
        owner_id: str,
        observed_head: TranscriptHeadRecord | None,
    ) -> TranscriptHeadRecord:
        head = observed_head
        if head is None:
            head = await self.get_head(owner_id)
        if (
            head is None
            or head.owner_domain is not self._owner_domain
            or head.owner_id != owner_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return head

    def _validate_spans(
        self,
        windows: Sequence[tuple[int, int]],
        values: tuple[tuple[ModelMessage, ...], ...],
    ) -> tuple[tuple[ModelMessage, ...], ...]:
        if any(
            start < 0
            or end < start
            or len(value) != end - start
            for (start, end), value in zip(windows, values, strict=True)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values


    async def _history_message_segments(
        self,
        history_id: str,
        *,
        tenant_id: str,
        start: int,
        end: int,
    ) -> _HistoryResolution:
        """Resolve only the visible transcript segments touched by a range."""
        require_no_run_history_lock(
            "TranscriptRepository._history_message_segments"
        )
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if self._history_repository is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

        async def read(
            transaction: StateTransaction,
        ) -> _HistoryResolution:
            record = await self._history_repository.get_in_transaction(
                transaction,
                history_id,
                tenant_id=tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            local_messages = await self._history_repository.local_head_in_transaction(
                transaction,
                history_id,
            )
            transcript_entry = await self.get_head_in_transaction(
                transaction,
                history_id,
            )
            if transcript_entry is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            transcript_head, _transcript_record = transcript_entry
            if transcript_head.message_count != local_messages:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            inherited = record.inherited_message_count
            total = inherited + transcript_head.message_count
            if end > total:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if start == end:
                return _HistoryResolution(())
            roots = await self._history_repository.get_forest_roots_in_transaction(
                transaction,
                record.prefix_index_head_id,
                max_roots=64,
            )
            resolved = await resolve_history_range_lazy(
                roots,
                lambda node_id: self._history_repository.get_index_node_in_transaction(
                    transaction,
                    node_id,
                ),
                owner_history_id=history_id,
                local_message_count=local_messages,
                inherited_message_count=inherited,
                range_start=start,
                range_end=end,
            )
            return _HistoryResolution(tuple(
                _HistorySegment(
                    item.segment.owner_history_id,
                    item.local_start,
                    item.local_end,
                )
                for item in resolved
            ))

        try:
            return await self._history_repository.state_store.read(read)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _iter_range(
        self,
        stream: bytes,
        *,
        start: int,
        end: int,
        seek_owner_id: "str | None" = None,
    ) -> AsyncIterator[ModelMessage]:
        if end <= start:
            return
        expected = end - start
        emitted = 0
        after_sequence: int | None = None
        if seek_owner_id is not None:
            after_sequence = await self._seek_fact_sequence(seek_owner_id, start)
        while True:
            facts = await self._store.read(
                lambda transaction, sequence=after_sequence: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=sequence,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not facts:
                if emitted != expected:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            after_sequence = facts[-1].sequence
            for fact in facts:
                chunk = self.decode_chunk(fact)
                chunk_start = chunk.first_message_index
                chunk_end = chunk_start + chunk.message_count
                if chunk_start >= end:
                    if emitted != expected:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return
                if chunk_end <= start:
                    continue
                messages = await self._decode_chunk_messages(chunk)
                left = max(start - chunk_start, 0)
                right = min(end - chunk_start, len(messages))
                for message in messages[left:right]:
                    emitted += 1
                    yield message

    def project_context(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        *,
        origins: Sequence[TranscriptOrigin] = (),
        sources: Sequence[TranscriptMessageRef | None] = (),
    ) -> ContextProjection:
        return self._projector.project(
            owner_id,
            messages,
            origins=origins,
            sources=sources,
        )

    async def _read_payload(self, payload: StoredPayload) -> bytes:
        require_no_run_history_lock("TranscriptRepository._read_payload")
        if payload.kind == "inline":
            try:
                value = payload.decode()
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if not isinstance(value, bytes):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return value
        if payload.ref is None or self._object_store is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await read_object(
            self._object_store,
            payload.ref.key,
            expected_digest=payload.digest,
            expected_size=payload.size,
        )

    async def read_payload(self, payload: StoredPayload) -> bytes:
        return await self._read_payload(payload)

    def decode_chunk(self, fact: StoredFact) -> TranscriptChunk:
        try:
            return _decode_enveloped_domain(fact.data, TranscriptChunk)
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _owner_key(self, owner_id: str) -> bytes:
        return self._head_key(owner_id)

    def _head_key(self, owner_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "transcript_head",
            owner_id,
        )

    def head_key(self, owner_id: str) -> bytes:
        """Return the physical key for one typed transcript head."""
        return self._head_key(owner_id)

    def _seek_key(self, owner_id: str, block_start: int) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "transcript_seek",
            [owner_id, TranscriptSeekDimension.MESSAGE.value, block_start],
        )

    def _transcript_stream(self, owner_id: str) -> bytes:
        return (
            self.history_stream(owner_id)
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else self.agent_run_stream(owner_id)
        )

    def _transcript_sequence(self, owner_id: str) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "history_transcript"
            if self._runtime_domain is RuntimeDomain.CONVERSATION
            else "run_transcript",
            owner_id,
        )

    def _pending_stream(self, owner_id: str, message_index: int) -> bytes:
        return stream_digest(
            self._namespace, self._tenant_id, self._runtime_domain.value,
            "transcript_pending_parts", [owner_id, message_index],
        )

    def _projection_key(self, agent_run_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "context_projection",
            agent_run_id,
        )

    def projection_key(self, agent_run_id: str) -> bytes:
        """Return the physical key for one context projection."""
        return self._projection_key(agent_run_id)


async def _one_chunk(value: bytes) -> AsyncIterator[bytes]:
    yield value


__all__ = ["TranscriptRepository"]
