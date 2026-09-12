#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical transcript chunks and bounded context projections."""

import hashlib
import json
import zlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace

from linktools.core import environ
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from ...core import canonical_json_bytes
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, StoredPayload, runtime_object_key
from .._message import decode_model_messages, encode_model_messages
from ._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_domain,
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
    partition_digest,
    record_key_digest,
    require_no_run_history_lock,
    sequence_key,
    stream_digest,
)
_logger = environ.get_logger("ai.runtime.state.history")
_CHUNK_TARGET = 256 * 1024
_COMPRESS_MINIMUM = 16 * 1024
_COMPRESS_RATIO = 0.9
_TRANSCRIPT_PAGE_SIZE = 64
_TRANSCRIPT_CHUNK_MAX_MESSAGES = 64
_TRANSCRIPT_SEEK_BLOCK = 128


def _overlap_signature(message: ModelMessage) -> bytes:
    """Timestamp-ignoring signature used only for overlap/dedup matching."""
    value = json.loads(
        encode_model_messages((message,)).decode("utf-8")
    )

    def remove_timestamps(candidate: object) -> object:
        if isinstance(candidate, list):
            return [remove_timestamps(item) for item in candidate]
        if isinstance(candidate, dict):
            return {
                key: remove_timestamps(item)
                for key, item in candidate.items()
                if key != "timestamp"
            }
        return candidate

    return canonical_json_bytes(remove_timestamps(value))


def _conversation_overlap_signature(message: ModelMessage) -> bytes:
    if not isinstance(message, ModelRequest):
        return _overlap_signature(message)
    return _overlap_signature(
        replace(
            message,
            parts=tuple(
                part
                for part in message.parts
                if not isinstance(part, SystemPromptPart)
            ),
        )
    )


def _exact_message_signature(message: ModelMessage) -> bytes:
    """Full canonical serialization including timestamp, for exact-content proof."""
    return encode_model_messages((message,))


def suffix_prefix_overlap(
    stored: Sequence[bytes],
    incoming: Sequence[bytes],
) -> int:
    """Return the longest suffix/prefix overlap using one linear pass."""
    if not stored or not incoming:
        return 0
    prefix = [0] * len(incoming)
    matched = 0
    for index in range(1, len(incoming)):
        while matched and incoming[index] != incoming[matched]:
            matched = prefix[matched - 1]
        if incoming[index] == incoming[matched]:
            matched += 1
        prefix[index] = matched
    matched = 0
    for index, value in enumerate(stored):
        while matched and value != incoming[matched]:
            matched = prefix[matched - 1]
        if value == incoming[matched]:
            matched += 1
        if matched == len(incoming):
            if index == len(stored) - 1:
                return matched
            matched = prefix[matched - 1]
    return matched


@dataclass(frozen=True, slots=True)
class TranscriptCapture:
    first_message_index: int
    messages: tuple[ModelMessage, ...]
    origins: tuple[TranscriptOrigin, ...]
    quality: HistoryQuality


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
            0,
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
            self._partition("transcript_head"),
            None,
            None,
            "transcript_head",
            head.owner_id,
            None,
            0,
            None,
            0,
            None,
            encode_envelope(
                {"type": "transcript_head", "payload": _encode_persisted_domain(head)}
            ),
        )

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
            revision=base_head.revision + 1,
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
                    self._partition("transcript_seek"),
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
        run_id: str,
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

    def run_stream(self, step_run_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "run_transcript",
            step_run_id,
        )

    def transcript_stream(self, run_id: str) -> bytes:
        """Return the owner stream used by this archive's domain."""
        if self._runtime_domain is RuntimeDomain.CONVERSATION:
            return self.history_stream(run_id)
        return self.run_stream(run_id)

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
            raw = zlib.decompress(raw)
        if hashlib.sha256(raw).hexdigest() != chunk.raw_digest or len(raw) != chunk.raw_size:
            raise ValueError("transcript chunk integrity check failed")
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
                        partition_digest=self._partition("transcript_head"),
                        kind="transcript_head",
                        after_sort_key=sort_key,
                        after_key_digest=key_digest,
                        limit=_TRANSCRIPT_PAGE_SIZE,
                    )
                )
            )
            if not page:
                return
            for record in page:
                await self._validate_head_record(record)
            last = page[-1]
            after_sort = last.sort_key
            after_key = last.key_digest

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
        run_id: str,
        projection: ContextProjection,
    ) -> None:
        key = self._projection_key(run_id)
        value = StoredRecord(
            key,
            self._partition("context_projection"),
            None,
            self._owner_key(run_id),
            "context_projection",
            run_id,
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
                partition_digest=current.partition_digest,
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
        span_refs: list[tuple[TranscriptSpanRef, tuple[TranscriptMessageRef, ...]]] = []
        refs: list[TranscriptMessageRef] = []
        for item in projection.items:
            if not isinstance(item, TranscriptSpanRef):
                span_refs.append((item, ()))  # type: ignore[arg-type]
                continue
            item_refs = tuple(
                TranscriptMessageRef(
                    item.source_domain,
                    item.owner_id,
                    index,
                )
                for index in range(item.start, item.end)
            )
            refs.extend(item_refs)
            span_refs.append((item, item_refs))
        resolved = await self.resolve_transcript_message_refs(tuple(refs))
        resolved_index = 0
        values: list[LoadedContextMessage] = []
        for item, item_refs in span_refs:
            if isinstance(item, TranscriptSpanRef):
                values.extend(
                    resolved[resolved_index : resolved_index + len(item_refs)]
                )
                resolved_index += len(item_refs)
                continue
            raw = await self._read_payload(item.content.payload)  # type: ignore[union-attr]
            values.extend(
                LoadedContextMessage(message, None)
                for message in decode_model_messages(raw)
            )
        return LoadedModelContext(tuple(values))

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
            value = payload.decode()
            if not isinstance(value, bytes):
                raise ValueError("inline transcript payload is not binary")
            return value
        if payload.ref is None or self._object_store is None:
            raise ValueError("object transcript payload has no reader")
        data = bytearray()
        async for chunk in self._object_store.open(payload.ref.key):
            data.extend(chunk)
        if len(data) != payload.size or hashlib.sha256(data).hexdigest() != payload.digest:
            raise ValueError("object transcript payload integrity check failed")
        return bytes(data)

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
            else self.run_stream(owner_id)
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

    def _projection_key(self, run_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "context_projection",
            run_id,
        )

    def projection_key(self, run_id: str) -> bytes:
        """Return the physical key for one context projection."""
        return self._projection_key(run_id)

    def _partition(self, kind: str) -> bytes:
        return partition_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            kind,
        )


async def _one_chunk(value: bytes) -> AsyncIterator[bytes]:
    yield value


__all__ = ["TranscriptRepository"]
