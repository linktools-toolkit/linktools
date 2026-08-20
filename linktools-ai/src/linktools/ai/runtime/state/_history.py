#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical transcript chunks and bounded context projections."""

import hashlib
import json
import zlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace

from linktools.core import environ
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from ...core import canonical_json_bytes
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, StoredPayload, runtime_object_key
from ._codec import decode_domain, encode_domain
from ._history_index import (
    HistoryIndexSnapshot,
    resolve_history_range,
)
from ._contracts import (
    ContextProjection,
    ConversationHistoryRecord,
    ConversationHistoryRepository,
    HistoryQuality,
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    RuntimePayloadRef,
    TranscriptChunk,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
    TranscriptSpanRef,
)
from ._plan import RuntimeDomain
from ._store import (
    FactQuery,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    partition_digest,
    record_key_digest,
    sequence_key,
    stream_digest,
    subject_digest,
)

_logger = environ.get_logger("ai.runtime.state.history")
_CHUNK_TARGET = 256 * 1024
_COMPRESS_MINIMUM = 16 * 1024
_COMPRESS_RATIO = 0.9
_TRANSCRIPT_PAGE_SIZE = 64
_TRANSCRIPT_CHUNK_MAX_MESSAGES = 64
_TRANSCRIPT_SEEK_BLOCK = 128
SESSION_HISTORY_VIEW_VERSION = 1


def _overlap_signature(message: ModelMessage) -> bytes:
    """Timestamp-ignoring signature used only for overlap/dedup matching."""
    value = json.loads(
        ModelMessagesTypeAdapter.dump_json([message]).decode("utf-8")
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


def _exact_message_signature(message: ModelMessage) -> bytes:
    """Full canonical serialization including timestamp, for exact-content proof."""
    return canonical_json_bytes(
        json.loads(ModelMessagesTypeAdapter.dump_json([message]).decode("utf-8"))
    )


@dataclass(frozen=True, slots=True)
class TranscriptCapture:
    first_message_index: int
    messages: tuple[ModelMessage, ...]
    origins: tuple[TranscriptOrigin, ...]
    quality: HistoryQuality


@dataclass(frozen=True, slots=True)
class TranscriptAccumulatorAdvance:
    run_id: str
    base_generation: int
    target_generation: int
    base_message_count: int
    target_message_count: int
    delta_messages: tuple[ModelMessage, ...]
    delta_signatures: tuple[bytes, ...]
    target_quality: HistoryQuality


@dataclass(frozen=True, slots=True)
class _HistorySegment:
    history_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _HistoryResolution:
    segments: tuple[_HistorySegment, ...]


@dataclass(frozen=True, slots=True)
class _TranscriptAccumulatorState:
    """Complete accumulator state; the only legal clone carrier."""

    run_id: str
    messages: tuple[ModelMessage, ...]
    signatures: tuple[bytes, ...]
    quality: HistoryQuality
    generation: int
    seeded: bool


class _TranscriptAccumulator:
    """Capture only messages proven to belong to one Pydantic AI run."""

    def __init__(
        self,
        run_id: str,
    ) -> None:
        self._run_id = run_id
        self._messages: list[ModelMessage] = []
        self._signatures: list[bytes] = []
        self._quality = HistoryQuality.COMPLETE
        self._generation = 0
        self._seeded = False

    @property
    def messages(self) -> tuple[ModelMessage, ...]:
        return tuple(self._messages)

    @property
    def quality(self) -> HistoryQuality:
        return self._quality

    @property
    def generation(self) -> int:
        return self._generation

    def snapshot_state(self) -> _TranscriptAccumulatorState:
        return _TranscriptAccumulatorState(
            self._run_id,
            tuple(self._messages),
            tuple(self._signatures),
            self._quality,
            self._generation,
            self._seeded,
        )

    @classmethod
    def from_state(
        cls,
        state: _TranscriptAccumulatorState,
    ) -> "_TranscriptAccumulator":
        accumulator = cls(state.run_id)
        accumulator._messages = list(state.messages)
        accumulator._signatures = list(state.signatures)
        accumulator._quality = state.quality
        accumulator._generation = state.generation
        accumulator._seeded = state.seeded
        return accumulator

    def seed(self, messages: Sequence[ModelMessage]) -> None:
        if self._seeded:
            raise RuntimeError("transcript accumulator is already seeded")
        self._messages.extend(messages)
        self._signatures.extend(self.signature(message) for message in messages)
        self._seeded = True

    def signature(self, message: ModelMessage) -> bytes:
        return _overlap_signature(message)

    def plan(
        self,
        messages: Sequence[ModelMessage],
    ) -> TranscriptAccumulatorAdvance:
        incoming = tuple(messages)
        incoming_signatures = tuple(self.signature(message) for message in incoming)
        known = tuple(
            message
            for message in incoming
            if message.run_id == self._run_id
        )
        known_signatures = tuple(
            signature
            for message, signature in zip(
                incoming,
                incoming_signatures,
                strict=True,
            )
            if message.run_id == self._run_id
        )
        stored_known = tuple(
            message
            for message, _signature in zip(
                self._messages,
                self._signatures,
                strict=True,
            )
            if self._seeded or message.run_id == self._run_id
        )
        stored_known_signatures = tuple(
            signature
            for message, signature in zip(
                self._messages,
                self._signatures,
                strict=True,
            )
            if self._seeded or message.run_id == self._run_id
        )
        quality = self._quality
        if len(known) < len(stored_known):
            quality = HistoryQuality.CONSERVATIVE
        overlap = self._overlap(known_signatures, stored_known_signatures)
        if known and stored_known and overlap == 0:
            quality = HistoryQuality.CONSERVATIVE
        delta = list(known[overlap:])
        delta_signatures = list(known_signatures[overlap:])
        known_signatures_set = set(self._signatures)
        known_signatures_set.update(delta_signatures)
        for message, signature in zip(incoming, incoming_signatures, strict=True):
            if message.run_id is not None:
                continue
            if signature in known_signatures_set:
                continue
            delta.append(message)
            delta_signatures.append(signature)
            known_signatures_set.add(signature)
            quality = HistoryQuality.CONSERVATIVE
        base_count = len(self._messages)
        return TranscriptAccumulatorAdvance(
            self._run_id,
            self._generation,
            self._generation + 1,
            base_count,
            base_count + len(delta),
            tuple(delta),
            tuple(delta_signatures),
            quality,
        )

    def apply(self, advance: TranscriptAccumulatorAdvance) -> None:
        if advance.run_id != self._run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            self._generation == advance.target_generation
            and len(self._messages) == advance.target_message_count
        ):
            signatures = (
                ()
                if not advance.delta_signatures
                else tuple(self._signatures[-len(advance.delta_signatures) :])
            )
            if (
                signatures != advance.delta_signatures
                or self._quality is not advance.target_quality
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if (
            self._generation != advance.base_generation
            or len(self._messages) != advance.base_message_count
            or len(self._signatures) != advance.base_message_count
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._messages.extend(advance.delta_messages)
        self._signatures.extend(advance.delta_signatures)
        self._quality = advance.target_quality
        self._generation = advance.target_generation

    def _overlap(
        self,
        incoming: tuple[bytes, ...],
        stored: tuple[bytes, ...],
    ) -> int:
        if not incoming or not stored:
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


class _ContextProjector:
    def __init__(self, runtime_domain: RuntimeDomain) -> None:
        self._runtime_domain = runtime_domain

    def project(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        *,
        binding_digest: str,
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
                raw = ModelMessagesTypeAdapter.dump_json(list(values[index:end]))
                items.append(
                    InlineContextBlock(
                        RuntimePayloadRef(
                            StoredPayload.inline_bytes(raw),
                            self._runtime_domain,
                        )
                    )
                )
            index = end
        digest = self._digest(binding_digest, items)
        return ContextProjection(binding_digest, tuple(items), digest)

    def _digest(
        self,
        binding_digest: str,
        items: Sequence[TranscriptSpanRef | InlineContextBlock],
    ) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "binding_digest": binding_digest,
                    "items": encode_domain(tuple(items)),
                }
            )
        ).hexdigest()


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

    async def prepare_chunks(
        self,
        owner_id: str,
        messages: Sequence[ModelMessage],
        *,
        first_message_index: int,
        origin: TranscriptOrigin = TranscriptOrigin.RAW,
    ) -> tuple[TranscriptChunk, ...]:
        values = tuple(messages)
        chunks: list[TranscriptChunk] = []
        current: list[ModelMessage] = []
        current_start = first_message_index
        current_size = 2
        for message in values:
            encoded = ModelMessagesTypeAdapter.dump_json([message])
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
        raw = ModelMessagesTypeAdapter.dump_json(list(messages))
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
            ObjectRef(self._object_store.store_id, key, stat.digest, stat.size)
        )

    async def append_chunks(
        self,
        transaction: StateTransaction,
        owner_id: str,
        chunks: Sequence[TranscriptChunk],
        chunk_history_item_counts: Sequence[int] | None = None,
    ) -> None:
        if not chunks:
            return
        stream = self._transcript_stream(owner_id)
        owner = self._owner_key(owner_id)
        owner_record = await transaction.get_record(owner)
        current_count = await self._owner_message_count(
            transaction,
            owner_id,
            owner_record,
        )
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
        added_items = (
            0
            if chunk_history_item_counts is None
            else sum(chunk_history_item_counts)
        )
        prior_items = 0 if owner_record is None else owner_record.data.get(
            "history_item_count",
            0,
        )
        if isinstance(prior_items, bool) or not isinstance(prior_items, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if owner_record is None:
            await transaction.insert_record(
                self._new_owner_record(
                    owner_id,
                    owner,
                    expected,
                    history_item_count=added_items,
                    chunk_count=len(chunks),
                )
            )
        else:
            upgraded = replace(
                owner_record,
                storage_version=owner_record.storage_version + 1,
                data={
                    **owner_record.data,
                    "message_count": expected,
                    "history_item_count": prior_items + added_items,
                    "chunk_count": owner_record.data.get("chunk_count", 0)
                    + len(chunks),
                },
            )
            if not await transaction.replace_record(
                upgraded,
                expected_storage_version=owner_record.storage_version,
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
                    {
                        "type": "transcript_chunk",
                        "payload": encode_domain(chunk),
                    },
                )
                for sequence, chunk in zip(sequences, chunks, strict=True)
            )
        )
        await self._insert_seek_boundaries(
            transaction,
            owner_id,
            chunks,
            sequences,
            chunk_history_item_counts,
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

    async def _insert_seek_boundaries(
        self,
        transaction: StateTransaction,
        owner_id: str,
        chunks: Sequence[TranscriptChunk],
        sequences: Sequence[int],
        chunk_history_item_counts: Sequence[int] | None,
    ) -> None:
        """Insert seek facts for block boundaries crossed by this append."""
        first = chunks[0].first_message_index
        last = chunks[-1].first_message_index + chunks[-1].message_count
        block = _TRANSCRIPT_SEEK_BLOCK
        starts = range(
            ((first + block - 1) // block) * block,
            ((last + block - 1) // block) * block,
            block,
        )
        if not starts:
            return
        head_record = await transaction.get_record(self._owner_key(owner_id))
        if head_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        item_base_value = head_record.data.get("history_item_count", 0)
        if isinstance(item_base_value, bool) or not isinstance(item_base_value, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if chunk_history_item_counts is None:
            return
        boundaries: dict[int, tuple[int, int]] = {}
        message_cursor = first
        item_cursor = item_base_value
        for chunk, sequence, item_count in zip(
            chunks,
            sequences,
            chunk_history_item_counts,
            strict=True,
        ):
            boundaries[message_cursor] = (sequence, item_cursor)
            message_cursor += chunk.message_count
            item_cursor += item_count
        facts: list[StoredFact] = []
        for block_start in starts:
            boundary = boundaries.get(block_start)
            if boundary is None:
                continue
            fact_sequence, item_index = boundary
            for dimension in TranscriptSeekDimension:
                facts.append(
                    StoredFact(
                        self._seek_stream(owner_id),
                        0,
                        self._owner_key(owner_id),
                        "transcript_seek",
                        subject_digest(dimension.value),
                        dimension.value,
                        {
                            "type": "transcript_seek",
                            "payload": encode_domain(
                                TranscriptSeekRecord(
                                    owner_id,
                                    dimension,
                                    block_start,
                                    fact_sequence,
                                    block_start,
                                    item_index,
                                    SESSION_HISTORY_VIEW_VERSION,
                                )
                            ),
                        },
                    )
                )
        if facts:
            final = await transaction.reserve_sequence(
                self._seek_sequence(owner_id),
                len(facts),
            )
            ordered = tuple(
                replace(
                    fact,
                    sequence=sequence,
                )
                for fact, sequence in zip(
                    facts,
                    range(final - len(facts) + 1, final + 1),
                    strict=True,
                )
            )
            await transaction.insert_facts(ordered)
            _logger.debug(
                "transcript seek boundaries inserted: domain=%s owner=%s count=%s",
                self._runtime_domain.value,
                owner_id,
                len(ordered),
            )

    def _seek_stream(self, owner_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "transcript_seek",
            owner_id,
        )

    def _seek_sequence(self, owner_id: str) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "transcript_seek",
            owner_id,
        )

    async def _owner_message_count(
        self,
        transaction: StateTransaction,
        owner_id: str,
        owner_record: StoredRecord | None,
    ) -> int:
        if owner_record is None:
            return 0
        value = owner_record.data.get("message_count")
        if value is None:
            values = await transaction.list_facts(
                FactQuery(self._transcript_stream(owner_id), latest=True)
            )
            if not values:
                return 0
            chunk = self.decode_chunk(values[0])
            return chunk.first_message_index + chunk.message_count
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def _new_owner_record(
        self,
        owner_id: str,
        owner: bytes,
        message_count: int,
        *,
        history_item_count: int,
        chunk_count: int,
    ) -> StoredRecord:
        return StoredRecord(
            owner,
            self._partition("history_owner"),
            None,
            owner,
            "history_owner",
            owner_id,
            None,
            0,
            None,
            0,
            None,
            {
                "type": "history_owner",
                "owner_id": owner_id,
                "message_count": message_count,
                "history_item_count": history_item_count,
                "chunk_count": chunk_count,
            },
        )

    async def prepare_projection(
        self,
        run_id: str,
        projection: ContextProjection,
    ) -> ContextProjection:
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
                                self._object_store.store_id,
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
        digest = self._projector._digest(projection.binding_digest, items)
        return ContextProjection(projection.binding_digest, tuple(items), digest)

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
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(self._transcript_stream(owner_id), latest=True)
            )
        )
        if not values:
            return None
        return self.decode_chunk(values[0])

    async def iter_messages(self, owner_id: str) -> AsyncIterator[ModelMessage]:
        stream = self._transcript_stream(owner_id)
        after_sequence: int | None = None
        while True:
            values = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=after_sequence,
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

    async def _decode_chunk_messages(self, chunk: TranscriptChunk) -> tuple[ModelMessage, ...]:
        raw = await self._read_payload(chunk.content.payload)
        if chunk.codec == "zlib":
            raw = zlib.decompress(raw)
        if hashlib.sha256(raw).hexdigest() != chunk.raw_digest or len(raw) != chunk.raw_size:
            raise ValueError("transcript chunk integrity check failed")
        return tuple(ModelMessagesTypeAdapter.validate_json(raw))

    async def load_messages(self, owner_id: str) -> tuple[ModelMessage, ...]:
        return tuple([message async for message in self.iter_messages(owner_id)])

    async def validate_integrity(self) -> None:
        async def read(transaction: StateTransaction) -> None:
            records = await transaction.scan_records()
            facts = await transaction.scan_facts()
            owners = {
                record.key_digest: record
                for record in records
                if record.kind in {"history_owner", "step_run"}
            }
            facts_by_owner: dict[bytes, list[StoredFact]] = {}
            for fact in facts:
                if fact.kind != "transcript_chunk":
                    continue
                try:
                    chunk = self.decode_chunk(fact)
                except (TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if (
                    fact.owner_key_digest != self._owner_key(chunk.owner_id)
                    or fact.stream_digest != self._transcript_stream(chunk.owner_id)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if fact.owner_key_digest not in owners:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                facts_by_owner.setdefault(fact.owner_key_digest, []).append(fact)
            for owner_key, record in owners.items():
                value = record.data.get("message_count")
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                expected = 0
                for fact in sorted(
                    facts_by_owner.get(owner_key, ()),
                    key=lambda item: item.sequence,
                ):
                    try:
                        chunk = self.decode_chunk(fact)
                    except (TypeError, ValueError) as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    if (
                        chunk.message_count <= 0
                        or chunk.first_message_index != expected
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    expected += chunk.message_count
                if expected != value:
                    _logger.error(
                        "history owner count mismatch: domain=%s owner=%s "
                        "record_count=%s fact_count=%s",
                        self._runtime_domain.value,
                        record.data.get("owner_id"),
                        value,
                        expected,
                    )
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        await self._store.read(read)

    async def history_message_count(self, history_id: str, *, tenant_id: str) -> int:
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
        local_messages = await self._history_repository.state_store.read(
            lambda transaction: (
                self._history_repository.local_head_in_transaction(
                    transaction,
                    history_id,
                )
            )
        )
        return record.inherited_message_count + local_messages[0]

    async def transcript_message_count(self, owner_id: str) -> int:
        async def read(transaction: StateTransaction) -> int:
            owner = await transaction.get_record(self._owner_key(owner_id))
            return await self._owner_message_count(transaction, owner_id, owner)

        return await self._store.read(read)

    async def load_session_model_context(
        self,
        history_id: str,
        *,
        tenant_id: str,
        binding_digest: str | None = None,
    ) -> LoadedModelContext:
        projection = await self.load_projection(history_id)
        if projection is not None:
            if binding_digest is not None and projection.binding_digest != binding_digest:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            resolution = await self._history_segments(
                history_id,
                tenant_id=tenant_id,
            )
            self._validate_session_projection_ranges(projection, resolution)
            return await self.load_model_context(
                history_id,
                binding_digest=binding_digest,
            )
        resolution = await self._history_segments(
            history_id,
            tenant_id=tenant_id,
        )
        values: list[LoadedContextMessage] = []
        for segment in resolution.segments:
            messages = await self.load_message_span(
                segment.history_id,
                segment.start,
                segment.end,
            )
            values.extend(
                LoadedContextMessage(
                    message,
                    TranscriptMessageRef(
                        RuntimeDomain.CONVERSATION,
                        segment.history_id,
                        segment.start + index,
                    ),
                )
                for index, message in enumerate(messages)
            )
        return LoadedModelContext(tuple(values))

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[ModelMessage]:
        if self._runtime_domain is not RuntimeDomain.CONVERSATION:
            raise ValueError("session messages require the conversation archive")
        resolution = await self._history_segments(
            history_id,
            tenant_id=tenant_id,
        )
        for segment in resolution.segments:
            async for message in self._iter_range(
                self.history_stream(segment.history_id),
                start=segment.start,
                end=segment.end,
                seek_owner_id=segment.history_id,
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
        if start < 0 or end < start:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        resolution = await self._history_segments(
            history_id,
            tenant_id=tenant_id,
        )
        if end > sum(segment.end - segment.start for segment in resolution.segments):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        position = 0
        for segment in resolution.segments:
            segment_length = segment.end - segment.start
            window_start = max(start - position, 0)
            window_end = min(end - position, segment_length)
            if window_start < window_end:
                async for message in self._iter_range(
                    self.history_stream(segment.history_id),
                    start=segment.start + window_start,
                    end=segment.start + window_end,
                    seek_owner_id=segment.history_id,
                ):
                    yield message
            position += segment_length

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
            {"type": "context_projection", "payload": encode_domain(projection)},
        )
        current = await transaction.get_record(key)
        if current is None:
            await transaction.insert_record(value)
            return
        await transaction.replace_record(
            StoredRecord(
                key,
                current.partition_digest,
                current.scope_digest,
                current.parent_digest,
                current.kind,
                current.sort_key,
                current.state,
                current.storage_version + 1,
                current.lease_owner,
                current.lease_fence,
                current.lease_expires_at,
                value.data,
            ),
            expected_storage_version=current.storage_version,
        )

    async def load_projection(self, owner_id: str) -> ContextProjection | None:
        value = await self._store.read(
            lambda transaction: transaction.get_record(self._projection_key(owner_id))
        )
        if value is None:
            return None
        payload = value.data.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("context projection payload is invalid")
        return decode_domain(payload, ContextProjection)

    async def load_model_context(
        self,
        owner_id: str,
        *,
        binding_digest: str | None = None,
    ) -> LoadedModelContext:
        projection = await self.load_projection(owner_id)
        if projection is None:
            values = await self.load_messages(owner_id)
            source_domain = self._runtime_domain
            return LoadedModelContext(
                tuple(
                    LoadedContextMessage(
                        message,
                        TranscriptMessageRef(source_domain, owner_id, index),
                    )
                    for index, message in enumerate(values)
                )
            )
        if binding_digest is not None and projection.binding_digest != binding_digest:
            _logger.info(
                "context projection binding mismatch: domain=%s owner=%s",
                self._runtime_domain.value,
                owner_id,
            )
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        span_groups: dict[
            tuple[TranscriptRepository, RuntimeDomain, str],
            list[tuple[int, int]],
        ] = {}
        for item in projection.items:
            if isinstance(item, TranscriptSpanRef):
                source = self
                if item.source_domain is not self._runtime_domain:
                    source = self._context_sources.get(item.source_domain)
                    if source is None:
                        raise ValueError("transcript context source is unavailable")
                span_groups.setdefault((source, item.source_domain, item.owner_id), []).append(
                    (item.start, item.end)
                )
        span_values: dict[
            tuple[TranscriptRepository, RuntimeDomain, str],
            tuple[tuple[ModelMessage, ...], ...],
        ] = {}
        for key, windows in span_groups.items():
            source, _source_domain, span_owner_id = key
            span_values[key] = await source.load_message_spans(span_owner_id, windows)
        span_offsets: dict[tuple[TranscriptRepository, RuntimeDomain, str], int] = {}
        values: list[LoadedContextMessage] = []
        for item in projection.items:
            if isinstance(item, TranscriptSpanRef):
                source = self
                if item.source_domain is not self._runtime_domain:
                    source = self._context_sources.get(item.source_domain)
                    if source is None:
                        raise ValueError("transcript context source is unavailable")
                key = (source, item.source_domain, item.owner_id)
                offset = span_offsets.get(key, 0)
                messages = span_values[key][offset]
                values.extend(
                    LoadedContextMessage(
                        message,
                        TranscriptMessageRef(
                            item.source_domain,
                            item.owner_id,
                            item.start + index,
                        ),
                    )
                    for index, message in enumerate(messages)
                )
                span_offsets[key] = offset + 1
            else:
                raw = await self._read_payload(item.content.payload)
                values.extend(
                    LoadedContextMessage(message, None)
                    for message in ModelMessagesTypeAdapter.validate_json(raw)
                )
        return LoadedModelContext(tuple(values))

    async def load_message_span(
        self,
        owner_id: str,
        start: int,
        end: int,
    ) -> tuple[ModelMessage, ...]:
        return (await self.load_message_spans(owner_id, ((start, end),)))[0]

    async def load_message_spans(
        self,
        owner_id: str,
        windows: Sequence[tuple[int, int]],
    ) -> tuple[tuple[ModelMessage, ...], ...]:
        if not windows:
            return ()
        values: list[list[ModelMessage]] = [[] for _ in windows]
        stop_at = max(end for _, end in windows)
        read_from = min(start for start, _ in windows)
        after_sequence = await self._seek_fact_sequence(owner_id, read_from)
        while True:
            facts = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        self._transcript_stream(owner_id),
                        after_sequence=after_sequence,
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
                if all(chunk_end <= start or chunk_start >= end for start, end in windows):
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

    async def _seek_fact_sequence(self, owner_id: str, message_index: int) -> "int | None":
        """Anchor a fact scan at the newest seek block at or before the index."""
        if message_index < _TRANSCRIPT_SEEK_BLOCK:
            return None
        block = (message_index // _TRANSCRIPT_SEEK_BLOCK) * _TRANSCRIPT_SEEK_BLOCK
        del block
        facts = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    self._seek_stream(owner_id),
                    subject_digest=subject_digest(
                        TranscriptSeekDimension.MESSAGE.value
                    ),
                    limit=64,
                )
            )
        )
        candidates = [
            decode_domain(fact.data["payload"], TranscriptSeekRecord)
            for fact in facts
            if fact.data.get("payload") is not None
        ]
        below = [
            record for record in candidates if record.block_start <= message_index
        ]
        if not below:
            return None
        best = max(below, key=lambda record: record.block_start)
        return best.fact_sequence - 1 if best.fact_sequence > 0 else None

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

    async def _history_segments(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> _HistoryResolution:
        """Resolve one branch's visible transcript into logical segments."""
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
            local_messages, _items = (
                await self._history_repository.local_head_in_transaction(
                    transaction,
                    history_id,
                )
            )
            inherited = record.inherited_message_count
            snapshot = (
                HistoryIndexSnapshot({}, ())
                if record.prefix_index_head_id is None
                else await self._history_repository.index_snapshot_in_transaction(
                    transaction,
                    record.prefix_index_head_id,
                )
            )
            resolved = resolve_history_range(
                snapshot,
                owner_history_id=history_id,
                local_message_count=local_messages,
                inherited_message_count=inherited,
                range_start=0,
                range_end=inherited + local_messages,
            )
            segments = tuple(
                _HistorySegment(
                    item.segment.owner_history_id,
                    item.local_start,
                    item.local_end,
                )
                for item in resolved
            )
            return _HistoryResolution(segments)

        try:
            return await self._history_repository.state_store.read(read)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_session_projection_ranges(
        self,
        projection: ContextProjection,
        resolution: _HistoryResolution,
    ) -> None:
        ranges = {
            segment.history_id: (segment.start, segment.end)
            for segment in resolution.segments
        }
        for item in projection.items:
            if not isinstance(item, TranscriptSpanRef):
                continue
            if item.source_domain is not RuntimeDomain.CONVERSATION:
                continue
            allowed = ranges.get(item.owner_id)
            if (
                allowed is None
                or item.start < allowed[0]
                or item.end > allowed[1]
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

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
        after_sequence: "int | None" = None
        if seek_owner_id is not None:
            after_sequence = await self._seek_fact_sequence(seek_owner_id, start)
        while True:
            facts = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=after_sequence,
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
        binding_digest: str,
        origins: Sequence[TranscriptOrigin] = (),
        sources: Sequence[TranscriptMessageRef | None] = (),
    ) -> ContextProjection:
        return self._projector.project(
            owner_id,
            messages,
            binding_digest=binding_digest,
            origins=origins,
            sources=sources,
        )

    async def _read_payload(self, payload: StoredPayload) -> bytes:
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
        payload = fact.data.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("transcript chunk payload is invalid")
        return decode_domain(payload, TranscriptChunk)

    def _owner_key(self, owner_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "history_owner" if self._runtime_domain is RuntimeDomain.CONVERSATION else "step_run",
            owner_id,
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

    def _partition(self, kind: str) -> bytes:
        return partition_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            kind,
        )


async def _one_chunk(value: bytes) -> AsyncIterator[bytes]:
    yield value


__all__ = ["TranscriptAccumulatorAdvance", "TranscriptRepository"]
