#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical transcript chunks and bounded context projections."""

import hashlib
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from linktools.core import environ

from ...core import canonical_json_bytes
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, StoredPayload, runtime_object_key
from ._codec import decode_domain, encode_domain
from ._contracts import (
    ContextProjection,
    ConversationHistoryRepository,
    ConversationHistoryRecord,
    HistoryQuality,
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    RuntimePayloadRef,
    TranscriptChunk,
    TranscriptMessageRef,
    TranscriptOrigin,
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
)

_logger = environ.get_logger("ai.runtime.state.history")
_CHUNK_TARGET = 256 * 1024
_COMPRESS_MINIMUM = 16 * 1024
_COMPRESS_RATIO = 0.9
_TRANSCRIPT_PAGE_SIZE = 64


@dataclass(frozen=True, slots=True)
class TranscriptCapture:
    first_message_index: int
    messages: tuple[ModelMessage, ...]
    origins: tuple[TranscriptOrigin, ...]
    quality: HistoryQuality


class _TranscriptAccumulator:
    """Capture only messages proven to belong to one Pydantic AI run."""

    def __init__(
        self,
        run_id: str,
    ) -> None:
        self._run_id = run_id
        self._messages: list[ModelMessage] = []
        self._quality = HistoryQuality.COMPLETE

    @property
    def messages(self) -> tuple[ModelMessage, ...]:
        return tuple(self._messages)

    def capture(
        self,
        messages: Sequence[ModelMessage],
        *,
        origin: TranscriptOrigin = TranscriptOrigin.RAW,
    ) -> TranscriptCapture:
        incoming = tuple(messages)
        known = tuple(
            message
            for message in incoming
            if message.run_id == self._run_id
        )
        stored_known = tuple(
            message
            for message in self._messages
            if message.run_id == self._run_id
        )
        if len(known) < len(stored_known):
            self._quality = HistoryQuality.CONSERVATIVE
        overlap = self._overlap(known, stored_known)
        if known and stored_known and overlap == 0:
            self._quality = HistoryQuality.CONSERVATIVE
        delta = list(known[overlap:])
        delta_origins = [origin] * len(delta)
        known_values = list(self._messages)
        for message in incoming:
            message_run_id = message.run_id
            if message_run_id is not None:
                continue
            if message in known_values:
                continue
            delta.append(message)
            delta_origins.append(TranscriptOrigin.UNKNOWN)
            known_values.append(message)
            self._quality = HistoryQuality.CONSERVATIVE
        first = len(self._messages)
        self._messages.extend(delta)
        return TranscriptCapture(
            first,
            tuple(delta),
            tuple(delta_origins),
            self._quality,
        )

    def _overlap(
        self,
        incoming: tuple[ModelMessage, ...],
        stored: tuple[ModelMessage, ...],
    ) -> int:
        if not incoming or not stored:
            return 0
        maximum = min(len(stored), len(incoming))
        for size in range(maximum, 0, -1):
            if stored[-size:] == incoming[:size]:
                return size
        return 0


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
        legacy_message_loader: Callable[[str], Awaitable[tuple[ModelMessage, ...]]] | None = None,
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
        self._legacy_message_loader = legacy_message_loader
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
        current: list[object] = []
        current_start = first_message_index
        for message in values:
            candidate = current + [message]
            raw = ModelMessagesTypeAdapter.dump_json(candidate)
            if current and len(raw) > _CHUNK_TARGET:
                chunks.append(
                    await self._make_chunk(owner_id, current_start, current, origin)
                )
                current = [message]
                current_start = first_message_index + sum(
                    item.message_count for item in chunks
                )
            else:
                current = candidate
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
    ) -> None:
        if not chunks:
            return
        stream = self._transcript_stream(owner_id)
        owner = self._owner_key(owner_id)
        await self._ensure_owner(transaction, owner_id, owner)
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

    async def _ensure_owner(
        self,
        transaction: StateTransaction,
        owner_id: str,
        owner: bytes,
    ) -> None:
        current = await transaction.get_record(owner)
        if current is None:
            await transaction.insert_record(
                StoredRecord(
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
                    {"type": "history_owner", "owner_id": owner_id},
                )
            )
            return
        if await transaction.guard_record(
            owner,
            expected_storage_version=current.storage_version,
        ) is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

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

    async def history_message_count(self, history_id: str, *, tenant_id: str) -> int:
        if self._history_repository is None:
            return 0
        record = await self._history_repository.get(
            history_id,
            tenant_id=tenant_id,
        )
        if record is None:
            raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
        return record.inherited_message_count + record.local_message_count

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
            return await self.load_model_context(
                history_id,
                binding_digest=binding_digest,
            )
        records, legacy_source_id, legacy_messages = await self._history_segments(
            history_id,
            tenant_id=tenant_id,
        )
        values: list[LoadedContextMessage] = []
        if legacy_messages is not None and legacy_source_id is not None:
            values.extend(
                LoadedContextMessage(
                    message,
                    TranscriptMessageRef(
                        RuntimeDomain.CONVERSATION,
                        legacy_source_id,
                        index,
                    ),
                )
                for index, message in enumerate(legacy_messages)
            )
        for record in records:
            start = record.inherited_message_count
            end = start + record.local_message_count
            messages = await self.load_message_span(record.history_id, start, end)
            values.extend(
                LoadedContextMessage(
                    message,
                    TranscriptMessageRef(
                        RuntimeDomain.CONVERSATION,
                        record.history_id,
                        start + index,
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
        records, _legacy_source_id, legacy_messages = await self._history_segments(
            history_id,
            tenant_id=tenant_id,
        )
        if legacy_messages is not None:
            for message in legacy_messages:
                yield message
        for record in records:
            start = record.inherited_message_count
            end = start + record.local_message_count
            async for message in self._iter_range(
                self.history_stream(record.history_id),
                start=start,
                end=end,
            ):
                yield message

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
        if (
            self._runtime_domain is RuntimeDomain.CONVERSATION
            and self._history_repository is not None
            and self._legacy_message_loader is not None
            and await self._history_repository.get(
                owner_id,
                tenant_id=self._tenant_id,
            )
            is None
        ):
            legacy_messages = await self._legacy_message_loader(owner_id)
            return self._validate_spans(
                windows,
                tuple(
                    tuple(
                        legacy_messages[
                            max(start, 0) : min(end, len(legacy_messages))
                        ]
                    )
                    for start, end in windows
                ),
            )
        values: list[list[ModelMessage]] = [[] for _ in windows]
        stop_at = max(end for _, end in windows)
        after_sequence: int | None = None
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
    ) -> tuple[
        tuple[ConversationHistoryRecord, ...],
        str | None,
        tuple[ModelMessage, ...] | None,
    ]:
        if self._history_repository is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        records: list[ConversationHistoryRecord] = []
        current_id: str | None = history_id
        visited: set[str] = set()
        legacy_source_id: str | None = None
        legacy_messages: tuple[ModelMessage, ...] | None = None
        while current_id is not None:
            if current_id in visited:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(current_id)
            record = await self._history_repository.get(
                current_id,
                tenant_id=tenant_id,
            )
            if record is None:
                if not records or self._legacy_message_loader is None:
                    raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
                legacy_source_id = current_id
                legacy_messages = await self._legacy_message_loader(current_id)
                break
            records.append(record)
            current_id = None if record.parent is None else record.parent.history_id
        records.reverse()
        return tuple(records), legacy_source_id, legacy_messages

    async def _iter_range(
        self,
        stream: bytes,
        *,
        start: int,
        end: int,
    ) -> AsyncIterator[ModelMessage]:
        if end <= start:
            return
        expected = end - start
        emitted = 0
        after_sequence: int | None = None
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


__all__ = ["TranscriptRepository"]
