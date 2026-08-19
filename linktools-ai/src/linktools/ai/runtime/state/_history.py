#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical transcript chunks and bounded context projections."""

import hashlib
import zlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from linktools.core import environ

from ...core import canonical_json_bytes
from ...storage import ObjectRef, ObjectStore, StoredPayload
from ._codec import decode_domain, encode_domain
from ._contracts import (
    ContextProjection,
    InlineContextBlock,
    RuntimePayloadRef,
    TranscriptChunk,
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


@dataclass(frozen=True, slots=True)
class TranscriptCapture:
    first_message_index: int
    messages: tuple[ModelMessage, ...]
    origins: tuple[TranscriptOrigin, ...]


class _TranscriptAccumulator:
    """Merge overlapping provider snapshots without rewriting prior chunks."""

    def __init__(
        self,
        messages: Sequence[ModelMessage] = (),
        origins: Sequence[TranscriptOrigin] = (),
    ) -> None:
        self._messages = list(messages)
        self._origins = list(origins)
        if len(self._origins) < len(self._messages):
            self._origins.extend(
                [TranscriptOrigin.RAW] * (len(self._messages) - len(self._origins))
            )

    @property
    def message_count(self) -> int:
        return len(self._messages)

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
        if len(incoming) < len(self._messages):
            return TranscriptCapture(len(self._messages), (), ())
        overlap = self._overlap(incoming)
        first = len(self._messages)
        delta = incoming[overlap:]
        self._messages.extend(delta)
        self._origins.extend([origin] * len(delta))
        return TranscriptCapture(first, delta, tuple([origin] * len(delta)))

    def _overlap(self, incoming: tuple[ModelMessage, ...]) -> int:
        if not incoming or not self._messages:
            return 0
        maximum = min(len(self._messages), len(incoming))
        for size in range(maximum, 0, -1):
            if self._messages[-size:] == list(incoming[:size]):
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
        canonical_message_count: int,
        origins: Sequence[TranscriptOrigin] = (),
        source_domain: RuntimeDomain | None = None,
    ) -> ContextProjection:
        values = tuple(messages)
        origin_values = tuple(origins)
        items: list[TranscriptSpanRef | InlineContextBlock] = []
        span_domain = source_domain or self._runtime_domain
        index = 0
        while index < len(values):
            origin = (
                origin_values[index]
                if index < len(origin_values)
                else TranscriptOrigin.RAW
            )
            end = index + 1
            while end < len(values):
                next_origin = (
                    origin_values[end]
                    if end < len(origin_values)
                    else TranscriptOrigin.RAW
                )
                if next_origin is not origin:
                    break
                end += 1
            if origin is TranscriptOrigin.RAW and end <= canonical_message_count:
                items.append(
                    TranscriptSpanRef(
                        span_domain,
                        owner_id,
                        index,
                        end,
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
    ) -> None:
        self._store = store
        self._object_store = object_store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._runtime_domain = runtime_domain
        self._context_sources = dict(context_sources or {})
        self._projector = _ContextProjector(runtime_domain)

    @property
    def runtime_domain(self) -> RuntimeDomain:
        return self._runtime_domain

    async def prepare_chunks(
        self,
        run_id: str,
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
                    await self._make_chunk(run_id, current_start, current, origin)
                )
                current = [message]
                current_start = first_message_index + sum(
                    item.message_count for item in chunks
                )
            else:
                current = candidate
        if current:
            chunks.append(await self._make_chunk(run_id, current_start, current, origin))
        return tuple(chunks)

    async def _make_chunk(
        self,
        run_id: str,
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
            run_id,
            raw_digest,
            content,
            raw_size=len(raw),
        )
        return TranscriptChunk(
            run_id,
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
        run_id: str,
        digest: str,
        value: bytes,
        *,
        raw_size: int,
    ) -> StoredPayload:
        if self._object_store is None or raw_size < _COMPRESS_MINIMUM:
            return StoredPayload.inline_bytes(value)
        key = f"transcript/{self._tenant_id}/{run_id}/{digest}"

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
        run_id: str,
        chunks: Sequence[TranscriptChunk],
    ) -> None:
        if not chunks:
            return
        stream = self._transcript_stream(run_id)
        owner = self._run_key(run_id)
        final = await transaction.reserve_sequence(
            self._transcript_sequence(run_id),
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
                f"context/{self._tenant_id}/{run_id}/{item.content.payload.digest}",
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

    def transcript_stream(self, run_id: str) -> bytes:
        return self._transcript_stream(run_id)

    async def latest_chunk(self, run_id: str) -> TranscriptChunk | None:
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(self._transcript_stream(run_id), latest=True)
            )
        )
        if not values:
            return None
        return self.decode_chunk(values[0])

    async def iter_messages(self, run_id: str) -> AsyncIterator[ModelMessage]:
        after_sequence: int | None = None
        while True:
            values = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        self._transcript_stream(run_id),
                        after_sequence=after_sequence,
                        limit=1,
                    )
                )
            )
            if not values:
                return
            fact = values[0]
            after_sequence = fact.sequence
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

    async def load_messages(self, run_id: str) -> tuple[ModelMessage, ...]:
        return tuple([message async for message in self.iter_messages(run_id)])

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
            self._run_key(run_id),
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

    async def load_projection(self, run_id: str) -> ContextProjection | None:
        value = await self._store.read(
            lambda transaction: transaction.get_record(self._projection_key(run_id))
        )
        if value is None:
            return None
        payload = value.data.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("context projection payload is invalid")
        return decode_domain(payload, ContextProjection)

    async def load_model_context(
        self,
        run_id: str,
        *,
        binding_digest: str | None = None,
    ) -> tuple[ModelMessage, ...]:
        projection = await self.load_projection(run_id)
        if projection is None:
            return await self.load_messages(run_id)
        if binding_digest is not None and projection.binding_digest != binding_digest:
            _logger.info(
                "context projection binding changed; rebuilding from canonical refs: "
                "domain=%s run=%s",
                self._runtime_domain.value,
                run_id,
            )
            projection = self.rebind_projection(
                projection,
                binding_digest=binding_digest,
            )
        span_groups: dict[
            tuple[TranscriptRepository, str],
            list[tuple[int, int]],
        ] = {}
        inline_values: list[tuple[ModelMessage, ...]] = []
        for item in projection.items:
            if isinstance(item, TranscriptSpanRef):
                source = self
                if item.source_domain is not self._runtime_domain:
                    source = self._context_sources.get(item.source_domain)
                    if source is None:
                        raise ValueError("transcript context source is unavailable")
                span_groups.setdefault((source, item.owner_id), []).append(
                    (item.start, item.end)
                )
                continue
            raw = await self._read_payload(item.content.payload)
            inline_values.append(tuple(ModelMessagesTypeAdapter.validate_json(raw)))
        span_values: dict[
            tuple[TranscriptRepository, str],
            tuple[tuple[ModelMessage, ...], ...],
        ] = {}
        for key, windows in span_groups.items():
            source, owner_id = key
            span_values[key] = await source.load_message_spans(owner_id, windows)
        values: list[object] = []
        span_offsets: dict[tuple[TranscriptRepository, str], int] = {}
        inline_index = 0
        for item in projection.items:
            if isinstance(item, TranscriptSpanRef):
                source = self
                if item.source_domain is not self._runtime_domain:
                    source = self._context_sources.get(item.source_domain)
                    if source is None:
                        raise ValueError("transcript context source is unavailable")
                key = (source, item.owner_id)
                offset = span_offsets.get(key, 0)
                values.extend(span_values[key][offset])
                span_offsets[key] = offset + 1
            else:
                values.extend(inline_values[inline_index])
                inline_index += 1
        return tuple(values)

    async def load_message_span(
        self,
        run_id: str,
        start: int,
        end: int,
    ) -> tuple[ModelMessage, ...]:
        return (await self.load_message_spans(run_id, ((start, end),)))[0]

    async def load_message_spans(
        self,
        run_id: str,
        windows: Sequence[tuple[int, int]],
    ) -> tuple[tuple[ModelMessage, ...], ...]:
        if not windows:
            return ()
        values: list[list[object]] = [[] for _ in windows]
        after_sequence: int | None = None
        while True:
            facts = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        self._transcript_stream(run_id),
                        after_sequence=after_sequence,
                        limit=1,
                    )
                )
            )
            if not facts:
                break
            fact = facts[0]
            after_sequence = fact.sequence
            chunk = self.decode_chunk(fact)
            chunk_start = chunk.first_message_index
            chunk_end = chunk_start + chunk.message_count
            if all(chunk_start >= end for _, end in windows):
                break
            if all(chunk_end <= start or chunk_start >= end for start, end in windows):
                continue
            messages = await self._decode_chunk_messages(chunk)
            for index, (start, end) in enumerate(windows):
                if chunk_end <= start or chunk_start >= end:
                    continue
                left = max(start - chunk_start, 0)
                right = min(end - chunk_start, len(messages))
                values[index].extend(messages[left:right])
        return tuple(tuple(value) for value in values)

    def project_context(
        self,
        run_id: str,
        messages: Sequence[ModelMessage],
        *,
        binding_digest: str,
        canonical_message_count: int,
        origins: Sequence[TranscriptOrigin] = (),
        source_domain: RuntimeDomain | None = None,
    ) -> ContextProjection:
        return self._projector.project(
            run_id,
            messages,
            binding_digest=binding_digest,
            canonical_message_count=canonical_message_count,
            origins=origins,
            source_domain=source_domain,
        )

    def rebind_projection(
        self,
        projection: ContextProjection,
        *,
        binding_digest: str,
    ) -> ContextProjection:
        digest = self._projector._digest(binding_digest, projection.items)
        return ContextProjection(binding_digest, projection.items, digest)

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

    def _run_key(self, run_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "step_run",
            run_id,
        )

    def _transcript_stream(self, run_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "transcript",
            run_id,
        )

    def _transcript_sequence(self, run_id: str) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            self._runtime_domain.value,
            "transcript",
            run_id,
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
