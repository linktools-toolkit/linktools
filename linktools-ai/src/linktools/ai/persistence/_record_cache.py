#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bounded ordinary record reads over source-owned transaction boundaries."""

import asyncio
from collections.abc import Mapping, Sequence
import hashlib
import json

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError
from linktools.ai.storage import CacheBackend, StorageReadCache
from linktools.ai.persistence.codec import decode_record, encode_record
from linktools.ai.persistence.state import StateStore, StoredRecord, active_state_transaction


class StateRecordCache:
    """Share committed record bytes without making cached values a CAS input."""

    def __init__(self, store: StateStore, cache: CacheBackend | None, *, scope: str) -> None:
        self._store = store
        self._scope = scope
        self._epoch = object()
        self._publisher: asyncio.Task[None] | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._cache = StorageReadCache(cache) if isinstance(cache, CacheBackend) else None

    async def read_record(self, key: bytes) -> StoredRecord | None:
        records = await self.read_records((key,))
        return records.get(key)

    async def read_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        ordered = tuple(sorted(set(keys)))
        active = active_state_transaction(self._store)
        if active is not None:
            return await active.get_records(ordered)
        if not ordered:
            return {}
        if self._cache is None:
            return await self._store.read(lambda transaction: transaction.get_records(ordered))

        async def load() -> bytes:
            records = await self._store.read(lambda transaction: transaction.get_records(ordered))
            return canonical_json_bytes([encode_record(records[key]) for key in ordered if key in records])

        cache_key = self._cache_key(ordered)
        content = await self._cache.get(cache_key, load)
        try:
            if content is None:
                raise ValueError("cached record batch is absent")
            values = json.loads(content)
            if not isinstance(values, list):
                raise ValueError("cached record batch must be a list")
            result: dict[bytes, StoredRecord] = {}
            for value in values:
                record = decode_record(value)
                if record.key_digest not in ordered or record.key_digest in result:
                    raise ValueError("cached record identity does not match")
                result[record.key_digest] = record
            return result
        except (AIError, TypeError, ValueError):
            await self._cache.invalidate(cache_key)
            return await self._store.read(lambda transaction: transaction.get_records(ordered))

    async def refresh_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        """Replace a stale ordinary batch from an authoritative source capture."""
        ordered = tuple(sorted(set(keys)))
        active = active_state_transaction(self._store)
        if active is not None:
            return await active.get_records(ordered)
        if not ordered:
            return {}
        if self._cache is None:
            return await self._store.read(lambda transaction: transaction.get_records(ordered))
        records: Mapping[bytes, StoredRecord] = {}

        async def load() -> bytes:
            nonlocal records
            records = await self._store.read(lambda transaction: transaction.get_records(ordered))
            return canonical_json_bytes([encode_record(records[key]) for key in ordered if key in records])

        # Allocate the publication generation before awaiting the source. A later
        # refresh or local mutation must win even when this source returns last.
        await self._cache.refresh(self._cache_key(ordered), load)
        return records

    def _cache_key(self, ordered: Sequence[bytes]) -> str:
        key_digest = hashlib.sha256(canonical_json_bytes([key.hex() for key in ordered])).hexdigest()
        return f"runtime-record:{self._scope}:{key_digest}"

    def publish_record(self, record: StoredRecord, *, captured_at: float) -> None:
        """Offer one complete committed record; busy publishers discard the offer."""
        if self._closed or self._cache is None or (
            self._publisher is not None and not self._publisher.done()
        ):
            return
        epoch = self._epoch

        async def publish() -> None:
            if self._closed or self._epoch is not epoch:
                return
            content = canonical_json_bytes([encode_record(record)])
            await self._cache.put(
                self._cache_key((record.key_digest,)), content, captured_at=captured_at,
            )

        self._publisher = asyncio.create_task(publish(), name="runtime-record-cache-publish")
        self._publisher.add_done_callback(self._consume_publisher)

    def _consume_publisher(self, task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        if self._publisher is task:
            self._publisher = None

    @property
    def capture_token(self) -> object:
        """Owner invalidation boundary at source capture start."""
        return self._epoch

    async def invalidate(self) -> object:
        previous = self._epoch
        self._epoch = object()
        if self._cache is not None:
            await self._cache.clear()
        return previous

    async def close(self) -> None:
        self._closed = True
        self._epoch = object()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned(), name="runtime-record-cache-close")
            self._close_task.add_done_callback(self._consume_close_result)
        await asyncio.shield(self._close_task)

    def _consume_close_result(self, task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def _close_owned(self) -> None:
        publisher = self._publisher
        if publisher is not None:
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
        if self._cache is not None:
            await self._cache.close()


__all__ = ["StateRecordCache"]
