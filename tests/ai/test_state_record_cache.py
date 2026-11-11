#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordinary record caches keep single records, batches and stores independent."""

import asyncio
import hashlib
from time import time

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._record_cache import StateRecordCache
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.storage import InMemoryCacheBackend


def _record(key: bytes, value: str) -> StoredRecord:
    return StoredRecord(key, None, None, "record", value, None, 0, None, 0, None, {"value": value})


@pytest.mark.asyncio
async def test_single_and_batch_records_share_no_cache_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _record(b"a" * 32, "first"), _record(b"b" * 32, "second")
    batch_key = hashlib.sha256(canonical_json_bytes([first.key_digest.hex(), second.key_digest.hex()])).digest()
    third = _record(batch_key, "third")
    store = InMemoryStateStore()
    await store.initialize()
    cache = StateRecordCache(store, InMemoryCacheBackend(max_bytes=100_000), scope="records")

    async def insert(transaction: StateTransaction) -> None:
        for record in (first, second, third):
            await transaction.insert_record(record)

    try:
        await store.mutate(insert)
        expected = {record.key_digest: record for record in (first, second)}
        assert await cache.read_records(tuple(expected)) == expected
        assert await cache.read_record(third.key_digest) == third
        assert await cache.read_record(first.key_digest) == first

        async def unavailable(callback):
            raise AssertionError("fresh cache entries must not require a source read")

        monkeypatch.setattr(store, "read", unavailable)
        assert await cache.read_records((second.key_digest, first.key_digest, second.key_digest)) == expected
        assert await cache.read_record(third.key_digest) == third
        assert await cache.read_records((first.key_digest, first.key_digest)) == {first.key_digest: first}
    finally:
        await cache.close()
        await store.close()


@pytest.mark.asyncio
async def test_published_record_is_reused_only_within_its_store(monkeypatch: pytest.MonkeyPatch) -> None:
    published = asyncio.Event()

    class Cache(InMemoryCacheBackend):
        async def put(self, key: str, content: bytes) -> None:
            await super().put(key, content)
            published.set()

    backend = Cache(max_bytes=100_000)
    store = InMemoryStateStore()
    await store.initialize()
    owner = StateRecordCache(store, backend, scope="owner")
    other = StateRecordCache(store, backend, scope="other")
    record = _record(b"k" * 32, "published")
    try:
        owner.publish_record(record, captured_at=time())
        await asyncio.wait_for(published.wait(), timeout=5)
        assert await other.read_record(record.key_digest) is None

        async def unavailable(callback):
            raise AssertionError("a published record must be readable without loading the source")

        monkeypatch.setattr(store, "read", unavailable)
        assert await owner.read_record(record.key_digest) == record
    finally:
        await owner.close()
        await other.close()
        await store.close()
