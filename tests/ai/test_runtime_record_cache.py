#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Ordinary record hot reads never replace committed control scopes."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.runtime.state._store import StateTransactionNestingError, StoredRecord
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.storage import _write_cache

pytestmark = pytest.mark.asyncio


def _record(value: int, identity: str = "record") -> StoredRecord:
    return StoredRecord(hashlib.sha256(identity.encode()).digest(), None, None,
                        "probe", identity, None, value - 1, None, 0, None,
                        {"value": value})


@asynccontextmanager
async def _store(backend: str, root: Path, cache):
    engine = None
    if backend == "memory":
        store = InMemoryStateStore(cache=cache)
    elif backend == "filesystem":
        store = FilesystemStateStore(root, namespace="cache", tenant_id="tenant",
                                     runtime_domain="conversation", cache=cache)
    else:
        root.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'state.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine, cache=cache)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_record_hot_reads_skip_source_but_control_scope_and_mutation_do_not(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _store(backend, tmp_path / backend, InMemoryCacheBackend(max_bytes=100000)) as store:
        first = _record(1)
        await store.mutate(lambda tx: tx.insert_record(first))
        calls = 0
        source_read = store.read

        async def read(fn):
            nonlocal calls
            calls += 1
            return await source_read(fn)

        monkeypatch.setattr(store, "read", read)
        assert await store.read_record(first.key_digest) == first
        assert await store.read_record(first.key_digest) == first
        assert await store.read_records((first.key_digest,)) == {first.key_digest: first}
        assert calls == 1

        async def committed(tx):
            assert await store.read_record(first.key_digest) == await tx.get_record(first.key_digest)
            assert await store.read_records((first.key_digest,)) == await tx.get_records((first.key_digest,))
            with pytest.raises(StateTransactionNestingError):
                await asyncio.create_task(store.read_record(first.key_digest))

        await store.read(committed)
        assert calls == 2
        second = _record(2)
        await store.mutate(lambda tx: tx.replace_record(second, expected_storage_revision=0))
        assert await store.read_record(first.key_digest) == second
        assert calls == 3
        await store.close()
        with pytest.raises(AIError) as raised:
            await store.read_record(first.key_digest)
        assert raised.value.code is ErrorCode.STORAGE_CLOSED


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_record_cache_without_backend_preserves_direct_source(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _store(backend, tmp_path / backend, None) as store:
        record = _record(1)
        await store.mutate(lambda tx: tx.insert_record(record))
        calls = 0
        source_read = store.read

        async def read(fn):
            nonlocal calls
            calls += 1
            return await source_read(fn)

        monkeypatch.setattr(store, "read", read)
        assert await store.read_record(record.key_digest) == record
        assert await store.read_record(record.key_digest) == record
        assert calls == 2


async def test_sql_shared_record_cache_expires_without_backend_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])
    cache = InMemoryCacheBackend(max_bytes=100000)
    async with _store("sql", tmp_path, cache) as first:
        record = _record(1)
        await first.mutate(lambda tx: tx.insert_record(record))
        assert await first.read_record(record.key_digest) == record
        async with _store("sql", tmp_path, cache) as second:
            reads = 0
            source_read = second.read

            async def read(fn):
                nonlocal reads
                reads += 1
                return await source_read(fn)

            monkeypatch.setattr(second, "read", read)
            assert await second.read_record(record.key_digest) == record
            assert reads == 0
            clock[0] += 3.1
            assert await second.read_record(record.key_digest) == record
            assert reads == 1


async def test_record_cache_does_not_hide_poisoned_filesystem_owner(tmp_path: Path) -> None:
    async with _store("filesystem", tmp_path, InMemoryCacheBackend(max_bytes=100000)) as store:
        record = _record(1)
        await store.mutate(lambda tx: tx.insert_record(record))
        assert await store.read_record(record.key_digest) == record
        store._poisoned = True
        with pytest.raises(AIError) as raised:
            await store.read_record(record.key_digest)
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        store._poisoned = False


async def test_distinct_memory_sources_do_not_share_equal_physical_record_keys() -> None:
    cache = InMemoryCacheBackend(max_bytes=100000)
    first, second = InMemoryStateStore(cache=cache), InMemoryStateStore(cache=cache)
    await first.initialize()
    await second.initialize()
    try:
        old = _record(1)
        other = replace(old, data={"value": 2})
        await first.mutate(lambda tx: tx.insert_record(old))
        await second.mutate(lambda tx: tx.insert_record(other))
        assert await first.read_record(old.key_digest) == old
        assert await second.read_record(other.key_digest) == other
    finally:
        await first.close()
        await second.close()


async def test_record_commit_does_not_wait_for_disposable_hot_delete() -> None:
    blocked = asyncio.Event()

    class Cache(InMemoryCacheBackend):
        async def delete(self, key: str) -> None:
            await blocked.wait()
            await super().delete(key)

    store = InMemoryStateStore(cache=Cache(max_bytes=100000))
    await store.initialize()
    try:
        first = _record(1)
        await store.mutate(lambda tx: tx.insert_record(first))
        assert await store.read_record(first.key_digest) == first
        second = _record(2)
        await asyncio.wait_for(
            store.mutate(lambda tx: tx.replace_record(second, expected_storage_revision=0)),
            timeout=0.2,
        )
        assert await store.read_record(first.key_digest) == second
    finally:
        blocked.set()
        await store.close()


async def test_batch_fill_cannot_publish_snapshot_after_local_commit() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Cache(InMemoryCacheBackend):
        blocked = True

        async def put(self, key: str, content: bytes) -> None:
            if self.blocked:
                self.blocked = False
                started.set()
                await release.wait()
            await super().put(key, content)

    store = InMemoryStateStore(cache=Cache(max_bytes=100000))
    await store.initialize()
    first, second = _record(1, "first"), _record(1, "second")
    keys = (first.key_digest, second.key_digest)
    await store.mutate(lambda tx: tx.insert_records((first, second)))
    read = asyncio.create_task(store.read_records(keys))
    try:
        await started.wait()
        changed = _record(2, "second")
        await asyncio.wait_for(
            store.mutate(lambda tx: tx.replace_record(changed, expected_storage_revision=0)),
            timeout=0.2,
        )
        release.set()
        assert (await read)[second.key_digest] == second
        assert (await store.read_records(keys))[second.key_digest] == changed
        assert await store.read_record(second.key_digest) == changed
    finally:
        release.set()
        await read
        await store.close()


async def test_batch_cache_uses_original_capture_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(_write_cache, "time", lambda: clock[0])

    class Cache(InMemoryCacheBackend):
        async def put(self, key: str, content: bytes) -> None:
            clock[0] = 102.0
            await super().put(key, content)

    store = InMemoryStateStore(cache=Cache(max_bytes=100000))
    await store.initialize()
    first, second = _record(1, "first"), _record(1, "second")
    keys = (first.key_digest, second.key_digest)
    await store.mutate(lambda tx: tx.insert_records((first, second)))
    reads = 0
    source_read = store.read

    async def read(fn):
        nonlocal reads
        reads += 1
        return await source_read(fn)

    monkeypatch.setattr(store, "read", read)
    try:
        assert await store.read_records(keys) == {first.key_digest: first, second.key_digest: second}
        assert reads == 1
        clock[0] = 104.0
        assert await store.read_records(keys) == {first.key_digest: first, second.key_digest: second}
        assert reads == 2
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ("filesystem", "sql"))
async def test_record_cache_checks_storage_group_lifetime(backend: str, tmp_path: Path) -> None:
    async with _store(backend, tmp_path / backend, InMemoryCacheBackend(max_bytes=100000)) as store:
        record = _record(1)
        await store.mutate(lambda tx: tx.insert_record(record))
        assert await store.read_record(record.key_digest) == record
        group = store.storage_group
        group._closed = True
        try:
            with pytest.raises(AIError) as raised:
                await store.read_record(record.key_digest)
            assert raised.value.code is ErrorCode.STORAGE_CLOSED
        finally:
            group._closed = False


async def test_record_cache_without_hot_capability_does_not_roundtrip_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.state import _record_cache

    store = InMemoryStateStore(cache=object())
    await store.initialize()
    record = _record(1)
    await store.mutate(lambda tx: tx.insert_record(record))

    def unexpected_codec(*args):
        raise AssertionError("direct source reads do not encode cache bytes")

    monkeypatch.setattr(_record_cache, "canonical_json_bytes", unexpected_codec)
    try:
        assert await store.read_record(record.key_digest) == record
    finally:
        await store.close()


async def test_sql_close_can_retry_after_cache_waiter_cancellation(tmp_path: Path) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Cache(InMemoryCacheBackend):
        blocked = False

        async def get(self, key: str) -> bytes | None:
            if self.blocked:
                started.set()
                await release.wait()
            return await super().get(key)

    cache = Cache(max_bytes=100000)
    async with _store("sql", tmp_path, cache) as store:
        record = _record(1)
        await store.mutate(lambda tx: tx.insert_record(record))
        assert await store.read_record(record.key_digest) == record
        cache.blocked = True
        read = asyncio.create_task(store.read_record(record.key_digest))
        await started.wait()
        close = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        release.set()
        await asyncio.gather(read, return_exceptions=True)
        await store.close()
        assert store._closed and store.storage_group._closed
