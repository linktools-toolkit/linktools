#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded, source-confirmed hot views over the shared cache budget."""

import asyncio
from collections.abc import Hashable, Sequence

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import (
    InMemoryCacheBackend,
    StorageBatchResult,
    StorageBuffer,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageOperation,
    StoragePutResult,
    StorageReadCache,
    StorageRevision,
    TieredCacheBackend,
)
from linktools.ai.storage import _write_cache as write_cache_module


class _Source:
    def __init__(self) -> None:
        self.values: dict[Hashable, bytes] = {"key": b"old"}
        self.reads = 0
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.read_release.set()
        self.failure: Exception | None = None

    async def get(self, key: Hashable) -> bytes | None:
        self.reads += 1
        value = self.values.get(key)
        self.read_started.set()
        await self.read_release.wait()
        return value

    async def apply_batch(
        self, changes: Sequence[StorageChange[Hashable, bytes]],
    ) -> StorageBatchResult[bytes, Hashable]:
        if self.failure is not None:
            raise self.failure
        revision = StorageRevision("committed")
        results: list[StoragePutResult[bytes] | StorageDeleteResult[Hashable]] = []
        for change in changes:
            if change.operation is StorageOperation.DELETE:
                deleted = self.values.pop(change.key, None) is not None
                results.append(StorageDeleteResult(change.key, deleted, None, revision))
            else:
                assert change.value is not None
                self.values[change.key] = change.value
                results.append(StoragePutResult(change.value, StorageEntryRevision(1), revision, True))
        return StorageBatchResult(revision, True, tuple(results))


@pytest.mark.asyncio
async def test_warm_reads_skip_source_until_reader_enforced_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(write_cache_module, "time", lambda: now[0])
    source = _Source()
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=1024))
    assert await buffer.get("key") == b"old"
    source.values["key"] = b"remote"
    now[0] += 2.999
    assert await buffer.get("key") == b"old"
    assert source.reads == 1
    now[0] = 103.0
    assert await buffer.get("key") == b"remote"
    assert await buffer.get("key") == b"remote"
    assert source.reads == 2
    await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_pending", [False, True])
async def test_pending_and_confirmed_hot_views_remain_distinct(read_pending: bool) -> None:
    source = _Source()
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=read_pending)
    assert await buffer.get("key") == b"old"
    await buffer.enqueue("key", b"new")
    assert await buffer.get("key") == (b"new" if read_pending else b"old")
    await buffer.flush()
    assert await buffer.get("key") == b"new"
    await buffer.enqueue_delete("key")
    assert await buffer.get("key") == (None if read_pending else b"new")
    await buffer.flush()
    assert await buffer.get("key") is None
    assert source.reads == 3
    await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_pending", [False, True])
async def test_late_source_snapshot_cannot_replace_newly_flushed_hot_value(read_pending: bool) -> None:
    source = _Source()
    source.read_release.clear()
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=read_pending)
    read = asyncio.create_task(buffer.get("key"))
    await source.read_started.wait()
    await buffer.enqueue("key", b"new")
    await buffer.flush()
    source.read_release.set()
    assert await read == (b"new" if read_pending else b"old")
    assert await buffer.get("key") == b"new"
    assert source.reads == 2
    await buffer.close()


@pytest.mark.asyncio
async def test_hash_collisions_and_shared_cache_do_not_alias_owner_keys() -> None:
    class Key:
        def __hash__(self) -> int:
            return 1

        def __repr__(self) -> str:
            return "same"

    first, second = Key(), Key()
    cache = InMemoryCacheBackend(max_bytes=1024)
    source = _Source()
    source.values.update({first: b"first", second: b"second"})
    other_source = _Source()
    other_source.values[first] = b"other-owner"
    buffer = StorageBuffer(source, cache=cache)
    other = StorageBuffer(other_source, cache=cache)
    for _ in range(2):
        assert await buffer.get(first) == b"first"
        assert await buffer.get(second) == b"second"
        assert await other.get(first) == b"other-owner"
    assert source.reads == 2
    assert other_source.reads == 1
    await asyncio.gather(buffer.close(), other.close())


@pytest.mark.asyncio
async def test_hot_key_metadata_is_bounded_without_losing_pending_intentions() -> None:
    source = _Source()
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=100_000), read_pending=True)
    await buffer.enqueue("pending", b"latest")
    for key in range(140):
        source.values[key] = b"value"
        assert await buffer.get(key) == b"value"
    # These maps protect distinct memory bounds: active work and hot metadata.
    assert len(buffer._hot_keys) <= 128
    assert len(buffer._keys) == 1
    assert await buffer.get("pending") == b"latest"
    assert await buffer.get(139) == b"value"
    assert source.reads == 140
    assert await buffer.get(0) == b"value"
    assert source.reads == 141
    await buffer.flush()
    assert await buffer.get("pending") == b"latest"
    assert buffer._keys == {}
    await buffer.close()


@pytest.mark.asyncio
async def test_pending_payloads_evict_hot_content_and_keep_the_shared_capacity_bound() -> None:
    source = _Source()
    cache = InMemoryCacheBackend(max_bytes=64, max_size=1)
    buffer = StorageBuffer(source, cache=cache, read_pending=True)
    assert await buffer.get("key") == b"old"
    await buffer.enqueue("key", b"x" * 64)
    assert await buffer.get("key") == b"x" * 64
    assert source.reads == 1
    assert not await cache.append("other-owner", "blocked", "key", b"x", wait=False)
    await buffer.flush()
    assert await buffer.get("key") == b"x" * 64
    assert source.reads == 2
    assert await cache.append("other-owner", "available", "key", b"x" * 64, wait=False)
    await cache.ack("other-owner", ("available",))
    await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_only", [False, True])
async def test_missing_hot_capability_keeps_source_reads(pending_only: bool) -> None:
    class PendingOnly:
        def __init__(self) -> None:
            cache = InMemoryCacheBackend(max_bytes=1024)
            self.append = cache.append
            self.read_batch = cache.read_batch
            self.ack = cache.ack

    source = _Source()
    buffer = StorageBuffer(source, cache=PendingOnly() if pending_only else None)
    assert await buffer.get("key") == b"old"
    source.values["key"] = b"remote"
    assert await buffer.get("key") == b"remote"
    assert source.reads == 2
    await buffer.enqueue("key", b"new")
    await buffer.flush()
    assert await buffer.get("key") == b"new"
    assert source.reads == 3
    await buffer.close()


@pytest.mark.asyncio
async def test_hot_only_cache_keeps_direct_writes_and_confirmed_reads() -> None:
    source = _Source()
    buffer = StorageBuffer(source, cache=TieredCacheBackend(InMemoryCacheBackend(max_bytes=1024)))
    await buffer.enqueue("key", b"")
    assert source.values["key"] == b""
    assert await buffer.get("key") == b""
    assert source.reads == 1
    await buffer.close()


@pytest.mark.asyncio
async def test_failed_sink_never_promotes_intention_and_bypasses_old_hot_content() -> None:
    source = _Source()
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=1024), read_pending=True)
    assert await buffer.get("key") == b"old"
    source.failure = RuntimeError("sink failed")
    await buffer.enqueue("key", b"unconfirmed")
    with pytest.raises(RuntimeError, match="sink failed"):
        await buffer.flush()
    source.values["key"] = b"authoritative"
    assert await buffer.get("key") == b"authoritative"
    assert source.reads == 2
    with pytest.raises(RuntimeError, match="sink failed"):
        await buffer.enqueue("key", b"rejected")
    with pytest.raises(RuntimeError, match="sink failed"):
        await buffer.close()


@pytest.mark.asyncio
async def test_incomplete_batch_receipt_is_unknown_and_cannot_promote_hot_content() -> None:
    class IncompleteSource(_Source):
        async def apply_batch(
            self, changes: Sequence[StorageChange[Hashable, bytes]],
        ) -> StorageBatchResult[bytes, Hashable]:
            return StorageBatchResult(StorageRevision("partial"), False, ())

    source = IncompleteSource()
    buffer = StorageBuffer(source, cache=InMemoryCacheBackend(max_bytes=1024))
    await buffer.enqueue("key", b"unconfirmed")
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.flush()
    assert await buffer.get("key") == b"old"
    with pytest.raises(AIError, match=ErrorCode.STORAGE_COMMIT_UNKNOWN.value):
        await buffer.close()


@pytest.mark.asyncio
async def test_shared_envelope_expiry_survives_tier_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(write_cache_module, "time", lambda: now[0])
    l1 = InMemoryCacheBackend(max_bytes=1024)
    l2 = InMemoryCacheBackend(max_bytes=1024)
    writer = StorageReadCache(l2)
    reader = StorageReadCache(TieredCacheBackend(l1, l2))
    source = _Source()
    await writer.put("shared", b"cached")
    now[0] = 102.0
    assert await reader.get("shared", lambda: source.get("key")) == b"cached"
    assert source.reads == 0
    now[0] = 103.0
    assert await reader.get("shared", lambda: source.get("key")) == b"old"
    assert source.reads == 1
    await writer.close()
    await reader.close()


@pytest.mark.asyncio
async def test_newer_shared_publication_remains_a_hit_for_an_existing_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(write_cache_module, "time", lambda: now[0])
    cache = InMemoryCacheBackend(max_bytes=1024)
    reader = StorageReadCache(cache)
    writer = StorageReadCache(cache)
    source = _Source()
    assert await reader.get("key", lambda: source.get("key")) == b"old"
    now[0] += 1
    await writer.put("key", b"remote")
    assert await reader.get("key", lambda: source.get("key")) == b"remote"
    assert source.reads == 1
    await reader.close()
    await writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["put", "invalidate", "clear"])
async def test_local_mutation_blocks_late_source_fill(operation: str) -> None:
    source = _Source()
    source.read_release.clear()
    view = StorageReadCache(InMemoryCacheBackend(max_bytes=1024))
    read = asyncio.create_task(view.get("key", lambda: source.get("key")))
    await source.read_started.wait()
    source.values["key"] = b"new"
    if operation == "put":
        await view.put("key", b"new")
    elif operation == "invalidate":
        await view.invalidate("key")
    else:
        await view.clear()
    source.read_release.set()
    assert await read == b"old"
    assert await view.get("key", lambda: source.get("key")) == b"new"
    assert source.reads == (1 if operation == "put" else 2)
    await view.close()


@pytest.mark.asyncio
async def test_failed_invalidation_and_declined_put_cannot_revive_an_old_value() -> None:
    class UnavailableCache(InMemoryCacheBackend):
        unavailable = False

        async def delete(self, key: str) -> None:
            if self.unavailable:
                raise OSError("cache unavailable")
            await super().delete(key)

        async def put(self, key: str, content: bytes) -> None:
            if not self.unavailable:
                await super().put(key, content)

    cache = UnavailableCache(max_bytes=1024)
    source = _Source()
    view = StorageReadCache(cache)
    assert await view.get("key", lambda: source.get("key")) == b"old"
    cache.unavailable = True
    source.values["key"] = b"new"
    await view.invalidate("key")
    assert await view.get("key", lambda: source.get("key")) == b"new"
    await view.put("key", b"new")
    assert await view.get("key", lambda: source.get("key")) == b"new"
    await view.clear()
    assert await view.get("key", lambda: source.get("key")) == b"new"
    # Forgetting a bounded mutation token cannot make failed deletion succeed.
    await view.invalidate("key")
    for number in range(140):
        await view.get(str(number), lambda: source.get(number))
    assert await view.get("key", lambda: source.get("key")) == b"new"
    await view.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("slow", "error", "cancelled"))
async def test_confirmed_write_does_not_depend_on_disposable_hot_cache(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    monkeypatch.setattr(write_cache_module, "_WRITE_TIMEOUT_SECONDS", 0.03)
    blocked = asyncio.Event()

    class UnavailableHotCache(InMemoryCacheBackend):
        unavailable = False
        calls = 0

        async def put(self, key: str, content: bytes) -> None:
            self.calls += 1
            if self.unavailable:
                if failure == "slow":
                    await blocked.wait()
                elif failure == "error":
                    raise OSError("hot cache unavailable")
                else:
                    raise asyncio.CancelledError
            await super().put(key, content)

    source = _Source()
    cache = UnavailableHotCache(max_bytes=4096)
    buffer = StorageBuffer(source, cache=cache)
    assert await buffer.get("key") == b"old"
    cache.unavailable = True
    await buffer.enqueue("key", b"new")
    await asyncio.wait_for(buffer.flush(), 0.2)
    assert source.values["key"] == b"new"
    assert cache.calls == 1
    assert buffer._scope is not None
    status = buffer._scope.status()
    assert status.confirmed == 1 and status.unconfirmed == 0
    await buffer.enqueue("key", b"latest")
    await asyncio.wait_for(buffer.flush(), 0.2)
    assert source.values["key"] == b"latest"
    cache.unavailable = False
    assert await buffer.get("key") == b"latest"
    assert await buffer.get("key") == b"latest"
    assert source.reads == 2
    await buffer.close()
