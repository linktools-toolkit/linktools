#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable cache misses share in-flight origin work without caching failures."""

import asyncio
import hashlib
import gc
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetCacheAdapter, AssetInfo, AssetKey, AssetStore, InMemoryAssetBackend, SqlAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import FilesystemContentCache, InMemoryContentCache, StorageOverlay


from tests.ai._storage_stream_helpers import collect_values


class _PausedBackend(InMemoryAssetBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Queue[tuple[AssetKey, ...]] = asyncio.Queue()
        self.release = asyncio.Event()
        self.loads: list[tuple[AssetKey, ...]] = []
        self.failure: Exception | None = None
        self.closed = False

    async def get_many(self, keys: Sequence[AssetKey]) -> AsyncIterator[bytes | None]:
        values = await collect_values(super().get_many(keys))
        self.loads.append(tuple(keys))
        self.started.put_nowait(tuple(keys))
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        for value in values:
            yield value

    async def close(self) -> None:
        self.closed = True
        await super().close()


class _ObservedOverlay(StorageOverlay):
    def __init__(self, backend, **kwargs):
        super().__init__(backend, **kwargs)
        self.entered = asyncio.Queue()

    async def _cache_reads(self, keys, state):
        self.entered.put_nowait(None)
        return await super()._cache_reads(keys, state)


class _Validator:
    def validate_value(self, key: AssetKey, value: bytes, info: AssetInfo) -> None:
        if hashlib.sha256(value).hexdigest() != info.digest or len(value) != info.size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _fixture():
    backend = _PausedBackend()
    cache = InMemoryContentCache(max_bytes=1024 * 1024)
    overlay = _ObservedOverlay(backend, writer=backend, content_cache=cache,
                               cache_adapter=AssetCacheAdapter(), validator=_Validator())
    store = AssetStore(overlay)
    await store.initialize()
    keys = tuple(AssetKey("test", name) for name in ("a", "b", "c"))
    for key in keys:
        value = bytearray(key.id.encode())
        await store.put(key, value)
        value[:] = b"changed"
    return backend, overlay, store, keys


@pytest.mark.asyncio
async def test_overlapping_batches_share_each_immutable_origin_and_keep_bulk() -> None:
    backend, overlay, store, (a, b, c) = await _fixture()
    first = asyncio.create_task(collect_values(store.get_many((a, b, a))))
    await backend.started.get()
    second = asyncio.create_task(collect_values(store.get_many((b, c))))
    await backend.started.get()
    scalar = asyncio.create_task(store.get(b))
    for _ in range(3):
        await overlay.entered.get()
    backend.release.set()
    try:
        assert await first == (b"a", b"b", b"a")
        assert await second == (b"b", b"c")
        assert await scalar == b"b"
        assert Counter(key for batch in backend.loads for key in batch) == Counter((a, b, c))
        assert any(len(batch) > 1 for batch in backend.loads)
        assert await collect_values(store.get_many((a, b, c))) == (b"a", b"b", b"c")
        await overlay.content_cache.delete(overlay.cache_adapter.cache_key(b, await backend.stat(b)))
        backend.loads.clear()
        assert await asyncio.gather(collect_values(store.get_many((a, b))), store.get(b)) == [(b"a", b"b"), b"b"]
        assert Counter(key for batch in backend.loads for key in batch) == Counter((b,))
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("filesystem", (False, True))
async def test_preload_rechecks_eviction_without_reading_warm_content(tmp_path: Path, filesystem: bool) -> None:
    class ObservedCache(FilesystemContentCache if filesystem else InMemoryContentCache):
        reads = 0
        async def get(self, key):
            self.reads += 1
            return await super().get(key)
    backend = _PausedBackend()
    backend.release.set()
    cache = ObservedCache(max_bytes=len(AssetCacheAdapter().to_cache(b"a")), **({"root": tmp_path} if filesystem else {}))
    overlay = StorageOverlay(backend, writer=backend, content_cache=cache, cache_adapter=AssetCacheAdapter())
    await overlay.initialize()
    a, b, absent = (AssetKey("test", name) for name in ("a", "b", "absent"))
    await overlay.put(a, b"a")
    await overlay.put(b, b"b")
    try:
        assert (await overlay.preload((a,))).loaded == 1
        cache.reads = 0
        warm = await overlay.preload((a,))
        assert (warm.requested, warm.loaded, warm.already_cached, warm.failed) == (1, 0, 1, 0)
        assert cache.reads == 0
        assert (await overlay.preload((b,))).loaded == 1
        evicted = await overlay.preload((a, a, absent))
        assert (evicted.requested, evicted.loaded, evicted.already_cached, evicted.failed) == (2, 1, 0, 1)
        cached = await cache.get(overlay.cache_adapter.cache_key(a, await backend.stat(a)))
        assert overlay.cache_adapter.from_cache(cached) == b"a"
        await overlay.put(a, b"z")
        changed = await overlay.preload((a,))
        assert changed.loaded == 1 and changed.already_cached == 0
        assert await overlay.get(a) == b"z"
        await overlay.delete(a)
        deleted = await overlay.preload((a, absent))
        assert deleted.failed == deleted.requested == 2
    finally:
        await overlay.close()


@pytest.mark.asyncio
async def test_preload_and_online_batches_share_origins_and_close_ownership() -> None:
    backend, overlay, store, (a, b, c) = await _fixture()
    preload = asyncio.create_task(overlay.preload((a, b)))
    await backend.started.get()
    online = asyncio.create_task(collect_values(store.get_many((b, c))))
    await backend.started.get()
    joined = asyncio.create_task(overlay.preload((b,)))
    for _ in range(3):
        await overlay.entered.get()
    preload.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preload
    closing = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    assert not closing.done() and not backend.closed
    backend.release.set()
    try:
        assert await online == (b"b", b"c")
        result = await joined
        assert (result.loaded, result.already_cached, result.failed) == (1, 0, 0)
        assert Counter(key for batch in backend.loads for key in batch) == Counter((a, b, c))
        assert any(len(batch) > 1 for batch in backend.loads)
        await closing
        assert backend.closed
    finally:
        backend.release.set()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("capacity", "missing", "no-cache"))
async def test_preload_reports_uncached_values_as_failed(failure: str) -> None:
    class MissingBackend(_PausedBackend):
        async def get_many(self, keys):
            await collect_values(super().get_many(keys))
            for _ in keys:
                yield None
    backend = MissingBackend() if failure == "missing" else _PausedBackend()
    backend.release.set()
    cache = None if failure == "no-cache" else InMemoryContentCache(max_bytes=0 if failure == "capacity" else 1024)
    overlay = StorageOverlay(backend, writer=backend, content_cache=cache,
                             cache_adapter=AssetCacheAdapter() if cache is not None else None)
    await overlay.initialize()
    key = AssetKey("test", "uncached")
    await overlay.put(key, b"value")
    try:
        result = await overlay.preload((key,))
        assert (result.requested, result.loaded, result.already_cached, result.failed) == (1, 0, 0, 1)
    finally:
        await overlay.close()


@pytest.mark.asyncio
async def test_preload_retains_bounded_bulk_loading() -> None:
    backend = _PausedBackend()
    backend.release.set()
    overlay = StorageOverlay(backend, writer=backend, content_cache=InMemoryContentCache(max_bytes=4096),
                             cache_adapter=AssetCacheAdapter(), preload_batch_size=2)
    await overlay.initialize()
    keys = tuple(AssetKey("test", str(index)) for index in range(5))
    for key in keys:
        await overlay.put(key, b"value")
    try:
        result = await overlay.preload(keys)
        assert result.loaded == len(keys) and result.failed == 0
        assert max(len(batch) for batch in backend.loads) <= 2
        assert len(backend.loads) < len(keys)
    finally:
        await overlay.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_origin_or_close_early() -> None:
    backend, overlay, store, (key, _, _) = await _fixture()
    first = asyncio.create_task(store.get(key))
    await backend.started.get()
    second = asyncio.create_task(collect_values(store.get_many((key,))))
    await overlay.entered.get()
    await overlay.entered.get()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    closing = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    assert not backend.closed
    backend.release.set()
    assert await second == (b"a",)
    await closing
    assert backend.closed


@pytest.mark.asyncio
async def test_shared_origin_failure_reaches_waiters_and_next_read_retries() -> None:
    backend, overlay, store, (key, _, _) = await _fixture()
    failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)
    backend.failure = failure
    reads = [asyncio.create_task(store.get(key)) for _ in range(8)]
    for _ in reads:
        await overlay.entered.get()
    await backend.started.get()
    backend.release.set()
    try:
        results = await asyncio.gather(*reads, return_exceptions=True)
        assert all(error is failure for error in results)
        backend.failure = None
        assert await store.get(key) == b"a"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_inflight_old_revision_does_not_supply_new_revision() -> None:
    backend, overlay, store, (key, _, _) = await _fixture()
    old = asyncio.create_task(store.get(key))
    await backend.started.get()
    await store.put(key, b"new")
    new = asyncio.create_task(store.get(key))
    await backend.started.get()
    backend.release.set()
    try:
        assert await old == b"a"
        assert await new == b"new"
        assert await store.get(key) == b"new"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sql_cold_evicted_and_warm_reads_have_bounded_origin_io(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="cache-miss")
    cache, adapter = InMemoryContentCache(max_bytes=1024 * 1024), AssetCacheAdapter()
    store = AssetStore(StorageOverlay(backend, writer=backend, content_cache=cache, cache_adapter=adapter))
    await store.initialize()
    key, content = AssetKey("test", "payload"), b"x" * 16384
    info = await store.put(key, content)
    await store.capture_metadata()
    content_reads = []
    def capture(conn, cursor, sql, params, context, many):
        if sql.startswith("SELECT") and "FROM ai_asset_entries" in sql:
            content_reads.append(sql)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        for _ in range(2):
            await cache.delete(adapter.cache_key(key, info))
            content_reads.clear()
            assert await asyncio.gather(*(store.get(key) for _ in range(16))) == [content] * 16
            assert 0 < len(content_reads) <= 4, "concurrent cache misses must not read the origin per waiter"
            content_reads.clear()
            assert await asyncio.gather(*(collect_values(store.get_many((key,))) for _ in range(16))) == [(content,)] * 16
            assert not content_reads
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_one_invalid_origin_does_not_poison_other_inflight_keys() -> None:
    backend, overlay, store, (bad, good, _) = await _fixture()
    class RejectOne:
        def validate_value(self, key, value, info):
            if key == bad:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    overlay.validator = RejectOne()
    batch = asyncio.create_task(collect_values(store.get_many((bad, good))))
    await backend.started.get()
    scalar = asyncio.create_task(store.get(good))
    await overlay.entered.get()
    await overlay.entered.get()
    backend.release.set()
    try:
        with pytest.raises(AIError) as failed:
            await batch
        assert failed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await scalar == b"b"
        assert await store.get(good) == b"b"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_shared_origin_does_not_share_mutable_decoded_values() -> None:
    class MutableBackend(_PausedBackend):
        async def get_many(self, keys):
            async for value in super().get_many(keys):
                yield None if value is None else list(value)
    class MutableAdapter(AssetCacheAdapter):
        def from_cache(self, value):
            return list(super().from_cache(value))
        def to_cache(self, value):
            return super().to_cache(bytes(value))
    backend = MutableBackend()
    key = AssetKey("test", "mutable")
    await backend.put(key, b"ab")
    overlay = _ObservedOverlay(backend, content_cache=InMemoryContentCache(max_bytes=1024),
                               cache_adapter=MutableAdapter())
    first = asyncio.create_task(overlay.get(key))
    await backend.started.get()
    second = asyncio.create_task(overlay.get(key))
    await overlay.entered.get()
    await overlay.entered.get()
    backend.release.set()
    try:
        left, right = await asyncio.gather(first, second)
        left.append(99)
        assert right == [97, 98]
        assert await overlay.get(key) == [97, 98]
    finally:
        await overlay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_waiter", (False, True))
@pytest.mark.parametrize("preload", (False, True))
async def test_cancelled_waiter_does_not_leave_unobserved_origin_error(active_waiter: bool, preload: bool) -> None:
    backend, overlay, store, (key, _, _) = await _fixture()
    failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)
    backend.failure = failure
    loop = asyncio.get_running_loop()
    errors = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    reader = asyncio.create_task(overlay.preload((key,)) if preload else store.get(key))
    try:
        await backend.started.get()
        remaining = asyncio.create_task(store.get(key)) if active_waiter else None
        if remaining is not None:
            await overlay.entered.get()
            await overlay.entered.get()
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        backend.release.set()
        if remaining is not None:
            with pytest.raises(AIError) as caught:
                await remaining
            assert caught.value is failure
        # Wait without observing the per-key Future's exception, before close.
        await asyncio.wait(tuple(overlay._cache_loaders))
        await asyncio.sleep(0)
        gc.collect()
        assert not errors
        await store.close()
    finally:
        backend.release.set()
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
async def test_layer_origin_failure_does_not_poison_or_abandon_other_layer() -> None:
    from linktools.ai.storage import StorageLayer
    bad, good = _PausedBackend(), _PausedBackend()
    a, b = AssetKey("test", "bad-layer"), AssetKey("test", "good-layer")
    await bad.put(a, b"bad")
    await good.put(b, b"good")
    failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)
    bad.failure = failure
    overlay = _ObservedOverlay(bad, layers=(StorageLayer("good", good),),
                               content_cache=InMemoryContentCache(max_bytes=1024), cache_adapter=AssetCacheAdapter())
    store = AssetStore(overlay)
    await store.initialize()
    batch = asyncio.create_task(collect_values(store.get_many((a, b))))
    await bad.started.get()
    await good.started.get()
    scalar = asyncio.create_task(store.get(b))
    await overlay.entered.get()
    await overlay.entered.get()
    bad.release.set()
    with pytest.raises(AIError) as caught:
        await batch
    assert caught.value is failure
    assert not scalar.done()
    closing = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    assert not good.closed
    good.release.set()
    results = await asyncio.gather(batch, scalar, closing, return_exceptions=True)
    assert results[0] is failure
    assert results[1] == b"good"
    assert results[2] is None or results[2] is failure
    await store.close()
    assert good.closed and bad.closed


@pytest.mark.asyncio
async def test_completed_layer_read_is_not_held_by_unrelated_inflight_layer() -> None:
    from linktools.ai.storage import StorageLayer
    slow, fast = _PausedBackend(), _PausedBackend()
    a, b = AssetKey("test", "slow"), AssetKey("test", "fast")
    await slow.put(a, b"slow")
    await fast.put(b, b"fast")
    overlay = _ObservedOverlay(slow, layers=(StorageLayer("fast", fast),),
                               content_cache=InMemoryContentCache(max_bytes=1024), cache_adapter=AssetCacheAdapter())
    store = AssetStore(overlay)
    await store.initialize()
    batch = asyncio.create_task(collect_values(store.get_many((a, b))))
    await slow.started.get()
    await fast.started.get()
    scalar = asyncio.create_task(store.get(b))
    await overlay.entered.get()
    await overlay.entered.get()
    fast.release.set()
    try:
        assert await asyncio.wait_for(scalar, 5) == b"fast"
        assert not batch.done()
    finally:
        slow.release.set()
        assert await batch == (b"slow", b"fast")
        await store.close()


@pytest.mark.asyncio
async def test_warm_key_does_not_wait_for_same_layer_cold_origin() -> None:
    backend, overlay, store, (cold, warm, _) = await _fixture()
    info = await backend.stat(warm)
    await overlay.content_cache.put(overlay.cache_adapter.cache_key(warm, info),
                            overlay.cache_adapter.to_cache(b"b"))
    batch = asyncio.create_task(collect_values(store.get_many((cold, warm))))
    await backend.started.get()
    scalar = asyncio.create_task(store.get(warm))
    try:
        assert await asyncio.wait_for(scalar, 5) == b"b"
        assert not batch.done()
    finally:
        backend.release.set()
        assert await batch == (b"a", b"b")
        await store.close()


@pytest.mark.asyncio
async def test_nonbatch_origin_publishes_finished_key_before_slow_peer() -> None:
    backend = InMemoryAssetBackend()
    slow, fast = AssetKey("test", "slow"), AssetKey("test", "fast")
    await backend.put(slow, b"slow")
    await backend.put(fast, b"fast")
    entered, release = asyncio.Event(), asyncio.Event()
    class SingleReader:
        async def head_revision(self):
            return await backend.head_revision()
        async def load_metadata(self, after_revision):
            return await backend.load_metadata(after_revision)
        async def get(self, key):
            if key == slow:
                entered.set()
                await release.wait()
            return await backend.get(key)
    overlay = _ObservedOverlay(SingleReader(), content_cache=InMemoryContentCache(max_bytes=1024),
                               cache_adapter=AssetCacheAdapter())
    batch = asyncio.create_task(collect_values(overlay.get_many((slow, fast))))
    await entered.wait()
    scalar = asyncio.create_task(overlay.get(fast))
    try:
        assert await asyncio.wait_for(scalar, 5) == b"fast"
        assert not batch.done()
    finally:
        release.set()
        assert await batch == (b"slow", b"fast")
        await overlay.close()
