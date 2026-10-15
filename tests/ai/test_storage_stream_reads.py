#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming storage preserves ordered prefixes and reader ownership."""

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from linktools.ai.asset import AssetCacheAdapter, AssetKey, InMemoryAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryContentCache, StorageOverlay
from tests.ai._storage_stream_helpers import collect_values

pytestmark = pytest.mark.asyncio


class _StreamingBackend(InMemoryAssetBackend):
    def __init__(self) -> None:
        super().__init__()
        self.reads: list[AssetKey] = []
        self.groups: list[tuple[AssetKey, ...]] = []
        self.stream_closed = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_second = False
        self.pause_second = False

    async def get_many(self, keys: Sequence[AssetKey]) -> AsyncIterator[bytes | None]:
        self.groups.append(tuple(keys))
        try:
            for index, key in enumerate(keys):
                self.reads.append(key)
                if index == 1:
                    self.second_started.set()
                    if self.pause_second:
                        await self.release.wait()
                    if self.fail_second:
                        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
                yield await self.get(key)
        finally:
            self.stream_closed.set()


async def _setup(cached: bool = False):
    backend = _StreamingBackend()
    first, second = AssetKey("test", "first"), AssetKey("test", "second")
    await backend.put(first, b"first")
    await backend.put(second, b"second")
    overlay = StorageOverlay(
        backend,
        content_cache=InMemoryContentCache(max_bytes=4096) if cached else None,
        cache_adapter=AssetCacheAdapter() if cached else None,
    )
    await overlay.initialize()
    return backend, overlay, first, second


async def test_uncached_stream_closes_before_reading_unrequested_suffix() -> None:
    backend, overlay, first, second = await _setup()
    stream = overlay.get_many((first, second))
    try:
        assert await anext(stream) == b"first"
        assert backend.reads == [first]
        assert backend.groups == [(first, second)]
        await stream.aclose()
        assert backend.stream_closed.is_set()
        assert backend.reads == [first]
    finally:
        await stream.aclose()
        await overlay.close()


@pytest.mark.parametrize("cached", (False, True))
async def test_stream_exposes_prefix_before_later_origin_failure(cached: bool) -> None:
    backend, overlay, first, second = await _setup(cached)
    backend.fail_second = True
    stream = overlay.get_many((first, second))
    try:
        assert await anext(stream) == b"first"
        with pytest.raises(AIError) as raised:
            await anext(stream)
        assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert backend.stream_closed.is_set()
    finally:
        await stream.aclose()
        await overlay.close()


async def test_cancelled_stream_keeps_other_reader_shared_origin_alive() -> None:
    backend, overlay, first, second = await _setup(True)
    backend.pause_second = True
    stream = overlay.get_many((first, second))
    waiter = None
    other = None
    try:
        assert await anext(stream) == b"first"
        await backend.second_started.wait()
        other = asyncio.create_task(overlay.get(second))
        waiter = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not other.cancelled()
        assert not backend.stream_closed.is_set()
        backend.release.set()
        assert await other == b"second"
        await overlay.close()
        assert backend.stream_closed.is_set()
        assert backend.groups == [(first, second)]
    finally:
        backend.release.set()
        await stream.aclose()
        for task in (waiter, other):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (waiter, other) if task is not None), return_exceptions=True)
        await overlay.close()


async def test_stream_refreshes_only_unpublished_suffix_and_keeps_duplicates() -> None:
    first, second = AssetKey("test", "first"), AssetKey("test", "second")

    class RacingBackend(InMemoryAssetBackend):
        def __init__(self) -> None:
            super().__init__()
            self.groups: list[tuple[AssetKey, ...]] = []
            self.miss_once = True

        async def get_many(self, keys: Sequence[AssetKey]) -> AsyncIterator[bytes | None]:
            self.groups.append(tuple(keys))
            for key in keys:
                if key == second and self.miss_once:
                    self.miss_once = False
                    yield None
                else:
                    yield await self.get(key)

    backend = RacingBackend()
    await backend.put(first, b"old-first")
    await backend.put(second, b"old-second")
    overlay = StorageOverlay(backend)
    await overlay.initialize()
    stream = overlay.get_many((first, second, first))
    try:
        assert await anext(stream) == b"old-first"
        await backend.put(first, b"new-first")
        await backend.put(second, b"new-second")
        assert await collect_values(stream) == (b"new-second", b"old-first")
        assert backend.groups == [(first, second), (second,)]
    finally:
        await stream.aclose()
        await overlay.close()


@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("shape", ("short", "extra"))
async def test_invalid_batch_cardinality_is_not_silently_missing(cached: bool, shape: str) -> None:
    class InvalidBackend(InMemoryAssetBackend):
        async def get_many(self, keys: Sequence[AssetKey]) -> AsyncIterator[bytes | None]:
            yield b"first"
            if shape == "extra":
                yield b"second"
                yield b"unexpected"

    backend = InvalidBackend()
    first, second = AssetKey("test", "first"), AssetKey("test", "second")
    await backend.put(first, b"first")
    await backend.put(second, b"second")
    overlay = StorageOverlay(
        backend,
        content_cache=InMemoryContentCache(max_bytes=4096) if cached else None,
        cache_adapter=AssetCacheAdapter() if cached else None,
    )
    await overlay.initialize()
    stream = overlay.get_many((first, second))
    try:
        assert await anext(stream) == b"first"
        with pytest.raises(AIError) as raised:
            await anext(stream)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await stream.aclose()
        await overlay.close()


async def test_closing_one_layer_failure_still_closes_other_open_streams() -> None:
    from linktools.ai.storage import StorageLayer

    class ClosingBackend(InMemoryAssetBackend):
        def __init__(self, fail_close: bool) -> None:
            super().__init__()
            self.fail_close = fail_close
            self.stream_closed = False

        async def get_many(self, keys: Sequence[AssetKey]) -> AsyncIterator[bytes | None]:
            try:
                for key in keys:
                    yield await self.get(key)
            finally:
                self.stream_closed = True
                if self.fail_close:
                    raise RuntimeError("stream close failed")

    first, second = ClosingBackend(True), ClosingBackend(False)
    keys = tuple(AssetKey("test", str(index)) for index in range(4))
    for backend, selected in ((first, keys[::2]), (second, keys[1::2])):
        for key in selected:
            await backend.put(key, key.id.encode())
    overlay = StorageOverlay(first, layers=(StorageLayer("second", second),))
    await overlay.initialize()
    stream = overlay.get_many(keys)
    try:
        assert await anext(stream) == b"0"
        assert await anext(stream) == b"1"
        with pytest.raises(RuntimeError, match="stream close failed"):
            await stream.aclose()
        assert first.stream_closed and second.stream_closed
    finally:
        await overlay.close()


async def test_cached_stream_close_failure_keeps_other_layer_owned_until_settled() -> None:
    from linktools.ai.storage import StorageLayer

    class FailingCloseBackend(InMemoryAssetBackend):
        def get_many(self, keys: Sequence[AssetKey]) -> AsyncIterator[bytes | None]:
            source = super().get_many(keys)

            class Stream:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    return await anext(source)

                async def aclose(self):
                    await source.aclose()
                    raise RuntimeError("stream close failed")

            return Stream()

    bad, slow = FailingCloseBackend(), _StreamingBackend()
    slow.pause_second = True
    a, b, c = (AssetKey("test", value) for value in ("a", "b", "c"))
    await bad.put(a, b"a")
    await slow.put(b, b"b")
    await slow.put(c, b"c")
    admitted = asyncio.Event()

    class ObservedOverlay(StorageOverlay):
        async def _cache_reads(self, keys, state):
            tasks = await super()._cache_reads(keys, state)
            if tuple(keys) == (c,):
                admitted.set()
            return tasks

    overlay = ObservedOverlay(
        bad, layers=(StorageLayer("slow", slow),),
        content_cache=InMemoryContentCache(max_bytes=4096), cache_adapter=AssetCacheAdapter(),
    )
    await overlay.initialize()
    stream = overlay.get_many((a, b, c))
    closing = None
    other = None
    try:
        with pytest.raises(RuntimeError, match="stream close failed"):
            await anext(stream)
        await slow.second_started.wait()
        other = asyncio.create_task(overlay.get(c))
        await admitted.wait()
        closing = asyncio.create_task(overlay.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert not slow.stream_closed.is_set()
        slow.release.set()
        assert await other == b"c"
        await closing
        assert slow.stream_closed.is_set()
        assert slow.groups == [(b, c)]
    finally:
        slow.release.set()
        await stream.aclose()
        await asyncio.gather(*(task for task in (other, closing) if task is not None), return_exceptions=True)
        await overlay.close()


@pytest.mark.parametrize("facade", ("live", "snapshot", "capture"))
async def test_asset_facades_share_ordered_async_stream_contract(facade: str) -> None:
    from linktools.ai.asset import AssetStore
    from linktools.ai.capability import CapabilityLoadContext
    from linktools.ai.storage import InMemoryObjectStore

    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    restored = None
    first, empty, missing = (AssetKey("test", value) for value in ("first", "empty", "missing"))
    try:
        await store.put(first, b"first")
        await store.put(empty, b"")
        if facade == "snapshot":
            objects = InMemoryObjectStore()
            reference = await store.snapshot((first, empty), object_store=objects)
            restored = AssetStore.from_snapshot(reference, object_store=objects)
            await restored.initialize()
            reader = restored
        elif facade == "capture":
            reader = (await CapabilityLoadContext.capture("stream", store)).asset_reader
        else:
            reader = store
        assert await collect_values(reader.get_many((empty, first, missing, first))) == (
            b"", b"first", None, b"first",
        )
        assert await collect_values(reader.get_many(())) == ()
    finally:
        if restored is not None:
            await restored.close()
        await store.close()
