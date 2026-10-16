#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset byte binding and accepted storage-read lifecycle regressions."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace

import pytest

from linktools.ai.asset import AssetContentCache, AssetInfo, AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryCacheBackend, MetadataLoad, StorageOverlay, StorageRevision


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", (False, True))
async def test_valid_cache_envelope_cannot_substitute_other_asset_bytes(batch: bool) -> None:
    backend = InMemoryAssetBackend()
    cache = InMemoryCacheBackend(max_bytes=4096)
    adapter = AssetContentCache(cache)
    store = AssetStore(StorageOverlay(backend, writer=backend, content_cache=adapter))
    await store.initialize()
    key = AssetKey("text", "x")
    try:
        info = await store.put(key, b"old")
        await cache.put(adapter.cache_key(key, info), adapter.to_cache(b"BAD"))
        if batch:
            assert await store.get_many((key,)) == (b"old",)
        else:
            assert await store.get(key) == b"old"
    finally:
        await store.close()


class PausedCache(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=4096)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def get(self, key: str) -> bytes | None:
        self.entered.set()
        await self.release.wait()
        return await super().get(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("get", "get_many", "preload"))
async def test_origin_race_cannot_publish_latest_under_old_metadata_key(operation: str) -> None:
    backend = InMemoryAssetBackend()
    cache = PausedCache()
    adapter = AssetContentCache(cache)
    overlay = StorageOverlay(backend, writer=backend, content_cache=adapter)
    store = AssetStore(overlay)
    await store.initialize()
    key = AssetKey("text", "x")
    info = await store.put(key, b"old")
    reader = asyncio.create_task(_read(overlay, operation, key))
    try:
        await asyncio.wait_for(cache.entered.wait(), timeout=2)
        await backend.put(key, b"new")
        cache.release.set()
        with pytest.raises(AIError) as raised:
            await reader
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await cache.get(adapter.cache_key(key, info)) is None
    finally:
        cache.release.set()
        await asyncio.gather(reader, return_exceptions=True)
        await store.close()


class PausedMetadataBackend(InMemoryAssetBackend):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.active = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.trace = []

    async def load_metadata(self, after: StorageRevision | None = None) -> MetadataLoad[AssetKey, AssetInfo]:
        self.trace.append(("metadata-start", self.closed))
        self.active = True
        self.entered.set()
        try:
            await self.release.wait()
            return await super().load_metadata(after)
        finally:
            self.active = False
            self.trace.append(("metadata-end", self.closed))

    async def close(self) -> None:
        self.trace.append(("close", self.active))
        self.closed = True
        await super().close()


@pytest.mark.asyncio
async def test_queued_metadata_refresh_cannot_outlive_backend_close() -> None:
    backend = PausedMetadataBackend()
    adapter = AssetContentCache(InMemoryCacheBackend(max_bytes=1024))
    store = AssetStore(StorageOverlay(backend, writer=backend, content_cache=adapter))
    await store.initialize()
    key = AssetKey("text", "x")
    await store.put(key, b"x")
    reader = asyncio.create_task(store.get(key))
    closing = asyncio.create_task(store.close())
    try:
        # A fix may drain the admitted refresh or reject it before source I/O.
        entered = asyncio.create_task(backend.entered.wait())
        await asyncio.wait((entered, closing), return_when=asyncio.FIRST_COMPLETED)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not (backend.closed and backend.active), backend.trace
        assert ("metadata-start", True) not in backend.trace, backend.trace
    finally:
        backend.release.set()
        entered.cancel()
        await asyncio.gather(entered, reader, closing, return_exceptions=True)
        await store.close()



@pytest.mark.asyncio
async def test_asset_validator_preserves_custom_validation_on_shared_overlay() -> None:
    class Validator:
        def __init__(self) -> None:
            self.values = []
            self.failure = None

        def validate_value(self, key: AssetKey, value: bytes, info: AssetInfo) -> None:
            self.values.append(value)
            if self.failure is not None:
                raise self.failure

    backend = InMemoryAssetBackend()
    validator = Validator()
    overlay = StorageOverlay(backend, writer=backend, validator=validator)
    first = AssetStore(overlay)
    await first.initialize()
    key = AssetKey("text", "shared")
    await backend.put(key, b"value")
    try:
        assert await first.get(key) == b"value"
        second = AssetStore(overlay)
        await second.initialize()
        assert await second.get(key) == b"value"
        assert validator.values == [b"value", b"value"]
        validator.failure = RuntimeError("custom validation failed")
        with pytest.raises(RuntimeError) as failed:
            await second.get(key)
        assert failed.value is validator.failure
    finally:
        await first.close()


@pytest.mark.asyncio
async def test_asset_validator_rejects_wrong_size_with_matching_digest() -> None:
    class Backend(InMemoryAssetBackend):
        async def load_metadata(self, after: StorageRevision | None = None) -> MetadataLoad[AssetKey, AssetInfo]:
            load = await super().load_metadata(after)
            return replace(load, changes=tuple(
                replace(change, info=replace(change.info, size=change.info.size + 1))
                if change.info is not None else change
                for change in load.changes
            ))

    backend = Backend()
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    key = AssetKey("text", "size")
    await backend.put(key, b"value")
    try:
        with pytest.raises(AIError) as failed:
            await store.get(key)
        assert failed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await store.close()


async def _read(
    overlay: StorageOverlay[AssetKey, bytes, AssetInfo], operation: str, key: AssetKey,
) -> object:
    if operation == "refresh":
        return await overlay.refresh()
    if operation == "current_revision":
        return await overlay.current_revision()
    if operation == "get":
        return await overlay.get(key)
    if operation == "stat":
        return await overlay.stat(key)
    if operation == "locate":
        return await overlay.locate(key)
    if operation == "locate_many":
        return await overlay.locate_many((key,))
    if operation == "get_many":
        return await overlay.get_many((key,))
    if operation == "list_info":
        return await overlay.list_info()
    if operation == "list_info_with_locations":
        return await overlay.list_info_with_locations()
    if operation == "write_states":
        return await overlay.write_states((key,))
    if operation == "preload":
        return await overlay.preload((key,))
    raise AssertionError(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", (
    "refresh", "current_revision", "get", "stat", "locate", "locate_many",
    "get_many", "list_info", "list_info_with_locations", "write_states", "preload",
))
async def test_close_drains_accepted_read_entry_and_rejects_later_read(operation: str) -> None:
    class Backend(PausedMetadataBackend):
        async def head_revision(self) -> StorageRevision:
            self.entered.set()
            await self.release.wait()
            assert not self.closed
            return await super().head_revision()

    backend = Backend()
    overlay = StorageOverlay(backend, writer=backend)
    await overlay.initialize()
    key = AssetKey("text", "entry")
    await backend.put(key, b"value")
    reader = asyncio.create_task(_read(overlay, operation, key))
    closing = asyncio.create_task(overlay.close())
    try:
        await asyncio.wait_for(backend.entered.wait(), 2)
        assert not closing.done()
        assert not backend.closed
        with pytest.raises(AIError) as failed:
            await _read(overlay, operation, key)
        assert failed.value.code is ErrorCode.STORAGE_CLOSED
        backend.release.set()
        await reader
        await closing
        assert ("metadata-start", True) not in backend.trace
        assert ("close", True) not in backend.trace
    finally:
        backend.release.set()
        await asyncio.gather(reader, closing, return_exceptions=True)
        await overlay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("value", "cancel", "error"))
async def test_close_waits_for_uncached_origin_and_its_cancellation_cleanup(outcome: str) -> None:
    class Backend(InMemoryAssetBackend):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False
            self.closes = 0

        async def get(self, key: AssetKey) -> bytes | None:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
                assert not self.closed
                raise
            assert not self.closed
            if outcome == "error":
                raise RuntimeError("origin failed")
            return await super().get(key)

        async def close(self) -> None:
            self.closes += 1
            self.closed = True
            await super().close()

    backend = Backend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    key = AssetKey("text", "uncached")
    await store.put(key, b"value")
    reader = asyncio.create_task(store.get(key))
    await backend.entered.wait()
    closing = asyncio.create_task(store.close())
    repeated = asyncio.create_task(store.close())
    try:
        if outcome == "cancel":
            reader.cancel()
            await backend.cancelled.wait()
        await asyncio.sleep(0)
        assert not closing.done()
        assert not repeated.done()
        assert not backend.closed
        backend.release.set()
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await reader
        elif outcome == "error":
            with pytest.raises(RuntimeError, match="origin failed"):
                await reader
        else:
            assert await reader == b"value"
        await asyncio.gather(closing, repeated)
        assert backend.closes == 1
    finally:
        backend.release.set()
        await asyncio.gather(reader, closing, repeated, return_exceptions=True)
        await store.close()


@pytest.mark.asyncio
async def test_repeated_close_shares_backend_cleanup_after_waiter_cancellation() -> None:
    class Backend(InMemoryAssetBackend):
        def __init__(self) -> None:
            super().__init__()
            self.closing = asyncio.Event()
            self.release = asyncio.Event()
            self.closes = 0

        async def close(self) -> None:
            self.closes += 1
            self.closing.set()
            await self.release.wait()
            await super().close()

    backend = Backend()
    overlay = StorageOverlay(backend)
    await overlay.initialize()
    first = asyncio.create_task(overlay.close())
    await backend.closing.wait()
    first.cancel()
    second = asyncio.create_task(overlay.close())
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        assert backend.closes == 1
        backend.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await overlay.close()
        assert backend.closes == 1
    finally:
        backend.release.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_reader_leaves_cache_loader_owned_until_close() -> None:
    class Backend(InMemoryAssetBackend):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def get_many(self, keys: Sequence[AssetKey]) -> dict[AssetKey, bytes]:
            assert not self.closed
            return await super().get_many(keys)

        async def close(self) -> None:
            self.closed = True
            await super().close()

    class Cache(PausedCache):
        async def close(self) -> None:
            raise AssertionError("borrowed cache must not be closed")

    backend = Backend()
    cache = Cache()
    store = AssetStore(StorageOverlay(
        backend, writer=backend, content_cache=AssetContentCache(cache),
    ))
    await store.initialize()
    key = AssetKey("text", "cancelled")
    await store.put(key, b"value")
    reader = asyncio.create_task(store.get(key))
    await cache.entered.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    closing = asyncio.create_task(store.close())
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done()
        assert not backend.closed
        cache.release.set()
        await closing
        assert backend.closed
    finally:
        cache.release.set()
        await closing


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("current_revision", "refresh"))
async def test_metadata_failure_keeps_unfinished_sibling_owned_by_close(operation: str) -> None:
    from linktools.ai.storage import StorageLayer

    entered = asyncio.Event()
    release = asyncio.Event()
    failure = RuntimeError("metadata failed")

    class Slow(InMemoryAssetBackend):
        active = False
        closed = False

        async def head_revision(self) -> StorageRevision:
            self.active = True
            entered.set()
            try:
                await release.wait()
                assert not self.closed
                return await super().head_revision()
            finally:
                self.active = False

        async def load_metadata(self, after: StorageRevision | None = None) -> MetadataLoad[AssetKey, AssetInfo]:
            await self.head_revision()
            return await super().load_metadata(after)

        async def close(self) -> None:
            assert not self.active
            self.closed = True
            await super().close()

    class Failing(InMemoryAssetBackend):
        async def head_revision(self) -> StorageRevision:
            await entered.wait()
            raise failure

        async def load_metadata(self, after: StorageRevision | None = None) -> MetadataLoad[AssetKey, AssetInfo]:
            await self.head_revision()
            raise AssertionError("unreachable")

    slow = Slow()
    overlay = StorageOverlay(Failing(), layers=(StorageLayer("slow", slow),))
    await overlay.initialize()
    reader = asyncio.create_task(_read(overlay, operation, AssetKey("text", "sibling")))
    closing = None
    try:
        with pytest.raises(RuntimeError) as failed:
            await asyncio.wait_for(reader, 2)
        assert failed.value is failure
        assert slow.active
        closing = asyncio.create_task(overlay.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done()
        assert not slow.closed
        release.set()
        await closing
    finally:
        release.set()
        await asyncio.gather(reader, return_exceptions=True)
        if closing is not None:
            await closing
        await overlay.close()


@pytest.mark.asyncio
async def test_cancelled_head_group_settles_request_owned_cleanup_before_close() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    class Backend(InMemoryAssetBackend):
        closed = False

        async def head_revision(self) -> StorageRevision:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                assert not self.closed
                raise
            return await super().head_revision()

        async def close(self) -> None:
            self.closed = True
            await super().close()

    backend = Backend()
    overlay = StorageOverlay(backend)
    await overlay.initialize()
    reader = asyncio.create_task(overlay.current_revision())
    await entered.wait()
    reader.cancel()
    await cancelled.wait()
    closing = asyncio.create_task(overlay.close())
    try:
        await asyncio.sleep(0)
        assert not reader.done()
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await reader
        await closing
    finally:
        release.set()
        await asyncio.gather(reader, closing, return_exceptions=True)
