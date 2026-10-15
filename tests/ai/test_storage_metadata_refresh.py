#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared metadata refresh retains work and failure ownership after cancellation."""

import asyncio
import gc

import pytest

from linktools.ai.asset import AssetInfo, AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.storage import (
    LayerMetadataView,
    LayerRefreshPolicy,
    MetadataLoad,
    StorageOverlay,
    StorageRevision,
)


class _PausedMetadataBackend(InMemoryAssetBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.failure: Exception | None = None
        self.loads = 0
        self.closed = False

    async def load_metadata(
        self, after_revision: StorageRevision | None = None,
    ) -> MetadataLoad[AssetKey, AssetInfo]:
        self.loads += 1
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        return await super().load_metadata(after_revision)

    async def close(self) -> None:
        self.closed = True
        await super().close()


@pytest.mark.asyncio
async def test_cold_metadata_readers_share_one_source_observation() -> None:
    class Backend(InMemoryAssetBackend):
        def __init__(self) -> None:
            super().__init__()
            self.head_reads = 0
            self.metadata_reads = 0

        async def head_revision(self) -> StorageRevision | None:
            self.head_reads += 1
            return await super().head_revision()

        async def load_metadata(
            self, after_revision: StorageRevision | None = None,
        ) -> MetadataLoad[AssetKey, AssetInfo]:
            self.metadata_reads += 1
            return await super().load_metadata(after_revision)

    backend = Backend()
    key = AssetKey("test", "concurrent-read")
    written = await backend.put(key, b"value")
    view = LayerMetadataView(backend, LayerRefreshPolicy.REVISIONED)
    try:
        values = await asyncio.gather(*(view.refresh() for _ in range(16)))
        assert all(value.entries[key] == written.info for value in values)
        assert backend.metadata_reads == 1
        assert backend.head_reads == 0
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_waiter", (False, True))
@pytest.mark.parametrize("fails", (False, True))
async def test_cancelled_metadata_waiter_retains_shared_result_and_error_ownership(
    active_waiter: bool, fails: bool,
) -> None:
    backend = _PausedMetadataBackend()
    failure = RuntimeError("metadata load failed")
    backend.failure = failure if fails else None
    view = LayerMetadataView(backend, LayerRefreshPolicy.ALWAYS)
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    waiter = asyncio.create_task(view.refresh())
    other = None
    try:
        await backend.started.wait()
        if active_waiter:
            other = asyncio.create_task(view.refresh())
            await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        del waiter
        backend.release.set()
        if other is not None:
            if fails:
                with pytest.raises(RuntimeError) as caught:
                    await other
                assert caught.value is failure
            else:
                assert (await other).revision == await backend.head_revision()
        for _ in range(10):
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        assert errors == []
        assert backend.loads <= 1
        backend.failure = None
        assert (await view.refresh()).revision == await backend.head_revision()
    finally:
        backend.release.set()
        loop.set_exception_handler(previous)
        await backend.close()


@pytest.mark.asyncio
async def test_cancelled_overlay_close_finishes_refresh_before_closing_backend() -> None:
    backend = _PausedMetadataBackend()
    failure = RuntimeError("metadata load failed")
    backend.failure = failure
    overlay = StorageOverlay(backend)
    await overlay.initialize()
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    waiter = asyncio.create_task(overlay.primary_view.refresh())
    closing = None
    try:
        await backend.started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        closing = asyncio.create_task(overlay.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        assert not backend.closed
        backend.release.set()
        with pytest.raises(RuntimeError) as caught:
            await closing
        assert caught.value is failure
        assert backend.closed
        await asyncio.sleep(0)
        assert errors == []
        await overlay.close()
    finally:
        backend.release.set()
        await asyncio.gather(waiter, *(() if closing is None else (closing,)), return_exceptions=True)
        loop.set_exception_handler(previous)
        await overlay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ("head", "source_head", "backend_correction", "source_correction"))
async def test_metadata_write_invalidates_an_inflight_revision_check(pause_at: str) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    pause = False

    async def wait_at(stage: str) -> None:
        if pause and pause_at == stage:
            started.set()
            await release.wait()

    class Backend(InMemoryAssetBackend):
        async def head_revision(self) -> StorageRevision | None:
            value = await super().head_revision()
            await wait_at("head" if pause_at == "head" else "backend_correction")
            return value

    class Source:
        async def head_revision(self) -> StorageRevision | None:
            value = (await InMemoryAssetBackend.head_revision(backend)
                     if pause_at == "source_head" else StorageRevision("source-needs-correction"))
            await wait_at("source_head")
            return value

        async def revision_bumped(self, revision: StorageRevision) -> None:
            await wait_at("source_correction")

    backend = Backend()
    view = LayerMetadataView(backend, LayerRefreshPolicy.REVISIONED,
                             revision_source=None if pause_at == "head" else Source())
    key = AssetKey("test", "write-during-head")
    await backend.put(key, b"old")
    await view.refresh()
    pause = True
    reader = asyncio.create_task(view.refresh())
    try:
        await started.wait()
        written = await backend.put(key, b"new")
        view.apply_write(key, written.info, written.store_revision)
        release.set()
        result = await reader
        assert result.revision == written.store_revision
        assert result.entries[key] == written.info
    finally:
        release.set()
        await asyncio.gather(reader, return_exceptions=True)
        await backend.close()


@pytest.mark.asyncio
async def test_revision_check_started_after_external_write_observes_it() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    pause = False

    class Backend(InMemoryAssetBackend):
        async def head_revision(self) -> StorageRevision | None:
            value = await super().head_revision()
            if pause:
                started.set()
                await release.wait()
            return value

    backend = Backend()
    view = LayerMetadataView(backend, LayerRefreshPolicy.REVISIONED)
    key = AssetKey("test", "external-write")
    await backend.put(key, b"old")
    await view.refresh()
    pause = True
    first = asyncio.create_task(view.refresh())
    second = None
    try:
        await started.wait()
        written = await backend.put(key, b"new")
        second = asyncio.create_task(view.refresh())
        await asyncio.sleep(0)
        release.set()
        await first
        result = await second
        assert result.revision == written.store_revision
        assert result.entries[key] == written.info
    finally:
        release.set()
        await asyncio.gather(first, *(() if second is None else (second,)), return_exceptions=True)
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", (False, True))
async def test_shared_metadata_snapshot_observes_writes_before_new_reader(warm: bool) -> None:
    captured, release, reader_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pause = False

    class Backend(InMemoryAssetBackend):
        async def load_metadata(
            self, after_revision: StorageRevision | None = None,
        ) -> MetadataLoad[AssetKey, AssetInfo]:
            result = await super().load_metadata(after_revision)
            if pause:
                captured.set()
                await release.wait()
            return result

    backend = Backend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    existing, new = AssetKey("test", "existing"), AssetKey("test", "new")
    await store.initialize()
    first = second = None
    try:
        await backend.put(existing, b"old")
        if warm:
            await store.list_info()
            await backend.put(existing, b"changed")
        pause = True
        first = asyncio.create_task(store.list_info())
        await captured.wait()
        await backend.put(new, b"committed before read")

        async def read_new() -> bytes | None:
            reader_started.set()
            return await store.get(new)

        second = asyncio.create_task(read_new())
        await reader_started.wait()
        release.set()
        await first
        assert await second == b"committed before read"
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        await store.close()
