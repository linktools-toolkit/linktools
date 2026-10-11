#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared metadata refresh retains work and failure ownership after cancellation."""

import asyncio
import gc

import pytest

from linktools.ai.asset import AssetInfo, AssetKey, InMemoryAssetBackend
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
