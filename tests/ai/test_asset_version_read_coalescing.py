#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-version read sharing retains historical authority and ownership."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.asset import AssetKey, AssetStore, AssetVersionRef, InMemoryAssetBackend
from linktools.ai.capability import CapabilityLoadContext
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StorageEntryRevision, StorageOverlay


class PausedHistoryBackend(InMemoryAssetBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.active = 0
        self.started: asyncio.Queue[AssetKey] = asyncio.Queue()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.cleanup_release.set()
        self.swallow_cancel = False
        self.missing = False
        self.failure: Exception | None = None
        self.closed = False
        self.closed_while_reading = False

    async def get_at_revision(self, key: AssetKey, revision: StorageEntryRevision) -> bytes | None:
        self.calls += 1
        self.active += 1
        value = None if self.missing else await super().get_at_revision(key, revision)
        self.started.put_nowait(key)
        try:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.cleanup_release.wait()
                if not self.swallow_cancel:
                    raise
            if self.failure is not None:
                raise self.failure
            return value
        finally:
            self.active -= 1

    async def close(self) -> None:
        self.closed_while_reading = self.active != 0
        self.closed = True
        await super().close()


async def make_store() -> tuple[PausedHistoryBackend, AssetStore, tuple[AssetVersionRef, ...]]:
    backend = PausedHistoryBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    keys = (AssetKey("text", "a"), AssetKey("text", "b"))
    for key in keys:
        await store.put(key, key.id.encode())
    return backend, store, await store.resolve_versions(keys)


@pytest.mark.asyncio
async def test_same_turn_version_reads_share_one_fresh_source_observation() -> None:
    backend, store, (ref, _) = await make_store()
    first = asyncio.create_task(store.read_versions((ref,)))
    second = asyncio.create_task(store.read_versions((ref,)))
    await backend.started.get()
    backend.release.set()
    try:
        assert await asyncio.gather(first, second) == [(b"a",), (b"a",)]
        assert backend.calls == 1
        assert await store.read_versions((ref,)) == (b"a",)
        assert backend.calls == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_late_version_reader_rechecks_history_after_source_read_started() -> None:
    backend, store, (ref, _) = await make_store()
    first = asyncio.create_task(store.read_versions((ref,)))
    await backend.started.get()
    backend.missing = True
    late = asyncio.create_task(store.read_versions((ref,)))
    await asyncio.wait_for(backend.started.get(), timeout=2)
    backend.release.set()
    try:
        assert await first == (b"a",)
        with pytest.raises(AIError) as raised:
            await late
        assert raised.value.code is ErrorCode.ASSET_VERSION_NOT_FOUND
        assert backend.calls == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_shared_version_reads_keep_each_reference_integrity_fields() -> None:
    backend, store, (ref, _) = await make_store()
    valid = asyncio.create_task(store.read_versions((ref,)))
    invalid = asyncio.create_task(store.read_versions((replace(ref, size=2),)))
    await backend.started.get()
    await backend.started.get()
    backend.release.set()
    try:
        assert await valid == (b"a",)
        with pytest.raises(AIError) as raised:
            await invalid
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert backend.calls == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cancelling_one_version_reader_keeps_another_reader_alive() -> None:
    backend, store, (ref, _) = await make_store()
    first = asyncio.create_task(store.read_versions((ref,)))
    second = asyncio.create_task(store.read_versions((ref,)))
    await backend.started.get()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not backend.cancelled.is_set()
    backend.release.set()
    try:
        assert await second == (b"a",)
        assert backend.calls == 1
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("swallow_cancel", (False, True))
async def test_last_reader_and_cancelled_close_wait_for_source_cleanup(swallow_cancel: bool) -> None:
    backend, store, (ref, _) = await make_store()
    backend.swallow_cancel = swallow_cancel
    backend.cleanup_release.clear()
    reader = asyncio.create_task(store.read_versions((ref,)))
    await backend.started.get()
    reader.cancel()
    await backend.cancelled.wait()
    closing = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    reader.cancel()
    await asyncio.sleep(0)
    try:
        assert not reader.done()
        assert not closing.done()
        assert not backend.closed
    finally:
        backend.cleanup_release.set()
        backend.release.set()
    with pytest.raises(asyncio.CancelledError):
        await reader
    with pytest.raises(asyncio.CancelledError):
        await closing
    await store.close()
    assert backend.closed and not backend.closed_while_reading


@pytest.mark.asyncio
async def test_close_drains_every_reference_of_an_admitted_read() -> None:
    backend, store, refs = await make_store()
    reader = asyncio.create_task(store.read_versions(refs))
    await backend.started.get()
    closing = asyncio.create_task(store.close())
    await asyncio.sleep(0)
    try:
        assert not closing.done()
        assert not backend.closed
        with pytest.raises(AIError):
            await store.read_versions((refs[0],))
    finally:
        backend.release.set()
    assert await reader == (b"a", b"b")
    await closing
    assert backend.calls == 2
    assert backend.closed and not backend.closed_while_reading


@pytest.mark.asyncio
async def test_capture_memo_coalesces_late_first_reads_without_cross_capture_reuse() -> None:
    backend, store, (ref, _) = await make_store()
    first_capture = await CapabilityLoadContext.capture("first", store)
    second_capture = await CapabilityLoadContext.capture("second", store)
    first = asyncio.create_task(first_capture.read(ref.key))
    await backend.started.get()
    late_same = asyncio.create_task(first_capture.read_many((ref.key, ref.key)))
    late_other = asyncio.create_task(second_capture.read(ref.key))
    await backend.started.get()
    backend.release.set()
    try:
        assert await first == b"a"
        assert await late_same == (b"a", b"a")
        assert await late_other == b"a"
        assert backend.calls == 2
        assert await first_capture.read(ref.key) == b"a"
        assert await first_capture.read(ref.key) == b"a"
        assert backend.calls == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_disjoint_capture_keys_can_load_concurrently() -> None:
    backend, store, (one, two) = await make_store()
    capture = await CapabilityLoadContext.capture("capture", store)
    first = asyncio.create_task(capture.read(one.key))
    await backend.started.get()
    second = asyncio.create_task(capture.read(two.key))
    assert await asyncio.wait_for(backend.started.get(), timeout=2) == two.key
    backend.release.set()
    try:
        assert await asyncio.gather(first, second) == [b"a", b"b"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_capture_fill_is_retried_and_not_memoized() -> None:
    backend, store, (ref, _) = await make_store()
    capture = await CapabilityLoadContext.capture("capture", store)
    failure = RuntimeError("history unavailable")
    backend.failure = failure
    backend.release.set()
    try:
        with pytest.raises(RuntimeError) as raised:
            await capture.read(ref.key)
        assert raised.value is failure
        backend.failure = None
        assert await capture.read(ref.key) == b"a"
        assert backend.calls == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_shared_version_read_is_observed_and_retried() -> None:
    backend, store, (ref, _) = await make_store()
    failure = RuntimeError("history unavailable")
    backend.failure = failure
    first = asyncio.create_task(store.read_versions((ref,)))
    second = asyncio.create_task(store.read_versions((ref,)))
    await backend.started.get()
    backend.release.set()
    try:
        assert await asyncio.gather(first, second, return_exceptions=True) == [failure, failure]
        assert backend.calls == 1
        backend.failure = None
        assert await store.read_versions((ref,)) == (b"a",)
        assert backend.calls == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_last_reader_preserves_source_cleanup_failure_over_cancellation() -> None:
    backend, store, (ref, _) = await make_store()
    backend.swallow_cancel = True
    backend.cleanup_release.clear()
    failure = RuntimeError("source cleanup failed")
    backend.failure = failure
    reader = asyncio.create_task(store.read_versions((ref,)))
    await backend.started.get()
    reader.cancel()
    await backend.cancelled.wait()
    backend.cleanup_release.set()
    try:
        with pytest.raises(RuntimeError) as raised:
            await reader
        assert raised.value is failure
    finally:
        await store.close()
    assert not backend.closed_while_reading


@pytest.mark.asyncio
async def test_cancelling_queued_capture_reader_does_not_cancel_memo_fill() -> None:
    backend, store, (ref, _) = await make_store()
    capture = await CapabilityLoadContext.capture("capture", store)
    first = asyncio.create_task(capture.read(ref.key))
    await backend.started.get()
    queued = asyncio.create_task(capture.read(ref.key))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert not backend.cancelled.is_set()
    backend.release.set()
    try:
        assert await first == b"a"
        assert await capture.read(ref.key) == b"a"
        assert backend.calls == 1
    finally:
        await store.close()
