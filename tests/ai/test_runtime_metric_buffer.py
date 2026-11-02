#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared batch-write backpressure, batching and lifecycle semantics."""

import asyncio
from collections.abc import Awaitable
from datetime import datetime, timezone

import pytest
from linktools.ai.observe import Observation
from linktools.ai.storage import InMemoryCacheBackend, BatchWriteScope
from linktools.ai.storage import _write_cache as write_cache


def _observation(identity: str) -> Observation:
    return Observation(
        version=1,
        observation_id=identity,
        kind="test.metric",
        occurred_at=datetime.now(timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ("second_close", "cancel_close"))
async def test_write_cache_close_keeps_callback_owned_until_its_real_result(
    monkeypatch: pytest.MonkeyPatch, interruption: str,
) -> None:
    monkeypatch.setattr(write_cache, "_CLOSE_DEADLINE_SECONDS", 0.01)
    entered = asyncio.Event()
    cancelling = asyncio.Event()
    release = asyncio.Event()
    returned = asyncio.Event()

    async def sink(batch: tuple[tuple[str, object, bytes | None], ...]) -> None:
        assert tuple(item[1] for item in batch) == ("head",)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
        returned.set()

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink)
    assert await scope.try_submit("head", b"head")
    await scope.flush(timeout_seconds=0)
    await asyncio.wait_for(entered.wait(), 1)
    assert await scope.try_submit("tail", b"tail")
    closing = asyncio.create_task(scope.close())
    second = None
    try:
        await asyncio.wait_for(cancelling.wait(), 1)
        if interruption == "second_close":
            second = asyncio.create_task(scope.close())
        else:
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            second = asyncio.create_task(scope.close())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert closing.cancelled() if interruption == "cancel_close" else not closing.done()
        assert second is None or not second.done()
        assert not returned.is_set()
        assert scope.status().pending == 2
        assert scope.status().confirmed == scope.status().unconfirmed == 0
    finally:
        release.set()
        if not closing.cancelled():
            await asyncio.wait_for(closing, 1)
        if second is not None:
            await asyncio.wait_for(second, 1)
    status = scope.status()
    assert returned.is_set()
    assert status.confirmed == status.unconfirmed == 1
    assert status.pending == 0
    assert status.last_failure_code == "CLOSE_TIMEOUT"
    assert status.queue_size == 0
    assert (await scope.flush()).completed
    await scope.close()
    assert scope.status() == status


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", (False, True))
async def test_write_cache_callback_cancellation_does_not_strand_later_admissions(
    synchronous: bool,
) -> None:
    async def write(batch: tuple[tuple[str, object, bytes | None], ...]) -> None:
        if tuple(item[1] for item in batch) == ("cancelled",):
            raise asyncio.CancelledError
        assert tuple(item[1] for item in batch) == ("next",)

    def sink(batch: tuple[tuple[str, object, bytes | None], ...]) -> Awaitable[None]:
        if synchronous and tuple(item[1] for item in batch) == ("cancelled",):
            raise asyncio.CancelledError
        return write(batch)

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink)
    try:
        assert await scope.try_submit("cancelled", b"cancelled")
        assert (await scope.flush()).completed
        assert scope.status().unconfirmed == 1
        assert scope.status().write_failures == 1
        assert scope.status().last_failure_code == "WRITE_CANCELLED"
        assert scope.status().accepting
        assert await scope.try_submit("next", b"next")
        assert (await scope.flush()).completed
        assert scope.status().confirmed == 1
    finally:
        await scope.close()


@pytest.mark.asyncio
async def test_write_cache_capacity_includes_inflight_and_releases_after_ack() -> None:
    batches: list[tuple[int, ...]] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def sink(batch: tuple[tuple[str, object, bytes | None], ...]) -> None:
        batches.append(tuple(item[1] for item in batch))
        entered.set()
        await release.wait()

    cache = InMemoryCacheBackend(max_bytes=1024, max_size=3)
    scope = BatchWriteScope(cache, sink)
    try:
        for item in range(3):
            assert await scope.try_submit(item, str(item).encode())
        await scope.flush(timeout_seconds=0)
        await asyncio.wait_for(entered.wait(), 1)
        assert not await scope.try_submit(3, b"3")
        assert scope.status().pending == 3
        assert scope.status().queue_size == scope.status().high_watermark == 3
        release.set()
        assert (await scope.flush()).completed
        assert await scope.try_submit(3, b"3")
        assert (await scope.flush()).completed
        assert batches == [(0, 1, 2), (3,)]
    finally:
        release.set()
        await scope.close()


@pytest.mark.asyncio
async def test_write_cache_callback_unknown_outcome_is_not_confirmed() -> None:
    visible: list[str] = []

    async def sink(batch: tuple[tuple[str, object, bytes | None], ...]) -> None:
        visible.extend(item[1] for item in batch)
        raise OSError("reply unavailable after write")

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink)
    assert await scope.try_submit("visible", b"visible")
    await scope.close()
    status = scope.status()
    assert visible == ["visible"]
    assert status.confirmed == 0
    assert status.unconfirmed == 1
    assert status.pending == 0
    assert status.write_failures == 1


@pytest.mark.asyncio
async def test_write_cache_write_timeout_does_not_stop_following_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache, "_WRITE_TIMEOUT_SECONDS", 0.01)
    calls = 0

    async def sink(batch: tuple[tuple[str, object, bytes | None], ...]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        else:
            assert tuple(item[1] for item in batch) == ("next",)

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink)
    try:
        assert await scope.try_submit("timeout", b"timeout")
        assert (await scope.flush()).completed
        assert scope.status().last_failure_code == "WRITE_TIMEOUT"
        assert scope.status().unconfirmed == 1
        assert await scope.try_submit("next", b"next")
        assert (await scope.flush()).completed
        assert scope.status().confirmed == 1
    finally:
        await scope.close()


@pytest.mark.parametrize("options", (
    {"max_bytes": -1}, {"max_bytes": True}, {"max_bytes": 1.5},
    {"max_bytes": 1, "max_size": -1}, {"max_bytes": 1, "max_size": True},
))
def test_metric_cache_rejects_invalid_capacity(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        InMemoryCacheBackend(**options)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_write_cache_completed_flush_does_not_bypass_later_batch_window() -> None:
    batches = []

    async def sink(batch) -> None:
        batches.append(tuple(entry[1] for entry in batch))

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink, batch_size=128, flush_interval=60)
    try:
        assert await scope.try_submit("prefix", b"prefix")
        assert (await scope.flush()).completed
        assert batches == [("prefix",)]
        assert await scope.try_submit("later", b"later")
        await asyncio.sleep(0)
        assert scope.status().confirmed == 1
        assert scope.status().pending == 1
        assert batches == [("prefix",)]
    finally:
        await scope.close()
    assert scope.status().confirmed == 2


@pytest.mark.asyncio
async def test_write_cache_expired_queued_batch_does_not_start_another_window(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(write_cache, "monotonic", lambda: clock[0])
    release = asyncio.Event()
    batches = asyncio.Queue()

    async def sink(batch) -> None:
        batches.put_nowait(tuple(entry[1] for entry in batch))
        await release.wait()

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink, batch_size=2, flush_interval=60)
    try:
        assert await scope.try_submit("first", b"first")
        assert await scope.try_submit("second", b"second")
        assert await asyncio.wait_for(batches.get(), 1) == ("first", "second")
        clock[0] = 0.1
        assert await scope.try_submit("queued", b"queued")
        clock[0] = 60.1
        release.set()
        assert await asyncio.wait_for(batches.get(), 0.1) == ("queued",)
    finally:
        release.set()
        await scope.close()


@pytest.mark.asyncio
async def test_write_cache_flush_cancellation_preserves_owned_accepted_prefix() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def sink(batch) -> None:
        entered.set()
        await release.wait()

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink, batch_size=128, flush_interval=60)
    try:
        assert await scope.try_submit("accepted", b"accepted")
        await scope.flush(timeout_seconds=0)
        await asyncio.wait_for(entered.wait(), 1)
        flushing = asyncio.create_task(scope.flush())
        await asyncio.sleep(0)
        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        assert scope.status().pending == 1
        assert scope.status().unconfirmed == 0
    finally:
        release.set()
        await scope.close()
    assert scope.status().confirmed == 1
    assert scope.status().pending == scope.status().unconfirmed == 0


@pytest.mark.asyncio
async def test_write_cache_flush_wakes_unexpired_tail_behind_inflight_prefix() -> None:
    release = asyncio.Event()
    batches = asyncio.Queue()

    async def sink(batch) -> None:
        batches.put_nowait(tuple(entry[1] for entry in batch))
        await release.wait()

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink, batch_size=2, flush_interval=60)
    flushing = None
    try:
        assert await scope.try_submit("first", b"first")
        assert await scope.try_submit("second", b"second")
        assert await asyncio.wait_for(batches.get(), 1) == ("first", "second")
        assert await scope.try_submit("queued", b"queued")
        flushing = asyncio.create_task(scope.flush(timeout_seconds=1))
        await asyncio.sleep(0)
        release.set()
        assert (await flushing).completed
        assert await asyncio.wait_for(batches.get(), 0.1) == ("queued",)
        assert scope.status().confirmed == 3
    finally:
        release.set()
        if flushing is not None:
            await flushing
        await scope.close()


@pytest.mark.asyncio
async def test_write_cache_later_admission_does_not_extend_first_pending_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(write_cache, "monotonic", lambda: clock[0])
    batches = asyncio.Queue()

    async def sink(batch) -> None:
        batches.put_nowait(tuple(entry[1] for entry in batch))

    scope = BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), sink, batch_size=128, flush_interval=60)
    try:
        assert await scope.try_submit("first", b"first")
        await asyncio.sleep(0)
        clock[0] = 59.9
        assert await scope.try_submit("second", b"second")
        await asyncio.sleep(0)
        assert batches.empty()
        clock[0] = 60.0
        assert await scope.try_submit("third", b"third")
        first_batch = await asyncio.wait_for(batches.get(), 1)
        # The expired timer may dispatch before the third submission is admitted.
        assert first_batch[:2] == ("first", "second")
        await scope.close()
        persisted = list(first_batch)
        while not batches.empty():
            persisted.extend(batches.get_nowait())
        assert persisted == ["first", "second", "third"]
    finally:
        await scope.close()
