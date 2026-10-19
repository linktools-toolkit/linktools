#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared batch-write backpressure, batching and lifecycle semantics."""

import asyncio
from collections.abc import Awaitable
from datetime import datetime, timezone

import pytest
from linktools.ai.core import Page
from linktools.ai.observe import MetricRecorder, Metrics, Observation
from linktools.ai.storage import InMemoryCacheBackend, BatchWriteScope
from linktools.ai.core import CommitPolicy
from linktools.ai.storage import _write_cache as write_cache


class _BlockingStore:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.written = asyncio.Event()

    async def put_definition(self, namespace: str, definition: object) -> object:
        del namespace, definition
        raise AssertionError("metric buffer does not define metrics")

    async def get_definition(self, namespace: str, name: str, revision: int) -> None:
        del namespace, name, revision
        return None

    async def latest_definition(self, namespace: str, name: str) -> None:
        del namespace, name
        return None

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace, observations
        self.entered.set()
        await self.release.wait()
        self.written.set()

    async def scan_observations(
        self,
        namespace: str,
        kind: str,
        start: datetime,
        end: datetime,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[Observation]:
        del namespace, kind, start, end, cursor, limit
        return Page(())

    async def prune_observations(self, namespace: str, *, before: datetime) -> int:
        del namespace, before
        return 0


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
async def test_metric_buffer_reports_backpressure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _BlockingStore()
    buffer = Metrics.from_store(
        store, namespace="backpressure",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024, max_size=2),
    ).recorder()

    assert await buffer.try_record(_observation("first")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert await buffer.try_record(_observation("second")) is True
    assert await buffer.try_record(_observation("third")) is False

    status = buffer.status()
    assert status.queue_size == 2
    assert status.queue_capacity is None
    assert status.last_failure_code == "QUEUE_FULL"
    assert status.rejected == 1

    store.release.set()
    await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_flush_waits_for_accepted_observations() -> None:
    store = _BlockingStore()
    buffer = Metrics.from_store(
        store, namespace="flush",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    assert await buffer.try_record(_observation("accepted")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    pending = await buffer.flush(timeout_seconds=0)
    assert pending.completed is False
    assert pending.status.accepted == 1
    assert pending.status.persisted == 0
    assert pending.status.pending == 1

    store.release.set()
    flushed = await buffer.flush(timeout_seconds=1)
    assert flushed.completed is True
    assert flushed.status.persisted == 1
    assert flushed.status.pending == 0
    await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_close_has_a_bounded_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache, "_WRITE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(write_cache, "_CLOSE_DEADLINE_SECONDS", 0.05)
    store = _BlockingStore()
    buffer = Metrics.from_store(
        store, namespace="close",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    assert await buffer.try_record(_observation("blocked")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    await asyncio.wait_for(buffer.close(), timeout=1)
    status = buffer.status()
    assert status.accepting is False
    assert status.pending == 0
    assert status.unconfirmed == 1
    assert status.last_failure_code == "CLOSE_TIMEOUT"


@pytest.mark.asyncio
async def test_metric_buffer_cancelled_batch_settles_owned_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _BlockingStore()
    buffer = Metrics.from_store(
        store, namespace="batch-cancel",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    assert await buffer.try_record(_observation("batched")) is True
    await asyncio.sleep(0)
    assert buffer.status().queue_size == 1
    assert not store.entered.is_set()

    assert buffer._scope._writer is not None
    buffer._scope._writer.cancel()
    await asyncio.gather(buffer._scope._writer, return_exceptions=True)
    await buffer.close()

    status = buffer.status()
    assert status.accepted == status.unconfirmed == 1
    assert status.persisted == status.pending == status.queue_size == 0
    assert not store.entered.is_set()
    assert (await buffer.flush(timeout_seconds=0)).completed is True
    await buffer.close()
    assert buffer.status() == status


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ("cancel", "deadline"))
async def test_metric_buffer_close_stops_when_write_completion_races_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    interrupt: str,
) -> None:
    monkeypatch.setattr(write_cache, "_CLOSE_DEADLINE_SECONDS", 0.01)
    store = _BlockingStore()
    buffer = Metrics.from_store(
        store, namespace="close-completion",  # type: ignore[arg-type]
        commit=CommitPolicy.batch(batch_size=1, flush_interval=60.0),
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    assert await buffer.try_record(_observation("completing")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    request_stop = buffer._scope._request_stop

    def stop_at_write_completion(code: str) -> None:
        # Coordinate the real callback with the stop request without replacing
        # wait_for or the callback task's cancellation implementation.
        asyncio.get_running_loop().call_soon(store.release.set)
        request_stop(code)

    monkeypatch.setattr(buffer._scope, "_request_stop", stop_at_write_completion)
    close = asyncio.create_task(buffer.close())
    try:
        await asyncio.sleep(0)
        if interrupt == "cancel":
            close.cancel()
        done, _ = await asyncio.wait((close,), timeout=1)
        assert close in done
    finally:
        store.release.set()
        if interrupt == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await close
        else:
            await asyncio.wait_for(close, timeout=1)
        await asyncio.wait_for(buffer.close(), timeout=1)

    status = buffer.status()
    assert status.persisted == int(store.written.is_set())
    assert status.accepted == status.persisted + status.unconfirmed == 1
    assert status.pending == 0
    assert status.queue_size == 0


@pytest.mark.asyncio
async def test_metric_buffer_flush_cancellation_preserves_accepted_observations() -> None:
    store = _BlockingStore()
    buffer = Metrics.from_store(
        store, namespace="flush-cancel",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    assert await buffer.try_record(_observation("accepted")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    flush = asyncio.create_task(buffer.flush(timeout_seconds=1))
    await asyncio.sleep(0)
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert buffer.status().pending == 1
    assert buffer.status().unconfirmed == 0

    store.release.set()
    await asyncio.gather(buffer.close(), buffer.close())
    status = buffer.status()
    assert status.persisted == 1
    assert status.pending == 0
    assert status.unconfirmed == 0


class _CancellingStore(_BlockingStore):
    def __init__(self) -> None:
        super().__init__()
        self.cancelling = asyncio.Event()
        self.finalize = asyncio.Event()
        self.observations: list[Observation] = []

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        self.observations.extend(observations)
        try:
            await super().put_observations(namespace, observations)
        except asyncio.CancelledError:
            self.cancelling.set()
            await self.finalize.wait()
            raise


@pytest.mark.asyncio
async def test_metric_buffer_close_settles_inflight_before_queued_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache, "_WRITE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(write_cache, "_CLOSE_DEADLINE_SECONDS", 0.01)
    store = _CancellingStore()
    buffer = Metrics.from_store(
        store, namespace="close-order",  # type: ignore[arg-type]
        commit=CommitPolicy.batch(batch_size=1, flush_interval=60.0),
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    first = _observation("inflight")
    assert await buffer.try_record(first) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    flush = asyncio.create_task(buffer.flush(timeout_seconds=1))
    await asyncio.sleep(0)
    assert await buffer.try_record(_observation("queued")) is True
    close = asyncio.create_task(buffer.close())
    try:
        await asyncio.wait_for(store.cancelling.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not close.done()
        assert not flush.done()
        assert buffer.status().pending == 2
        assert buffer.status().unconfirmed == 0
    finally:
        store.finalize.set()
        await asyncio.wait_for(asyncio.gather(close, flush), timeout=1)

    assert flush.result().completed is True
    assert store.observations == [first]
    assert buffer.status().unconfirmed == 2
    assert buffer.status().pending == 0
    assert buffer.status().queue_size == 0


class _RecordingStore(_BlockingStore):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[Observation, ...]] = []
        self.next_batch: asyncio.Queue[tuple[Observation, ...]] = asyncio.Queue()

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        self.batches.append(observations)
        self.next_batch.put_nowait(observations)
        await super().put_observations(namespace, observations)


async def _settle_admission(buffer: MetricRecorder) -> None:
    assert buffer.status().pending > 0
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_metric_buffer_default_batch_reaches_128_before_one_minute() -> None:
    store = _RecordingStore()
    store.release.set()
    buffer = Metrics.from_store(
        store, namespace="batch-size",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        for index in range(127):
            assert await buffer.try_record(_observation(str(index))) is True
        await asyncio.wait_for(_settle_admission(buffer), timeout=1)
        assert not store.entered.is_set()
        assert await buffer.try_record(_observation("127")) is True
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert len(batch) == 128
    finally:
        await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_window_stays_with_first_accepted_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(write_cache, "monotonic", lambda: clock[0])
    store = _RecordingStore()
    store.release.set()
    buffer = Metrics.from_store(
        store, namespace="batch-window",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        assert await buffer.try_record(_observation("first")) is True
        await asyncio.wait_for(_settle_admission(buffer), timeout=1)
        clock[0] = 59.9
        assert await buffer.try_record(_observation("second")) is True
        await asyncio.wait_for(_settle_admission(buffer), timeout=1)
        assert not store.entered.is_set()
        clock[0] = 60.0
        assert await buffer.try_record(_observation("third")) is True
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=0.05)
        assert tuple(item.observation_id for item in batch) == ("first", "second", "third")
    finally:
        await buffer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("flush", "close"))
async def test_metric_buffer_explicit_drain_wakes_partial_batch(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store = _RecordingStore()
    store.release.set()
    buffer = Metrics.from_store(
        store, namespace="explicit-drain",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        assert await buffer.try_record(_observation("partial")) is True
        await asyncio.wait_for(_settle_admission(buffer), timeout=1)
        if operation == "flush":
            result = await buffer.flush(timeout_seconds=0.05)
            assert result.completed is True
        else:
            closing = asyncio.create_task(buffer.close())
            done, _ = await asyncio.wait((closing,), timeout=0.05)
            if closing not in done:
                closing.cancel()
                await closing
            assert closing in done
        assert buffer.status().persisted == 1
        assert buffer.status().pending == 0
    finally:
        await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_expired_queued_batch_does_not_start_another_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(write_cache, "monotonic", lambda: clock[0])
    store = _RecordingStore()
    buffer = Metrics.from_store(
        store, namespace="queued-window",  # type: ignore[arg-type]
        commit=CommitPolicy.batch(batch_size=2, flush_interval=60.0),
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        assert await buffer.try_record(_observation("first")) is True
        assert await buffer.try_record(_observation("second")) is True
        await asyncio.wait_for(store.next_batch.get(), timeout=1)
        clock[0] = 0.1
        assert await buffer.try_record(_observation("queued")) is True
        clock[0] = 60.1
        store.release.set()
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=0.05)
        assert tuple(item.observation_id for item in batch) == ("queued",)
    finally:
        store.release.set()
        await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_flush_wakes_prefix_queued_behind_inflight_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RecordingStore()
    buffer = Metrics.from_store(
        store, namespace="inflight-flush",  # type: ignore[arg-type]
        commit=CommitPolicy.batch(batch_size=2, flush_interval=60.0),
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        assert await buffer.try_record(_observation("first")) is True
        assert await buffer.try_record(_observation("second")) is True
        await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert await buffer.try_record(_observation("queued")) is True
        flushing = asyncio.create_task(buffer.flush(timeout_seconds=0.05))
        await asyncio.sleep(0)
        store.release.set()
        assert (await flushing).completed is True
        assert buffer.status().persisted == 3
        assert len(store.batches) == 2
    finally:
        store.release.set()
        await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_timer_writes_without_another_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _RecordingStore()
    store.release.set()
    buffer = Metrics.from_store(
        store, namespace="idle-window",  # type: ignore[arg-type]
        commit=CommitPolicy.batch(batch_size=128, flush_interval=0.01),
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        assert await buffer.try_record(_observation("alone")) is True
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert tuple(item.observation_id for item in batch) == ("alone",)
    finally:
        await buffer.close()
    assert buffer.status().persisted == 1


@pytest.mark.asyncio
async def test_metric_buffer_completed_flush_does_not_bypass_later_batch_window() -> None:
    store = _RecordingStore()
    store.release.set()
    buffer = Metrics.from_store(
        store, namespace="flush-prefix",  # type: ignore[arg-type]
        cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    ).recorder()
    try:
        assert await buffer.try_record(_observation("prefix")) is True
        assert (await buffer.flush(timeout_seconds=0.05)).completed is True
        assert await buffer.try_record(_observation("later")) is True
        await asyncio.wait_for(_settle_admission(buffer), timeout=1)
        assert buffer.status().persisted == 1
        assert buffer.status().pending == 1
        assert len(store.batches) == 1
    finally:
        await buffer.close()
    assert buffer.status().persisted == 2


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
