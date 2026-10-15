#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared write-cache backpressure, batching and lifecycle semantics."""

import asyncio
from datetime import datetime, timezone

import pytest
from linktools.ai.core import Page
from linktools.ai.observe import Metrics, Observation
from linktools.ai.storage import WriteCache, WriteCacheScope
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
    buffer = WriteCache(queue_capacity=1).open(
        Metrics.from_store(store, namespace="backpressure").record_observations  # type: ignore[arg-type]
    )

    assert await buffer.try_submit(_observation("first")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert await buffer.try_submit(_observation("second")) is True
    assert await buffer.try_submit(_observation("third")) is False

    status = buffer.status()
    assert status.queue_size == 1
    assert status.queue_capacity == 1
    assert status.last_failure_code == "QUEUE_FULL"
    assert status.rejected == 1

    store.release.set()
    await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_flush_waits_for_accepted_observations() -> None:
    store = _BlockingStore()
    buffer = WriteCache().open(
        Metrics.from_store(store, namespace="flush").record_observations  # type: ignore[arg-type]
    )
    assert await buffer.try_submit(_observation("accepted")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    pending = await buffer.flush(timeout_seconds=0)
    assert pending.completed is False
    assert pending.status.accepted == 1
    assert pending.status.confirmed == 0
    assert pending.status.pending == 1

    store.release.set()
    flushed = await buffer.flush(timeout_seconds=1)
    assert flushed.completed is True
    assert flushed.status.confirmed == 1
    assert flushed.status.pending == 0
    await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_close_has_a_bounded_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(write_cache, "_WRITE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(write_cache, "_CLOSE_DEADLINE_SECONDS", 0.05)
    store = _BlockingStore()
    buffer = WriteCache().open(
        Metrics.from_store(store, namespace="close").record_observations  # type: ignore[arg-type]
    )
    assert await buffer.try_submit(_observation("blocked")) is True
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
    buffer = WriteCache(flush_interval_seconds=60.0).open(
        Metrics.from_store(store, namespace="batch-cancel").record_observations  # type: ignore[arg-type]
    )
    assert await buffer.try_submit(_observation("batched")) is True
    await asyncio.sleep(0)
    assert buffer.status().queue_size == 0
    assert not store.entered.is_set()

    assert buffer._writer is not None
    buffer._writer.cancel()
    await asyncio.gather(buffer._writer, return_exceptions=True)
    await buffer.close()

    status = buffer.status()
    assert status.accepted == status.unconfirmed == 1
    assert status.confirmed == status.pending == status.queue_size == 0
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
    buffer = WriteCache(batch_size=1).open(
        Metrics.from_store(store, namespace="close-completion").record_observations  # type: ignore[arg-type]
    )
    assert await buffer.try_submit(_observation("completing")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    request_stop = buffer._request_stop

    def stop_at_write_completion(code: str) -> None:
        # Coordinate the real callback with the stop request without replacing
        # wait_for or the callback task's cancellation implementation.
        asyncio.get_running_loop().call_soon(store.release.set)
        request_stop(code)

    monkeypatch.setattr(buffer, "_request_stop", stop_at_write_completion)
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
    assert status.confirmed == int(store.written.is_set())
    assert status.accepted == status.confirmed + status.unconfirmed == 1
    assert status.pending == 0
    assert status.queue_size == 0


@pytest.mark.asyncio
async def test_metric_buffer_flush_cancellation_preserves_accepted_observations() -> None:
    store = _BlockingStore()
    buffer = WriteCache().open(
        Metrics.from_store(store, namespace="flush-cancel").record_observations  # type: ignore[arg-type]
    )
    assert await buffer.try_submit(_observation("accepted")) is True
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
    assert status.confirmed == 1
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
    buffer = WriteCache(batch_size=1).open(
        Metrics.from_store(store, namespace="close-order").record_observations  # type: ignore[arg-type]
    )
    first = _observation("inflight")
    assert await buffer.try_submit(first) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    flush = asyncio.create_task(buffer.flush(timeout_seconds=1))
    await asyncio.sleep(0)
    assert await buffer.try_submit(_observation("queued")) is True
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


async def _wait_queue_empty(buffer: WriteCacheScope[Observation]) -> None:
    while buffer.status().queue_size:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_metric_buffer_default_batch_reaches_128_before_three_seconds() -> None:
    assert WriteCache().batch_size == 128
    assert WriteCache().flush_interval_seconds == 3.0
    store = _RecordingStore()
    store.release.set()
    buffer = WriteCache().open(
        Metrics.from_store(store, namespace="batch-size").record_observations  # type: ignore[arg-type]
    )
    try:
        for index in range(127):
            assert await buffer.try_submit(_observation(str(index))) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        assert not store.entered.is_set()
        assert await buffer.try_submit(_observation("127")) is True
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
    buffer = WriteCache().open(
        Metrics.from_store(store, namespace="batch-window").record_observations  # type: ignore[arg-type]
    )
    try:
        assert await buffer.try_submit(_observation("first")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        clock[0] = 2.9
        assert await buffer.try_submit(_observation("second")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        assert not store.entered.is_set()
        clock[0] = 3.0
        assert await buffer.try_submit(_observation("third")) is True
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
    buffer = WriteCache(flush_interval_seconds=60.0).open(
        Metrics.from_store(store, namespace="explicit-drain").record_observations  # type: ignore[arg-type]
    )
    try:
        assert await buffer.try_submit(_observation("partial")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
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
        assert buffer.status().confirmed == 1
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
    buffer = WriteCache(batch_size=2).open(
        Metrics.from_store(store, namespace="queued-window").record_observations  # type: ignore[arg-type]
    )
    try:
        assert await buffer.try_submit(_observation("first")) is True
        assert await buffer.try_submit(_observation("second")) is True
        await asyncio.wait_for(store.next_batch.get(), timeout=1)
        clock[0] = 0.1
        assert await buffer.try_submit(_observation("queued")) is True
        clock[0] = 3.1
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
    buffer = WriteCache(batch_size=2, flush_interval_seconds=60.0).open(
        Metrics.from_store(store, namespace="inflight-flush").record_observations  # type: ignore[arg-type]
    )
    try:
        assert await buffer.try_submit(_observation("first")) is True
        assert await buffer.try_submit(_observation("second")) is True
        await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert await buffer.try_submit(_observation("queued")) is True
        flushing = asyncio.create_task(buffer.flush(timeout_seconds=0.05))
        await asyncio.sleep(0)
        store.release.set()
        assert (await flushing).completed is True
        assert buffer.status().confirmed == 3
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
    buffer = WriteCache(flush_interval_seconds=0.01).open(
        Metrics.from_store(store, namespace="idle-window").record_observations  # type: ignore[arg-type]
    )
    try:
        assert await buffer.try_submit(_observation("alone")) is True
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert tuple(item.observation_id for item in batch) == ("alone",)
    finally:
        await buffer.close()
    assert buffer.status().confirmed == 1


@pytest.mark.asyncio
async def test_metric_buffer_completed_flush_does_not_bypass_later_batch_window() -> None:
    store = _RecordingStore()
    store.release.set()
    buffer = WriteCache().open(
        Metrics.from_store(store, namespace="flush-prefix").record_observations  # type: ignore[arg-type]
    )
    try:
        assert await buffer.try_submit(_observation("prefix")) is True
        assert (await buffer.flush(timeout_seconds=0.05)).completed is True
        assert await buffer.try_submit(_observation("later")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        assert buffer.status().confirmed == 1
        assert buffer.status().pending == 1
        assert len(store.batches) == 1
    finally:
        await buffer.close()
    assert buffer.status().confirmed == 2


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

    async def sink(batch: tuple[str, ...]) -> None:
        assert batch == ("head",)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
        returned.set()

    scope = WriteCache(batch_size=1).open(sink)
    assert await scope.try_submit("head")
    await asyncio.wait_for(entered.wait(), 1)
    assert await scope.try_submit("tail")
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
    await asyncio.wait_for(scope._queue.join(), 1)
    assert (await scope.flush()).completed
    await scope.close()
    assert scope.status() == status


@pytest.mark.asyncio
async def test_write_cache_callback_cancellation_does_not_strand_later_admissions() -> None:
    calls = 0

    async def sink(batch: tuple[str, ...]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        assert batch == ("next",)

    scope = WriteCache(batch_size=1).open(sink)
    try:
        assert await scope.try_submit("cancelled")
        assert (await scope.flush()).completed
        assert scope.status().unconfirmed == 1
        assert scope.status().accepting
        assert await scope.try_submit("next")
        assert (await scope.flush()).completed
        assert scope.status().confirmed == 1
    finally:
        await scope.close()


@pytest.mark.asyncio
async def test_write_cache_can_form_a_batch_larger_than_its_queue() -> None:
    batches: list[tuple[int, ...]] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def sink(batch: tuple[int, ...]) -> None:
        batches.append(batch)
        entered.set()
        await release.wait()

    scope = WriteCache(queue_capacity=1, batch_size=3).open(sink)
    try:
        for item in range(3):
            assert await scope.try_submit(item)
            while scope.status().queue_size:
                await asyncio.sleep(0)
        await asyncio.wait_for(entered.wait(), 1)
        assert await scope.try_submit(3)
        assert not await scope.try_submit(4)
        assert scope.status().pending == 4
        assert scope.status().queue_size == scope.status().high_watermark == 1
        release.set()
        assert (await scope.flush()).completed
        assert batches == [(0, 1, 2), (3,)]
    finally:
        release.set()
        await scope.close()


@pytest.mark.asyncio
async def test_write_cache_callback_unknown_outcome_is_not_confirmed() -> None:
    visible: list[str] = []

    async def sink(batch: tuple[str, ...]) -> None:
        visible.extend(batch)
        raise OSError("reply unavailable after write")

    scope = WriteCache().open(sink)
    assert await scope.try_submit("visible")
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

    async def sink(batch: tuple[str, ...]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        else:
            assert batch == ("next",)

    scope = WriteCache(batch_size=1).open(sink)
    try:
        assert await scope.try_submit("timeout")
        assert (await scope.flush()).completed
        assert scope.status().last_failure_code == "WRITE_TIMEOUT"
        assert scope.status().unconfirmed == 1
        assert await scope.try_submit("next")
        assert (await scope.flush()).completed
        assert scope.status().confirmed == 1
    finally:
        await scope.close()


@pytest.mark.parametrize("options", (
    {"queue_capacity": 0}, {"queue_capacity": True}, {"batch_size": -1},
    {"batch_size": 1.5}, {"flush_interval_seconds": 0},
    {"flush_interval_seconds": float("nan")},
    {"flush_interval_seconds": float("inf")},
))
def test_write_cache_rejects_invalid_configuration(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        WriteCache(**options)  # type: ignore[arg-type]
