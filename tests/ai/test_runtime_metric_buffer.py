#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime metric buffer backpressure and lifecycle semantics."""

import asyncio
from datetime import datetime, timezone

import pytest
from linktools.ai.core import Page
from linktools.ai.observe import Metrics, Observation
from linktools.ai.runtime import _metrics as runtime_metrics


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
    monkeypatch.setattr(runtime_metrics, "_QUEUE_CAPACITY", 1)
    store = _BlockingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="backpressure")  # type: ignore[arg-type]
    )

    assert buffer.try_record(_observation("first")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert buffer.try_record(_observation("second")) is True
    assert buffer.try_record(_observation("third")) is False

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
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="flush")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("accepted")) is True
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
    monkeypatch.setattr(runtime_metrics, "_WRITE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(runtime_metrics, "_CLOSE_DEADLINE_SECONDS", 0.05)
    store = _BlockingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="close")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("blocked")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    await asyncio.wait_for(buffer.close(), timeout=1)
    status = buffer.status()
    assert status.accepting is False
    assert status.pending == 0
    assert status.lost == 1
    assert status.last_failure_code == "CLOSE_TIMEOUT"


@pytest.mark.asyncio
async def test_metric_buffer_cancelled_batch_settles_owned_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_metrics, "_FLUSH_INTERVAL_SECONDS", 60.0)
    store = _BlockingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="batch-cancel")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("batched")) is True
    await asyncio.sleep(0)
    assert buffer.status().queue_size == 0
    assert not store.entered.is_set()

    assert buffer._writer is not None
    buffer._writer.cancel()
    await asyncio.gather(buffer._writer, return_exceptions=True)
    await buffer.close()

    status = buffer.status()
    assert status.accepted == status.lost == 1
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
    monkeypatch.setattr(runtime_metrics, "_BATCH_SIZE", 1)
    monkeypatch.setattr(runtime_metrics, "_CLOSE_DEADLINE_SECONDS", 0.01)
    store = _BlockingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="close-completion")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("completing")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    join = buffer._queue.join
    joining = asyncio.Event()

    async def join_at_write_completion() -> None:
        try:
            joining.set()
            await join()
        except asyncio.CancelledError:
            # Complete the real sink while close leaves its drain wait, before
            # writer cancellation races with wait_for's completed inner task.
            asyncio.get_running_loop().call_soon(store.release.set)
            raise

    monkeypatch.setattr(buffer._queue, "join", join_at_write_completion)
    close = asyncio.create_task(buffer.close())
    try:
        await asyncio.wait_for(joining.wait(), timeout=1)
        if interrupt == "cancel":
            close.cancel()
        done, _ = await asyncio.wait((close,), timeout=1)
        assert close in done
    finally:
        if not close.done():
            assert buffer._writer is not None
            buffer._writer.cancel()
        await asyncio.wait_for(close, timeout=1)

    status = buffer.status()
    assert status.persisted == int(store.written.is_set())
    assert status.accepted == status.persisted + status.lost == 1
    assert status.pending == 0
    assert status.queue_size == 0


@pytest.mark.asyncio
async def test_metric_buffer_flush_cancellation_preserves_accepted_observations() -> None:
    store = _BlockingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="flush-cancel")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("accepted")) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    flush = asyncio.create_task(buffer.flush(timeout_seconds=1))
    await asyncio.sleep(0)
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert buffer.status().pending == 1
    assert buffer.status().lost == 0

    store.release.set()
    await asyncio.gather(buffer.close(), buffer.close())
    status = buffer.status()
    assert status.persisted == 1
    assert status.pending == 0
    assert status.lost == 0


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
    monkeypatch.setattr(runtime_metrics, "_BATCH_SIZE", 1)
    monkeypatch.setattr(runtime_metrics, "_WRITE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(runtime_metrics, "_CLOSE_DEADLINE_SECONDS", 0.01)
    store = _CancellingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="close-order")  # type: ignore[arg-type]
    )
    first = _observation("inflight")
    assert buffer.try_record(first) is True
    await buffer.flush(timeout_seconds=0)
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    flush = asyncio.create_task(buffer.flush(timeout_seconds=1))
    await asyncio.sleep(0)
    assert buffer.try_record(_observation("queued")) is True
    close = asyncio.create_task(buffer.close())
    try:
        await asyncio.wait_for(store.cancelling.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not close.done()
        assert not flush.done()
        assert buffer.status().pending == 2
        assert buffer.status().lost == 0
    finally:
        store.finalize.set()
        await asyncio.wait_for(asyncio.gather(close, flush), timeout=1)

    assert flush.result().completed is True
    assert store.observations == [first]
    assert buffer.status().lost == 2
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


async def _wait_queue_empty(buffer: runtime_metrics._MetricBuffer) -> None:
    while buffer.status().queue_size:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_metric_buffer_default_batch_reaches_128_before_three_seconds() -> None:
    assert runtime_metrics._BATCH_SIZE == 128
    assert runtime_metrics._FLUSH_INTERVAL_SECONDS == 3.0
    store = _RecordingStore()
    store.release.set()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="batch-size")  # type: ignore[arg-type]
    )
    try:
        for index in range(127):
            assert buffer.try_record(_observation(str(index))) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        assert not store.entered.is_set()
        assert buffer.try_record(_observation("127")) is True
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert len(batch) == 128
    finally:
        await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_window_stays_with_first_accepted_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(runtime_metrics, "monotonic", lambda: clock[0])
    store = _RecordingStore()
    store.release.set()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="batch-window")  # type: ignore[arg-type]
    )
    try:
        assert buffer.try_record(_observation("first")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        clock[0] = 2.9
        assert buffer.try_record(_observation("second")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        assert not store.entered.is_set()
        clock[0] = 3.0
        assert buffer.try_record(_observation("third")) is True
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
    monkeypatch.setattr(runtime_metrics, "_FLUSH_INTERVAL_SECONDS", 60.0)
    store = _RecordingStore()
    store.release.set()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="explicit-drain")  # type: ignore[arg-type]
    )
    try:
        assert buffer.try_record(_observation("partial")) is True
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
        assert buffer.status().persisted == 1
        assert buffer.status().pending == 0
    finally:
        await buffer.close()


@pytest.mark.asyncio
async def test_metric_buffer_expired_queued_batch_does_not_start_another_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(runtime_metrics, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime_metrics, "_BATCH_SIZE", 2)
    store = _RecordingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="queued-window")  # type: ignore[arg-type]
    )
    try:
        assert buffer.try_record(_observation("first")) is True
        assert buffer.try_record(_observation("second")) is True
        await asyncio.wait_for(store.next_batch.get(), timeout=1)
        clock[0] = 0.1
        assert buffer.try_record(_observation("queued")) is True
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
    monkeypatch.setattr(runtime_metrics, "_BATCH_SIZE", 2)
    monkeypatch.setattr(runtime_metrics, "_FLUSH_INTERVAL_SECONDS", 60.0)
    store = _RecordingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="inflight-flush")  # type: ignore[arg-type]
    )
    try:
        assert buffer.try_record(_observation("first")) is True
        assert buffer.try_record(_observation("second")) is True
        await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert buffer.try_record(_observation("queued")) is True
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
    monkeypatch.setattr(runtime_metrics, "_FLUSH_INTERVAL_SECONDS", 0.01)
    store = _RecordingStore()
    store.release.set()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="idle-window")  # type: ignore[arg-type]
    )
    try:
        assert buffer.try_record(_observation("alone")) is True
        batch = await asyncio.wait_for(store.next_batch.get(), timeout=1)
        assert tuple(item.observation_id for item in batch) == ("alone",)
    finally:
        await buffer.close()
    assert buffer.status().persisted == 1


@pytest.mark.asyncio
async def test_metric_buffer_completed_flush_does_not_bypass_later_batch_window() -> None:
    store = _RecordingStore()
    store.release.set()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="flush-prefix")  # type: ignore[arg-type]
    )
    try:
        assert buffer.try_record(_observation("prefix")) is True
        assert (await buffer.flush(timeout_seconds=0.05)).completed is True
        assert buffer.try_record(_observation("later")) is True
        await asyncio.wait_for(_wait_queue_empty(buffer), timeout=1)
        assert buffer.status().persisted == 1
        assert buffer.status().pending == 1
        assert len(store.batches) == 1
    finally:
        await buffer.close()
    assert buffer.status().persisted == 2
