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
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    await asyncio.wait_for(buffer.close(), timeout=1)
    status = buffer.status()
    assert status.accepting is False
    assert status.pending == 0
    assert status.lost == 1
    assert status.last_failure_code == "CLOSE_TIMEOUT"


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ("cancel", "deadline"))
async def test_metric_buffer_close_settles_observations_being_batched(
    monkeypatch: pytest.MonkeyPatch,
    interrupt: str,
) -> None:
    monkeypatch.setattr(runtime_metrics, "_FLUSH_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(runtime_metrics, "_CLOSE_DEADLINE_SECONDS", 0.01)
    store = _BlockingStore()
    buffer = runtime_metrics._MetricBuffer(
        Metrics.from_store(store, namespace="batch-close")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("batched")) is True
    await asyncio.sleep(0)
    assert buffer.status().queue_size == 0
    assert not store.entered.is_set()

    close = asyncio.create_task(buffer.close())
    if interrupt == "cancel":
        await asyncio.sleep(0)
        close.cancel()
    try:
        done, _ = await asyncio.wait((close,), timeout=1)
        assert close in done
        assert not store.entered.is_set()
    finally:
        store.release.set()
        await asyncio.wait_for(close, timeout=1)

    status = buffer.status()
    assert status.accepted == status.persisted + status.lost == 1
    assert status.pending == 0
    assert status.queue_size == 0
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
