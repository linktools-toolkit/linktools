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
    buffer = runtime_metrics._RuntimeMetricBuffer(
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
    buffer = runtime_metrics._RuntimeMetricBuffer(
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
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="close")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("blocked")) is True
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    await asyncio.wait_for(buffer.close(), timeout=1)
