#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime metric buffer deadline and shape regressions."""

import asyncio
from datetime import datetime, timezone

import pytest
from linktools.ai.core import Page
from linktools.ai.observe import Metrics, Observation
from linktools.ai.runtime import _metrics as runtime_metrics


class _NeverCompletesStore:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def put_definition(self, namespace: str, definition: object) -> object:
        del namespace, definition
        raise AssertionError("buffer never defines metrics")

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> None:
        del namespace, name, revision
        raise AssertionError("buffer never reads definitions")

    async def latest_definition(self, namespace: str, name: str) -> None:
        del namespace, name
        raise AssertionError("buffer never reads definitions")

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


@pytest.mark.asyncio
async def test_metric_buffer_close_deadline_cancels_writer_and_settles_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_metrics, "_WRITE_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(runtime_metrics, "_CLOSE_DEADLINE_SECONDS", 0.05)
    store = _NeverCompletesStore()
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="deadline")  # type: ignore[arg-type]
    )
    observation = Observation(
        version=1,
        observation_id="deadline-observation",
        kind="test.deadline",
        occurred_at=datetime.now(timezone.utc),
        source_namespace="workspace",
        tenant_id="default",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(),
    )

    assert buffer._writer is None  # type: ignore[attr-defined]
    assert buffer.try_record(observation) is True
    writer = buffer._writer  # type: ignore[attr-defined]
    assert writer is not None
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    await asyncio.wait_for(buffer.close(), timeout=1)

    assert writer.done()
    await asyncio.wait_for(buffer._queue.join(), timeout=0.1)  # type: ignore[attr-defined]


def test_metric_buffer_rejects_invalid_shape_without_raising() -> None:
    metrics = Metrics.in_memory(namespace="shape")

    async def exercise() -> None:
        buffer = runtime_metrics._RuntimeMetricBuffer(metrics)
        assert buffer._writer is None  # type: ignore[attr-defined]
        assert buffer.try_record(object()) is False  # type: ignore[arg-type]
        assert buffer._writer is None  # type: ignore[attr-defined]
        await buffer.close()
        assert buffer._writer is None  # type: ignore[attr-defined]

    asyncio.run(exercise())
