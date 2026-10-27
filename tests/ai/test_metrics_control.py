#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Producer controls retain scope, settlement, and diagnostic information."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import CommitPolicy, JsonValue, TaskStatus
from linktools.ai.errors import AIError
from linktools.ai.observe import (
    MetricFlushResult, MetricRecorderStatus, Metrics, MetricsControl, Observation,
    RuntimeMetricsControl, RuntimeMetricsStatus,
)
from linktools.ai.observe.service import MetricsService
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.task.engine import TaskEngine
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext


def _status(**changes: object) -> MetricRecorderStatus:
    return replace(MetricRecorderStatus(True, True, 0, 0, 0, 0, 0, 0), **changes)


def test_aggregate_preserves_recorder_diagnostics_and_derives_counts() -> None:
    first = _status(accepted=4, persisted=2, pending=1, rejected=3, unconfirmed=1,
                    write_failures=2, queue_size=1, queue_capacity=4,
                    high_watermark=3, last_failure_code="WRITE_FAILED")
    second = _status(accepted=8, persisted=6, pending=2, rejected=2, unconfirmed=0,
                     write_failures=1, queue_size=2, queue_capacity=8,
                     high_watermark=7, last_failure_code="ENRICHMENT_FAILED", accepting=False)
    status = RuntimeMetricsStatus((first, second, _status(last_failure_code="WRITE_FAILED")))
    assert status.recorders == (first, second, _status(last_failure_code="WRITE_FAILED"))
    assert (status.accepted, status.persisted, status.pending, status.rejected,
            status.unconfirmed, status.write_failures, status.queue_size) == (12, 8, 3, 5, 1, 3, 3)
    assert status.enabled and not status.accepting
    assert status.failure_codes == ("WRITE_FAILED", "ENRICHMENT_FAILED")
    assert status.queue_capacity is None
    assert RuntimeMetricsStatus((first, second)).queue_capacity == 12
    assert RuntimeMetricsStatus((first, _status(enabled=False, accepting=False))).accepting


@pytest.mark.asyncio
async def test_empty_aggregate_is_disabled_and_settled() -> None:
    control = RuntimeMetricsControl(lambda: ())
    status = control.status()
    assert status.recorders == ()
    assert not status.enabled and not status.accepting
    assert status.accepted == status.pending == status.queue_capacity == 0
    assert status.failure_codes == ()
    assert (await control.flush(timeout_seconds=0)).completed


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
async def test_empty_aggregate_and_disabled_producer_validate_timeout(timeout: float) -> None:
    with pytest.raises(AIError):
        await RuntimeMetricsControl(lambda: ()).flush(timeout_seconds=timeout)
    async with Persistence.open("metrics-disabled", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(persistence=persistence) as engine:
            assert not engine.metrics.status().enabled
            with pytest.raises(AIError):
                await engine.metrics.flush(timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_aggregate_flush_snapshots_owners_and_shares_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [10.0]
    monkeypatch.setattr("linktools.ai.observe._control.monotonic", lambda: clock[0])
    controls = []
    budgets = []
    settled = _status(unconfirmed=1, last_failure_code="WRITE_FAILED")

    async def first_flush(*, timeout_seconds: float) -> MetricFlushResult:
        budgets.append(timeout_seconds)
        clock[0] += 0.75
        controls.clear()
        return MetricFlushResult(True, settled)

    async def second_flush(*, timeout_seconds: float) -> MetricFlushResult:
        budgets.append(timeout_seconds)
        return MetricFlushResult(False, _status(pending=1))

    controls.extend((MetricsControl(lambda: settled, first_flush), MetricsControl(_status, second_flush)))
    result = await RuntimeMetricsControl(lambda: controls).flush(timeout_seconds=1.0)
    assert budgets == [1.0, 0.25]
    assert not result.completed
    assert result.status.recorders == (settled, _status(pending=1))
    assert result.status.unconfirmed == 1
    single = RuntimeMetricsControl(lambda: (MetricsControl(lambda: settled, first_flush),))
    assert (await single.flush()).completed
    assert single.status().unconfirmed == 1


@pytest.mark.asyncio
async def test_aggregate_does_not_turn_flush_exception_into_success() -> None:
    async def fail(*, timeout_seconds: float) -> MetricFlushResult:
        raise OSError("store unavailable")

    with pytest.raises(OSError, match="store unavailable"):
        await RuntimeMetricsControl(lambda: (MetricsControl(_status, fail),)).flush()


@pytest.mark.asyncio
async def test_closed_producer_keeps_cumulative_status_without_owning_dataset() -> None:
    dataset = Metrics.in_memory(commit=CommitPolicy.batch(
        buffer=InMemoryCacheBackend(max_bytes=2**20), batch_size=128, flush_interval=60,
    ))
    first, borrowed = MetricsService(dataset), MetricsService(dataset)
    control = MetricsControl(first.status, first.flush)
    aggregate = RuntimeMetricsControl(lambda: (control,))
    observation = Observation(
        version=1, observation_id="owned", kind="sample", occurred_at=datetime.now(timezone.utc),
        source_namespace="metrics", tenant_id=None, status=None, error_code=None,
        correlation={}, dimensions={}, measurements=(),
    )
    try:
        assert await first.try_record(observation)
        assert await borrowed.try_record(replace(observation, observation_id="borrowed"))
        assert aggregate.status().pending == 1
        assert (await aggregate.flush()).completed
        assert borrowed.status().pending == 1
        await first.close()
        assert aggregate.status().persisted == 1
        assert not aggregate.status().accepting
        assert (await aggregate.flush()).status.persisted == 1
        assert borrowed.status().accepting
    finally:
        await first.close()
        await borrowed.close()


@pytest.mark.asyncio
async def test_task_engine_views_share_public_metrics_control() -> None:
    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return context.input["value"]

    task = Task("metrics.echo", echo, effect_policy="none")
    async with Persistence.open("metrics-view", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(persistence=persistence, metrics=Metrics.in_memory()) as engine:
            view = engine.with_definitions(task)
            assert view.metrics is engine.metrics
            assert view.metrics.status().enabled and view.metrics.status().accepting
            run = await view.start(TaskGraph("metrics-graph", (
                TaskNode("echo", task=task, input={"value": 1}),
            )), idempotency_key="metrics-graph")
            assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
            flushed = await view.metrics.flush()
            assert flushed.completed and flushed.status.accepted >= 2
        assert not view.metrics.status().accepting
        assert (await view.metrics.flush()).status.accepted == flushed.status.accepted
