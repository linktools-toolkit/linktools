#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for the unified Metrics gap closure."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import Page, TaskStatus, UsageMetrics, canonical_sha256
from linktools.ai.observe import Metrics, Observation
from linktools.ai.runtime import _metrics as runtime_metrics
from linktools.ai.task._event import TaskEvent, TaskEventType
from linktools.ai.task._metrics import _TaskMetricProjector


def _measurements(observation: Observation) -> dict[str, int | float]:
    return {item.name: item.value for item in observation.measurements}


def test_execution_terminal_usage_is_complete_and_cumulative() -> None:
    usage = UsageMetrics(
        model_requests=3,
        tool_calls=2,
        input_tokens=100,
        output_tokens=25,
        cache_read_tokens=40,
        cache_write_tokens=10,
    )

    values = {
        item.name: item.value
        for item in runtime_metrics._execution_usage_measurements(usage)
    }

    assert values == {
        "model_requests": 3,
        "tool_calls": 2,
        "input_tokens": 100,
        "output_tokens": 25,
        "cache_read_tokens": 40,
        "cache_write_tokens": 10,
        "total_tokens": 125,
    }


class _BlockingStore:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def put_definition(self, namespace: str, definition: object) -> object:
        del namespace, definition
        raise AssertionError("automatic metrics never define metrics")

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> None:
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
        kind="test.health",
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
async def test_metric_buffer_reports_watermark_and_safe_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_metrics, "_QUEUE_CAPACITY", 1)
    store = _BlockingStore()
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="health")  # type: ignore[arg-type]
    )

    assert buffer.try_record(_observation("first")) is True
    await asyncio.wait_for(store.entered.wait(), timeout=1)
    assert buffer.try_record(_observation("second")) is True
    assert buffer.try_record(_observation("third")) is False

    status = buffer.status()
    assert status.queue_size == 1
    assert status.queue_capacity == 1
    assert status.high_watermark == 1
    assert status.last_failure_code == "QUEUE_FULL"
    assert status.rejected == 1

    store.release.set()
    await buffer.close()


class _TaskRepository:
    def __init__(self, events: tuple[TaskEvent, ...]) -> None:
        self.events = events

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]:
        del tenant_id
        selected = tuple(
            event
            for event in self.events
            if event.graph_id == graph_id and event.sequence > after_sequence
        )
        page = selected[:limit]
        return Page(page, "more" if len(selected) > limit else None)

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None:
        del tenant_id
        selected = tuple(event for event in self.events if event.graph_id == graph_id)
        return selected[-1] if selected else None


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


def _node_event(
    graph_id: str,
    sequence: int,
    at: datetime,
    status: TaskStatus,
    previous: TaskStatus,
    *,
    fence: int = 0,
    owner: str | None = None,
    execution_id: str | None = None,
) -> TaskEvent:
    result_digest = (
        canonical_sha256({"sequence": sequence})
        if status is TaskStatus.SUCCEEDED
        else None
    )
    error_code = "NODE_FAILED" if status is TaskStatus.FAILED else None
    error_digest = (
        canonical_sha256({"error": sequence})
        if status is TaskStatus.FAILED
        else None
    )
    return TaskEvent(
        1,
        graph_id,
        sequence,
        TaskEventType.NODE_CHANGED,
        at,
        status,
        previous_status=previous,
        node_id="node",
        owner=owner,
        fence=fence,
        execution_id=execution_id,
        result_digest=result_digest,
        error_code=error_code,
        error_digest=error_digest,
    )


@pytest.mark.asyncio
async def test_task_metrics_derive_timing_attempt_index_and_retry_from_durable_events() -> None:
    start = datetime(2026, 9, 6, tzinfo=timezone.utc)
    graph_id = "graph"
    events = (
        TaskEvent(
            1,
            graph_id,
            1,
            TaskEventType.GRAPH_ADMITTED,
            start,
            TaskStatus.PENDING,
        ),
        _node_event(
            graph_id,
            2,
            start + timedelta(seconds=1),
            TaskStatus.READY,
            TaskStatus.PENDING,
        ),
        _node_event(
            graph_id,
            3,
            start + timedelta(seconds=2),
            TaskStatus.RUNNING,
            TaskStatus.READY,
            fence=1,
            owner="owner",
        ),
        _node_event(
            graph_id,
            4,
            start + timedelta(seconds=5),
            TaskStatus.FAILED,
            TaskStatus.RUNNING,
            fence=1,
            execution_id="execution-1",
        ),
        _node_event(
            graph_id,
            5,
            start + timedelta(seconds=6),
            TaskStatus.READY,
            TaskStatus.FAILED,
        ),
        _node_event(
            graph_id,
            6,
            start + timedelta(seconds=8),
            TaskStatus.RUNNING,
            TaskStatus.READY,
            fence=2,
            owner="owner",
        ),
        _node_event(
            graph_id,
            7,
            start + timedelta(seconds=12),
            TaskStatus.SUCCEEDED,
            TaskStatus.RUNNING,
            fence=2,
            execution_id="execution-2",
        ),
        TaskEvent(
            1,
            graph_id,
            8,
            TaskEventType.GRAPH_CHANGED,
            start + timedelta(seconds=13),
            TaskStatus.SUCCEEDED,
            previous_status=TaskStatus.PENDING,
        ),
    )
    recorder = _Recorder()
    projector = _TaskMetricProjector(
        _TaskRepository(events),
        recorder,
        source_namespace="workspace",
    )

    assert await projector._project(graph_id, tenant_id="tenant") is True

    graph = next(
        item
        for item in recorder.observations
        if item.kind == "linktools.task.graph.terminal"
    )
    assert _measurements(graph) == {
        "latency_ns": 13_000_000_000,
        "retry_count": 1,
    }

    attempts = sorted(
        (
            item
            for item in recorder.observations
            if item.kind == "linktools.task.node.attempt"
        ),
        key=lambda item: int(item.correlation["linktools.attempt_index"]),
    )
    assert len(attempts) == 2
    assert attempts[0].correlation["linktools.attempt_index"] == 1
    assert _measurements(attempts[0]) == {
        "latency_ns": 3_000_000_000,
        "retry_count": 0,
        "queue_wait_ns": 1_000_000_000,
    }
    assert attempts[1].correlation["linktools.attempt_index"] == 2
    assert _measurements(attempts[1]) == {
        "latency_ns": 4_000_000_000,
        "retry_count": 1,
        "queue_wait_ns": 2_000_000_000,
    }
