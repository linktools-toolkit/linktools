#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Metrics review repairs."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import Page, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import (
    MetricAggregation,
    MetricDefinition,
    MetricMeasurement,
    MetricQuery,
    MetricSource,
    MetricType,
    MetricWindow,
    Metrics,
    Observation,
)
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.observe._query import _Accumulator
from linktools.ai.runtime import _metric_capability as metric_capability
from linktools.ai.runtime._metric_capability import _RuntimeModelMetricCapability
from linktools.ai.task._event import TaskEvent, TaskEventType
from linktools.ai.task._metrics import _TaskMetricProjector
from pydantic_ai.exceptions import RunCancelled

pytestmark = pytest.mark.asyncio


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _FailingRecorder:
    def try_record(self, observation: Observation) -> bool:
        del observation
        raise RuntimeError("synthetic metric rejection")


class _CaptureLogger:
    def __init__(self) -> None:
        self.exceptions: list[str] = []

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.exceptions.append(message)


def _distribution_definition(
    name: str,
    default: MetricAggregation,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=f"{name}.sample",
        source=MetricSource.measurement("value"),
        metric_type=MetricType.DISTRIBUTION,
        unit="1",
        default_aggregation=default,
    )


def _observation(identity: str, *, status: str = "SUCCEEDED") -> Observation:
    return Observation(
        version=1,
        observation_id=identity,
        kind="business.commit.sample",
        occurred_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status=status,
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(MetricMeasurement("value", 1, 1),),
    )


async def test_default_percentile_accepts_query_percentile() -> None:
    metrics = Metrics.in_memory(namespace="default-percentile")
    definition = _distribution_definition(
        "business.default.percentile",
        MetricAggregation.PERCENTILE,
    )
    await metrics.define(definition)
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    for index, value in enumerate((1, 2, 3), start=1):
        await metrics.record(
            definition.name,
            value,
            observation_id=f"sample-{index}",
            occurred_at=start + timedelta(seconds=index),
        )

    result = await metrics.query(
        MetricQuery(
            definition.name,
            MetricWindow.between(start, start + timedelta(minutes=1)),
            percentile=0.5,
        )
    )

    assert result.aggregation is MetricAggregation.PERCENTILE
    assert result.points[0].value == 2
    assert result.points[0].sample_count == 3


async def test_percentile_is_rejected_when_resolved_default_is_not_percentile() -> None:
    metrics = Metrics.in_memory(namespace="invalid-default-percentile")
    definition = _distribution_definition(
        "business.default.mean",
        MetricAggregation.MEAN,
    )
    await metrics.define(definition)
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)

    with pytest.raises(AIError) as raised:
        await metrics.query(
            MetricQuery(
                definition.name,
                MetricWindow.between(start, start + timedelta(minutes=1)),
                percentile=0.95,
            )
        )
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_run_cancelled_records_cancelled_model_observation() -> None:
    recorder = _Recorder()
    capability = _RuntimeModelMetricCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="step-run",
        agent_id="agent",
        provider="test",
        model_identity="test:model",
        route_id="default",
    )

    async def handler(_request: object) -> object:
        raise RunCancelled("cancelled by application")

    with pytest.raises(RunCancelled):
        await capability.wrap_model_request(
            None,
            request_context=object(),  # type: ignore[arg-type]
            handler=handler,  # type: ignore[arg-type]
        )

    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.status == "CANCELLED"
    assert observation.error_code == ErrorCode.EXECUTION_CANCELLED.value


async def test_model_metric_rejection_is_logged_without_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _CaptureLogger()
    monkeypatch.setattr(metric_capability, "_logger", logger)
    capability = _RuntimeModelMetricCapability(
        _FailingRecorder(),
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="step-run",
        agent_id="agent",
        provider="test",
        model_identity="test:model",
        route_id="default",
    )

    capability._record_model(
        None,
        "attempt",
        0,
        status="SUCCEEDED",
        error_code=None,
        measurements=(),
    )

    assert logger.exceptions == ["model metric observation rejected"]


def test_non_percentile_accumulators_do_not_retain_samples() -> None:
    at = datetime(2026, 9, 7, tzinfo=timezone.utc)
    for aggregation in (
        MetricAggregation.COUNT,
        MetricAggregation.SUM,
        MetricAggregation.MEAN,
        MetricAggregation.MIN,
        MetricAggregation.MAX,
        MetricAggregation.LATEST,
        MetricAggregation.RATE,
    ):
        accumulator = _Accumulator(aggregation)
        for index in range(100):
            accumulator.add(
                index,
                occurred_at=at + timedelta(microseconds=index),
                digest=f"{index:064x}" if aggregation is MetricAggregation.LATEST else None,
            )
        assert accumulator.count == 100
        assert accumulator.samples is None

    percentile = _Accumulator(MetricAggregation.PERCENTILE)
    percentile.add(1, occurred_at=at, digest=None)
    assert percentile.samples == [1]


class _DefinitionCommitUnknownStore(InMemoryMetricStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> MetricDefinition:
        self.calls += 1
        stored = await super().put_definition(namespace, definition)
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)


class _ObservationCommitUnknownStore(InMemoryMetricStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        self.calls += 1
        await super().put_observations(namespace, observations)
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)


class _ObservationCommitUnknownTwiceStore(InMemoryMetricStore):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        self.calls += 1
        if self.calls == 1:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        await super().put_observations(namespace, observations)
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)


class _ObservationConflictReadbackStore(InMemoryMetricStore):
    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        conflicting = tuple(replace(item, status="FAILED") for item in observations)
        await super().put_observations(namespace, conflicting)
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)


async def test_definition_commit_unknown_resolves_from_readback_without_replay() -> None:
    store = _DefinitionCommitUnknownStore()
    metrics = Metrics.from_store(store, namespace="definition-readback")
    definition = _distribution_definition(
        "business.definition.readback",
        MetricAggregation.MEAN,
    )

    stored = await metrics.define(definition)

    assert stored == definition
    assert store.calls == 1


async def test_observation_commit_unknown_resolves_from_readback_without_replay() -> None:
    store = _ObservationCommitUnknownStore()
    metrics = Metrics.from_store(store, namespace="observation-readback")

    await metrics.record_observations((_observation("committed"),))

    assert store.calls == 1


async def test_second_commit_unknown_resolves_after_exact_replay_readback() -> None:
    store = _ObservationCommitUnknownTwiceStore()
    metrics = Metrics.from_store(store, namespace="observation-replay")

    await metrics.record_observations((_observation("replayed"),))

    assert store.calls == 2


async def test_commit_unknown_conflicting_readback_fails_closed() -> None:
    metrics = Metrics.from_store(
        _ObservationConflictReadbackStore(),
        namespace="observation-conflict",
    )

    with pytest.raises(AIError) as raised:
        await metrics.record_observations((_observation("conflict"),))
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT


class _TaskRepository:
    def __init__(self, events: tuple[TaskEvent, ...]) -> None:
        self.events = events
        self.list_calls: list[tuple[int, int]] = []

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]:
        del tenant_id
        self.list_calls.append((after_sequence, limit))
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


async def test_task_projection_reads_admission_only_as_part_of_history_scan() -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    events = (
        TaskEvent(
            1,
            "graph",
            1,
            TaskEventType.GRAPH_ADMITTED,
            start,
            TaskStatus.PENDING,
        ),
        TaskEvent(
            1,
            "graph",
            2,
            TaskEventType.GRAPH_CHANGED,
            start + timedelta(seconds=1),
            TaskStatus.SUCCEEDED,
            previous_status=TaskStatus.PENDING,
        ),
    )
    repository = _TaskRepository(events)
    recorder = _Recorder()
    projector = _TaskMetricProjector(
        repository,
        recorder,
        source_namespace="workspace",
    )

    assert await projector._project("graph", tenant_id="tenant") is True

    assert repository.list_calls == [(0, 1000)]
    assert [item.kind for item in recorder.observations] == [
        "linktools.task.graph.terminal"
    ]
