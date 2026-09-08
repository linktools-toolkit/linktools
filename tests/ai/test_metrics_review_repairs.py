#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regressions for metric retention and best-effort recording."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai.messages import ModelResponse

import linktools.ai.runtime._metric_capability as metric_capability
from linktools.ai.core import UsageMetrics
from linktools.ai.observe import MetricMeasurement, MetricQuery, Observation
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._metric_capability import _RuntimeModelMetricCapability
from linktools.ai.runtime._metrics import _BufferedMetricRecorder
from linktools.ai.runtime._tool_metrics import _ToolMetricContext


class _FailingRecorder:
    def record(self, observation: Observation) -> None:
        del observation
        raise RuntimeError("metric sink rejected observation")


class _CaptureLogger:
    def __init__(self) -> None:
        self.exceptions: list[str] = []

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.exceptions.append(message)


class _NullRecorder:
    def record(self, observation: Observation) -> None:
        del observation


class _CaptureRecorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def record(self, observation: Observation) -> None:
        self.observations.append(observation)


class _StatusRecorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def record(self, observation: Observation) -> None:
        self.observations.append(observation)


class _RejectingBuffer:
    def __init__(self) -> None:
        self.items: list[Observation] = []

    def append(self, observation: Observation) -> bool:
        del observation
        return False


class _Sink:
    async def write(self, observations: tuple[Observation, ...]) -> None:
        del observations


class _CapturingSink:
    def __init__(self) -> None:
        self.batches: list[tuple[Observation, ...]] = []

    async def write(self, observations: tuple[Observation, ...]) -> None:
        self.batches.append(observations)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 7, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


async def test_metric_buffer_rejection_does_not_escape() -> None:
    recorder = _BufferedMetricRecorder(
        _RejectingBuffer(),  # type: ignore[arg-type]
        _Sink(),
    )
    observation = Observation(
        version=1,
        observation_id="metric-rejected",
        kind="test.metric",
        occurred_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
    )

    recorder.record(observation)

    status = recorder.status()
    assert status.dropped == 1
    assert status.last_error_code == "METRIC_BUFFER_REJECTED"


async def test_metric_flush_status_tracks_successful_batch() -> None:
    sink = _CapturingSink()
    recorder = _BufferedMetricRecorder(
        [],  # type: ignore[arg-type]
        sink,
    )
    observation = Observation(
        version=1,
        observation_id="metric-success",
        kind="test.metric",
        occurred_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
    )
    recorder._buffer.append(observation)  # type: ignore[union-attr]

    flushed = await recorder.flush()

    assert flushed.written == 1
    assert flushed.pending == 0
    assert sink.batches == [(observation,)]
    assert recorder.status().written == 1


async def test_tool_metric_rejection_is_logged_without_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _CaptureLogger()
    monkeypatch.setattr(
        "linktools.ai.runtime._tool_metrics._logger",
        logger,
    )
    context = _ToolMetricContext(
        recorder=_FailingRecorder(),
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="step-run",
        agent_id="agent",
    )

    context.record(
        tool_name="read_file",
        tool_class="filesystem.read",
        operation_id="operation",
        status="SUCCEEDED",
        started_ns=1,
        finished_ns=2,
        replayed=False,
    )

    assert logger.exceptions == ["tool metric observation rejected"]


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

    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step-run",
    )
    capability._journal = journal
    fact = journal.begin(0)
    fact = journal.finish(fact.request_sequence, status="SUCCEEDED")
    capability._record_model(
        None,
        fact,
        status="SUCCEEDED",
        error_code=None,
        measurements=(),
    )

    assert logger.exceptions == ["model metric observation rejected"]


async def test_non_percentile_accumulators_do_not_retain_samples() -> None:
    at = datetime(2026, 9, 7, tzinfo=timezone.utc)
    query = MetricQuery(
        source_namespace="workspace",
        tenant_id="tenant",
        kinds=("test.metric",),
        start_at=at - timedelta(minutes=1),
        end_at=at + timedelta(minutes=1),
        measurement="latency_ns",
        aggregation="sum",
    )
    from linktools.ai.observe._memory import MemoryMetricStore

    store = MemoryMetricStore(max_observations=32)
    await store.write(
        tuple(
            Observation(
                version=1,
                observation_id=f"metric-{index}",
                kind="test.metric",
                occurred_at=at,
                source_namespace="workspace",
                tenant_id="tenant",
                measurements=(MetricMeasurement("latency_ns", 1, index + 1),),
            )
            for index in range(4)
        )
    )

    result = await store.query(query)

    assert result.rows[0].value == 10


async def test_model_metric_external_observer_preserves_usage_measurements() -> None:
    recorder = _CaptureRecorder()
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
        external_requests=True,
    )
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step-run",
    )
    fact = journal.begin(0)
    response = ModelResponse()

    await capability.record_external_model_request(
        None,  # type: ignore[arg-type]
        fact,
        "completed",
        response,
        None,
    )

    assert recorder.observations[0].observation_id == fact.observation_id


async def test_tool_metric_context_uses_passed_usage() -> None:
    recorder = _StatusRecorder()
    context = _ToolMetricContext(
        recorder=recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="step-run",
        agent_id="agent",
    )

    context.record(
        tool_name="read_file",
        tool_class="filesystem.read",
        operation_id="operation",
        status="SUCCEEDED",
        started_ns=1,
        finished_ns=2,
        replayed=False,
        usage=UsageMetrics(input_tokens=1, output_tokens=2, total_tokens=3),
    )

    observation = recorder.observations[0]
    assert {measurement.name: measurement.value for measurement in observation.measurements}[
        "total_tokens"
    ] == 3
