#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-layer regressions for Runtime context and observability closure."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.core import Page, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics, Observation
from linktools.ai.runtime import Runtime, RuntimeContext
from linktools.ai.runtime import _metrics as runtime_metrics
from linktools.ai.runtime._execution import _overlay_execution_context
from linktools.ai.runtime._history import _trace_item
from linktools.ai.task import TaskEvent, TaskEventType
from linktools.ai.task._metrics import _TaskMetricProjector
from pydantic_ai_harness.step_persistence import StepEvent


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _BlockingMetricStore:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.items: dict[str, Observation] = {}

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        del namespace
        self.entered.set()
        await self.release.wait()
        for observation in observations:
            self.items[observation.observation_id] = observation


class _TaskEvents:
    def __init__(self) -> None:
        now = datetime(2026, 9, 6, tzinfo=timezone.utc)
        self.events = (
            TaskEvent(
                version=1,
                graph_id="graph",
                sequence=1,
                event_type=TaskEventType.GRAPH_ADMITTED,
                occurred_at=now,
                status=TaskStatus.PENDING,
            ),
            TaskEvent(
                version=1,
                graph_id="graph",
                sequence=2,
                event_type=TaskEventType.GRAPH_CHANGED,
                occurred_at=now + timedelta(seconds=2),
                status=TaskStatus.SUCCEEDED,
                previous_status=TaskStatus.PENDING,
            ),
        )

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        values = tuple(event for event in self.events if event.sequence > after_sequence)
        items = values[:limit]
        return Page(items, "more" if len(values) > limit else None)

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        return self.events[-1]


class _TaskAdmissions:
    async def get(self, graph_id: str, *, tenant_id: str) -> object:
        assert graph_id == "graph"
        assert tenant_id == "tenant"
        return SimpleNamespace(context={"audit_run_id": "audit-1", "stage": "analysis"})


def _observation(observation_id: str) -> Observation:
    return Observation(
        version=1,
        observation_id=observation_id,
        kind="test.observation",
        occurred_at=datetime.now(timezone.utc),
        source_namespace="workspace",
        tenant_id="tenant",
        status="SUCCEEDED",
        error_code=None,
        correlation={},
        dimensions={},
        measurements=(),
    )


def test_runtime_context_overlay_is_explicit_and_bounded() -> None:
    root = RuntimeContext(
        object(),
        {"audit_run_id": "audit-1", "stage": "collect"},
    )

    effective = root.overlay({"stage": "analyze", "event_id": "event-1"})

    assert effective.app is root.app
    assert dict(root.values) == {"audit_run_id": "audit-1", "stage": "collect"}
    assert dict(effective.values) == {
        "audit_run_id": "audit-1",
        "stage": "analyze",
        "event_id": "event-1",
    }


def test_retry_and_fork_context_inherit_source_then_overlay() -> None:
    source = {"audit_run_id": "audit-1", "stage": "collect"}

    inherited = _overlay_execution_context(source, {})
    overridden = _overlay_execution_context(source, {"stage": "analyze"})

    assert dict(inherited) == source
    assert dict(overridden) == {"audit_run_id": "audit-1", "stage": "analyze"}


def test_retry_and_fork_context_reject_invalid_overlay() -> None:
    with pytest.raises(AIError) as raised:
        _overlay_execution_context(
            {"audit_run_id": "audit-1"},
            {f"key_{index}": str(index) for index in range(8)},
        )
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_disabled_runtime_metric_control_has_stable_public_contract() -> None:
    runtime = Runtime(
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        workspace=object(),
        context=RuntimeContext(None),
    )

    status = runtime.metric_status()
    assert status.enabled is False
    assert status.accepting is False
    assert status.pending == 0
    flushed = await runtime.flush_metrics(timeout_seconds=0)
    assert flushed.completed is True
    assert flushed.status == status
    with pytest.raises(AIError) as raised:
        await runtime.flush_metrics(timeout_seconds=-1)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_runtime_metric_flush_is_an_acceptance_barrier() -> None:
    store = _BlockingMetricStore()
    buffer = runtime_metrics._RuntimeMetricBuffer(
        Metrics.from_store(store, namespace="barrier")  # type: ignore[arg-type]
    )
    assert buffer.try_record(_observation("barrier-observation")) is True
    await asyncio.wait_for(store.entered.wait(), timeout=1)

    pending = await buffer.flush(timeout_seconds=0)
    assert pending.completed is False
    assert pending.status.accepted == 1
    assert pending.status.persisted == 0
    assert pending.status.pending == 1
    assert pending.status.lost == 0

    store.release.set()
    flushed = await buffer.flush(timeout_seconds=1)
    assert flushed.completed is True
    assert flushed.status.accepted == 1
    assert flushed.status.persisted == 1
    assert flushed.status.pending == 0
    await buffer.close()


@pytest.mark.asyncio
async def test_task_metric_projector_joins_durable_admission_context() -> None:
    recorder = _Recorder()
    projector = _TaskMetricProjector(
        _TaskEvents(),  # type: ignore[arg-type]
        recorder,
        source_namespace="workspace",
        admissions=_TaskAdmissions(),  # type: ignore[arg-type]
    )

    await projector._project("graph", tenant_id="tenant")

    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.task.graph.terminal"
    assert observation.correlation["audit_run_id"] == "audit-1"
    assert observation.correlation["stage"] == "analysis"
    assert observation.correlation["linktools.graph_id"] == "graph"


@pytest.mark.asyncio
async def test_trace_observation_id_supports_exact_metric_loss_check() -> None:
    observation_id = runtime_metrics._tool_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        "call-1",
    )
    event = StepEvent(
        run_id="step-run",
        kind="tool_call_completed",
        step_index=3,
        timestamp=datetime(2026, 9, 6, tzinfo=timezone.utc),
        tool_call_id="call-1",
        tool_name="read_file",
        metadata={
            "linktools.ai.observation_id": observation_id,
            "linktools.ai.duration_ns": "1234",
        },
    )
    trace = _trace_item(
        SimpleNamespace(execution_id="execution"),  # type: ignore[arg-type]
        1,
        0,
        1,
        event,
    )
    assert trace is not None
    assert trace.payload["observation_id"] == observation_id
    assert trace.payload["duration_ns"] == 1234
    assert trace.payload["occurred_at"] == "2026-09-06T00:00:00+00:00"

    metrics = Metrics.in_memory(namespace="trace-link")
    assert await metrics.get_observation(observation_id) is None
    await metrics.record_observations(
        (
            Observation(
                version=1,
                observation_id=observation_id,
                kind="linktools.tool.execution",
                occurred_at=datetime.now(timezone.utc),
                source_namespace="workspace",
                tenant_id="tenant",
                status="SUCCEEDED",
                error_code=None,
                correlation={"linktools.execution_id": "execution"},
                dimensions={"agent_id": "agent", "tool_name": "read_file"},
                measurements=(),
            ),
        )
    )
    stored = await metrics.get_observation(observation_id)
    assert stored is not None
    assert stored.observation_id == trace.payload["observation_id"]


def test_model_and_tool_observation_ids_are_replay_stable() -> None:
    model = runtime_metrics._model_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        4,
        1,
    )
    assert model == runtime_metrics._model_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        4,
        1,
    )
    assert model != runtime_metrics._model_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        4,
        2,
    )
    tool = runtime_metrics._tool_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        "call-1",
    )
    assert tool == runtime_metrics._tool_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        "call-1",
    )
    assert tool != runtime_metrics._tool_observation_id(
        "workspace",
        "tenant",
        "execution",
        "step-run",
        "call-2",
    )
