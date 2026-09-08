#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model Metric observation identity and durable Trace linkage regressions."""

from types import SimpleNamespace

import pytest
from linktools.ai.observe import Observation
from linktools.ai.runtime._capabilities import (
    ToolOperationDecision,
    _RuntimeStepPersistence,
)
from linktools.ai.runtime._history import _trace_item
from linktools.ai.runtime._metric_capability import _RuntimeModelMetricCapability
from linktools.ai.runtime._metric_id import _model_observation_id
from linktools.ai.runtime._tool_metrics import _ToolMetricContext
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import InMemoryStepStore


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _ToolOperations:
    async def begin(self, ctx, call, tool_def, args, replay_safe):
        del ctx, tool_def, args
        return ToolOperationDecision(
            operation_id=f"operation:{call.tool_call_id}",
            owner="owner",
            fence=1,
            replay_safe=replay_safe,
        )

    async def renew(self, decision):
        return decision

    async def complete(self, decision, result):
        del decision, result
        return False

    async def fail(self, decision, error):
        del decision, error
        return False

    async def unknown(self, decision, error):
        del decision, error
        raise AssertionError("unexpected unknown tool effect")


def _persistence(
    store: InMemoryStepStore,
    run_id: str,
    recorder: _Recorder | None,
) -> _RuntimeStepPersistence:
    tool_metrics = (
        None
        if recorder is None
        else _ToolMetricContext(
            recorder,
            source_namespace="workspace",
            tenant_id="tenant",
            execution_id="execution",
            session_id=None,
            step_run_id=run_id,
            agent_id="agent",
        )
    )
    return _RuntimeStepPersistence(
        store=store,
        agent_name="agent",
        run_id=run_id,
        tool_operations=_ToolOperations(),
        tool_metrics=tool_metrics,
    )


def _model_metrics(recorder: _Recorder, run_id: str) -> _RuntimeModelMetricCapability:
    return _RuntimeModelMetricCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id=run_id,
        agent_id="agent",
        provider="test",
        model_identity="test:test",
        route_id="default",
    )


def _trace(event) -> object:
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    return item


@pytest.mark.asyncio
async def test_model_metric_and_trace_share_observation_id_and_duration() -> None:
    store = InMemoryStepStore()
    recorder = _Recorder()
    run_id = "metric-trace-run"
    agent = Agent(
        TestModel(custom_output_text="done"),
        deps_type=object,
        capabilities=[_model_metrics(recorder, run_id), _persistence(store, run_id, recorder)],
    )

    await agent.run("hello", deps=SimpleNamespace(correlation={}))

    model_observations = [
        value for value in recorder.observations if value.kind == "linktools.model.request"
    ]
    assert len(model_observations) == 1
    completed = [
        event
        for event in await store.list_events(run_id=run_id)
        if event.kind == "model_request_completed"
    ]
    assert len(completed) == 1
    event = completed[0]
    expected = _model_observation_id(run_id, event.step_index)
    assert model_observations[0].observation_id == expected
    assert event.metadata["linktools.ai.observation_id"] == expected
    assert int(event.metadata["linktools.ai.duration_ns"]) >= 0
    trace = _trace(event)
    assert trace.payload["observation_id"] == expected
    assert trace.payload["duration_ns"] == int(event.metadata["linktools.ai.duration_ns"])


@pytest.mark.asyncio
async def test_failed_model_metric_and_trace_share_observation_id_and_duration() -> None:
    async def fail_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        raise RuntimeError("boom")

    store = InMemoryStepStore()
    recorder = _Recorder()
    run_id = "metric-trace-failed-run"
    agent = Agent(
        FunctionModel(fail_model),
        deps_type=object,
        capabilities=[_model_metrics(recorder, run_id), _persistence(store, run_id, recorder)],
    )

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("hello", deps=SimpleNamespace(correlation={}))

    model_observations = [
        value for value in recorder.observations if value.kind == "linktools.model.request"
    ]
    assert len(model_observations) == 1
    failed = [
        event
        for event in await store.list_events(run_id=run_id)
        if event.kind == "model_request_failed"
    ]
    assert len(failed) == 1
    event = failed[0]
    expected = _model_observation_id(run_id, event.step_index)
    assert model_observations[0].observation_id == expected
    assert event.metadata["linktools.ai.observation_id"] == expected
    assert int(event.metadata["linktools.ai.duration_ns"]) >= 0
    trace = _trace(event)
    assert trace.payload["observation_id"] == expected
    assert trace.payload["duration_ns"] == int(event.metadata["linktools.ai.duration_ns"])


@pytest.mark.asyncio
async def test_output_retry_metric_lineage_uses_pydantic_retry_state() -> None:
    store = InMemoryStepStore()
    recorder = _Recorder()
    run_id = "output-retry-metric-run"
    agent = Agent(
        TestModel(custom_output_text="done"),
        deps_type=object,
        capabilities=[_model_metrics(recorder, run_id), _persistence(store, run_id, recorder)],
        retries={"output": 1},
    )

    @agent.output_validator
    def retry_once(ctx: RunContext[object], output: str) -> str:
        if ctx.retry == 0:
            raise ModelRetry("retry output")
        return output

    result = await agent.run("hello", deps=SimpleNamespace(correlation={}))

    assert result.output == "done"
    model_observations = [
        value for value in recorder.observations if value.kind == "linktools.model.request"
    ]
    assert len(model_observations) == 2
    assert "linktools.output_retry_index" not in model_observations[0].correlation
    assert model_observations[1].correlation["linktools.output_retry_index"] == 1


@pytest.mark.asyncio
async def test_model_trace_omits_metric_metadata_when_metrics_disabled() -> None:
    store = InMemoryStepStore()
    run_id = "no-metrics-run"
    agent = Agent(
        TestModel(custom_output_text="done"),
        capabilities=[_persistence(store, run_id, None)],
    )

    await agent.run("hello")

    completed = [
        event
        for event in await store.list_events(run_id=run_id)
        if event.kind == "model_request_completed"
    ]
    assert len(completed) == 1
    assert "linktools.ai.observation_id" not in completed[0].metadata
    assert "linktools.ai.duration_ns" not in completed[0].metadata
