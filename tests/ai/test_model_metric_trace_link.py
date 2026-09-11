#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model Metric observation identity and durable Trace linkage regressions."""

from types import SimpleNamespace

import pytest
from linktools.ai.observe import Observation
from linktools.ai.runtime._harness import HarnessStepStoreAdapter
from linktools.ai.runtime._capabilities import (
    _RuntimeStepPersistence,
)
from linktools.ai.runtime._metric_capability import RuntimeModelObservationCapability
from linktools.ai.runtime._history import _trace_item
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._metric_id import _model_observation_id
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from linktools.ai.runtime.state._steps import (
    StagingStepStore,
)


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


async def _text_model(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    del messages, info
    return ModelResponse(parts=[TextPart("done")])


def _persistence(
    store: StagingStepStore,
    run_id: str,
    recorder: _Recorder | None,
    journal: ModelRequestJournal,
) -> _RuntimeStepPersistence:
    return _RuntimeStepPersistence(
        store=HarnessStepStoreAdapter(store, execution_id=None),
        agent_name="agent",
        run_id=run_id,
        model_journal=journal,
        model_observation_enabled=recorder is not None,
    )


def _model_metrics(
    recorder: _Recorder | None,
    run_id: str,
    journal: ModelRequestJournal,
) -> RuntimeModelObservationCapability:
    return RuntimeModelObservationCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id=run_id,
        agent_id="agent",
        journal=journal,
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
    store = StagingStepStore()
    recorder = _Recorder()
    run_id = "metric-trace-run"
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id=run_id,
    )
    agent = Agent(
        TestModel(custom_output_text="done"),
        deps_type=object,
        capabilities=[
            _model_metrics(recorder, run_id, journal),
            _persistence(store, run_id, recorder, journal),
        ],
    )

    await agent.run("hello", deps=SimpleNamespace(correlation={}))

    model_observations = [
        value
        for value in recorder.observations
        if value.kind == "linktools.model.request"
    ]
    assert len(model_observations) == 1
    completed = [
        event
        for event in await store.list_events(run_id=run_id)
        if event.kind == "model_request_completed"
    ]
    assert len(completed) == 1
    event = completed[0]
    expected = _model_observation_id(
        "workspace",
        "tenant",
        "execution",
        run_id,
        1,
        "agent",
    )
    assert model_observations[0].observation_id == expected
    assert event.metadata["linktools.ai.observation_id"] == expected
    assert int(event.metadata["linktools.ai.duration_ns"]) >= 0
    trace = _trace(event)
    assert trace.payload["observation_id"] == expected
    assert trace.payload["duration_ns"] == int(
        event.metadata["linktools.ai.duration_ns"]
    )


@pytest.mark.asyncio
async def test_failed_model_metric_and_trace_share_observation_id_and_duration() -> (
    None
):
    async def fail_model(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        del messages, info
        raise RuntimeError("boom")

    store = StagingStepStore()
    recorder = _Recorder()
    run_id = "metric-trace-failed-run"
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id=run_id,
    )
    agent = Agent(
        FunctionModel(fail_model),
        deps_type=object,
        capabilities=[
            _model_metrics(recorder, run_id, journal),
            _persistence(store, run_id, recorder, journal),
        ],
    )

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("hello", deps=SimpleNamespace(correlation={}))

    model_observations = [
        value
        for value in recorder.observations
        if value.kind == "linktools.model.request"
    ]
    assert len(model_observations) == 1
    failed = [
        event
        for event in await store.list_events(run_id=run_id)
        if event.kind == "model_request_failed"
    ]
    assert len(failed) == 1
    event = failed[0]
    expected = _model_observation_id(
        "workspace",
        "tenant",
        "execution",
        run_id,
        1,
        "agent",
    )
    assert model_observations[0].observation_id == expected
    assert event.metadata["linktools.ai.observation_id"] == expected
    assert int(event.metadata["linktools.ai.duration_ns"]) >= 0
    trace = _trace(event)
    assert trace.payload["observation_id"] == expected
    assert trace.payload["duration_ns"] == int(
        event.metadata["linktools.ai.duration_ns"]
    )


@pytest.mark.asyncio
async def test_output_retry_metric_lineage_uses_pydantic_retry_state() -> None:
    store = StagingStepStore()
    recorder = _Recorder()
    run_id = "output-retry-metric-run"
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id=run_id,
    )
    agent = Agent(
        FunctionModel(_text_model),
        deps_type=object,
        capabilities=[
            _model_metrics(recorder, run_id, journal),
            _persistence(store, run_id, recorder, journal),
        ],
        retries={"output": 2},
    )

    @agent.output_validator
    def retry_twice(ctx: RunContext[object], output: str) -> str:
        if ctx.retry < 2:
            raise ModelRetry("retry output")
        return output

    result = await agent.run("hello", deps=SimpleNamespace(correlation={}))

    assert result.output == "done"
    model_observations = [
        value for value in recorder.observations if value.kind == "linktools.model.request"
    ]
    assert len(model_observations) == 3
    assert len({value.observation_id for value in model_observations}) == 3
    assert "linktools.output_retry_index" not in model_observations[0].correlation
    assert model_observations[1].correlation["linktools.output_retry_index"] == 1
    assert model_observations[2].correlation["linktools.output_retry_index"] == 2

    events = await store.list_events(run_id=run_id)
    started = [event for event in events if event.kind == "model_request_started"]
    completed = [event for event in events if event.kind == "model_request_completed"]
    assert [
        event.metadata.get("linktools.ai.output_retry_index") for event in started
    ] == [None, "1", "2"]
    assert [
        event.metadata.get("linktools.ai.output_retry_index") for event in completed
    ] == [None, "1", "2"]
    assert [
        _trace(event).payload.get("output_retry_index") for event in started
    ] == [None, 1, 2]
    assert [
        _trace(event).payload.get("output_retry_index") for event in completed
    ] == [None, 1, 2]
    assert [event.metadata["linktools.ai.observation_id"] for event in completed] == [
        value.observation_id for value in model_observations
    ]


@pytest.mark.asyncio
async def test_output_retry_trace_lineage_does_not_require_metrics() -> None:
    store = StagingStepStore()
    run_id = "output-retry-no-metrics-run"
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id=run_id,
    )
    agent = Agent(
        FunctionModel(_text_model),
        capabilities=[
            _model_metrics(None, run_id, journal),
            _persistence(store, run_id, None, journal),
        ],
        retries={"output": 1},
    )

    @agent.output_validator
    def retry_once(ctx: RunContext[None], output: str) -> str:
        if ctx.retry == 0:
            raise ModelRetry("retry output")
        return output

    result = await agent.run("hello")

    assert result.output == "done"
    events = await store.list_events(run_id=run_id)
    completed = [event for event in events if event.kind == "model_request_completed"]
    assert [
        event.metadata.get("linktools.ai.output_retry_index") for event in completed
    ] == [None, "1"]
    assert [
        _trace(event).payload.get("output_retry_index") for event in completed
    ] == [None, 1]
    assert all("linktools.ai.observation_id" not in event.metadata for event in completed)


@pytest.mark.asyncio
async def test_model_trace_omits_metric_metadata_when_metrics_disabled() -> None:
    store = StagingStepStore()
    run_id = "no-metrics-run"
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id=run_id,
    )
    agent = Agent(
        TestModel(custom_output_text="done"),
        capabilities=[
            _model_metrics(None, run_id, journal),
            _persistence(store, run_id, None, journal),
        ],
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
    assert "linktools.ai.output_retry_index" not in completed[0].metadata
