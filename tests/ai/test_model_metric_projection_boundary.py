#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional metric projection cannot become the model request authority."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from linktools.ai.core import ExecutionEventType, JsonValue
from linktools.ai.observe import Observation
from linktools.ai.observe.journal import ModelRequestFact, ModelRequestJournal
import linktools.ai.runtime._metric_capability as boundary
from .test_model_request_lifecycle import _LifecycleRecorder, _context


class _Recorder:
    def __init__(self, mode: str, order: list[str]) -> None:
        self.mode = mode
        self.order = order
        self.observations: list[Observation] = []

    async def try_record(self, observation: Observation) -> bool:
        self.order.append("record")
        self.observations.append(observation)
        if self.mode == "error":
            raise RuntimeError("metric backend unavailable")
        if self.mode == "cancel":
            raise asyncio.CancelledError
        return self.mode != "false"


class _History(_LifecycleRecorder):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order
        self.includes: list[bool] = []
        self.completed: ModelRequestFact | None = None

    async def record_model_event(
        self, fact: ModelRequestFact, *, phase: str,
        response: ModelResponse | None = None, error_code: str | None = None,
        include_observation: bool, usage: RequestUsage | None = None,
    ) -> None:
        self.includes.append(include_observation)
        if phase == "completed":
            self.order.append("history")
            self.completed = fact
        await super().record_model_event(
            fact, phase=phase, response=response, error_code=error_code,
            include_observation=include_observation, usage=usage,
        )


class _Budget:
    def __init__(self) -> None:
        self.identities: list[str] = []

    async def run_model(
        self, identity: str, call: Callable[[], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.identities.append(identity)
        return await call()


@pytest.mark.asyncio
@pytest.mark.parametrize("external", (False, True), ids=("ordinary", "external-compaction"))
@pytest.mark.parametrize("mode", ("none", "accept", "false", "error", "cancel", "capture-reject"))
async def test_projection_preserves_request_authority_and_handoff_order(
    monkeypatch: pytest.MonkeyPatch, external: bool, mode: str,
) -> None:
    order: list[str] = []
    metric = None if mode == "none" else _Recorder(mode, order)
    history = _History(order)
    journal = ModelRequestJournal(
        source_namespace="source", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    budget = _Budget()
    events: list[JsonValue] = []

    async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
        if event_type is ExecutionEventType.MODEL_REQUEST_FINISHED:
            order.append("event")
            events.append(payload)

    capability = boundary.ModelObservationCapability(
        metric, source_namespace="source", tenant_id="tenant", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="agent", journal=journal,
        interaction_recorder=history, event_sink=publish, budget=budget,
    )
    assert capability._metrics.enabled is (metric is not None)
    original_capture = boundary._ModelMetricProjection.capture

    def capture(self: boundary._ModelMetricProjection, *args: object, **kwargs: object) -> Observation | None:
        order.append("capture")
        return original_capture(self, *args, **kwargs)

    monkeypatch.setattr(boundary._ModelMetricProjection, "capture", capture)
    if mode == "capture-reject":
        def reject_observation(**kwargs: object) -> Observation:
            raise ValueError("invalid optional metric projection")
        monkeypatch.setattr(boundary, "Observation", reject_observation)

    model = TestModel()
    context, request_context = _context(model)
    response = ModelResponse(
        parts=[TextPart("private answer")],
        usage=RequestUsage(input_tokens=11, output_tokens=7, cache_read_tokens=3),
    )

    async def run() -> None:
        if external:
            fact = journal.begin(1, purpose="compaction")
            await capability.record_external_model_request(
                context, fact, "started", model, None, None, [], None, ModelRequestParameters(),
            )
            finished = journal.finish(fact.model_request_seq, status="SUCCEEDED", duration_ns=12345)
            await capability.record_external_model_request(
                context, finished, "completed", model, response, None, [], None, ModelRequestParameters(),
            )
        else:
            async def handler(request: ModelRequestContext) -> ModelResponse:
                return response
            assert await capability.wrap_model_request(
                context, request_context=request_context, handler=handler,
            ) is response

    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run()
    else:
        await run()

    assert order == ["capture", "history", "event"] + (
        [] if mode in {"none", "capture-reject"} else ["record"]
    )
    assert history.includes == [metric is not None, metric is not None]
    fact = history.completed
    assert fact is not None and fact.status == "SUCCEEDED"
    assert history.finish_calls == {1: 1}
    usage = history.values[1]["usage"]
    assert usage.input_tokens == 11 and usage.output_tokens == 7 and usage.cache_read_tokens == 3
    assert len(events) == 1
    assert events[0]["status"] == "SUCCEEDED"
    assert events[0]["model_request_seq"] == fact.model_request_seq
    assert events[0]["duration_ns"] == fact.duration_ns
    if external:
        assert journal.current(1) is fact and fact.duration_ns == 12345
        assert budget.identities == []  # External dispatch is budgeted by its owner, not this callback.
    else:
        assert budget.identities == [fact.observation_id]
    if metric is not None and mode != "capture-reject":
        assert len(metric.observations) == 1
        observation = metric.observations[0]
        assert observation.observation_id == fact.observation_id
        assert observation.status == fact.status
        measurements = {value.name: value.value for value in observation.measurements}
        assert measurements == {"latency_ns": fact.duration_ns, "input_tokens": 11,
                                "output_tokens": 7, "total_tokens": 18, "cache_read_tokens": 3}
        assert observation.correlation["linktools.model_request_seq"] == fact.model_request_seq
        assert observation.correlation["linktools.request_purpose"] == ("compaction" if external else "agent")
        assert "private answer" not in repr(observation)
    elif metric is not None:
        assert metric.observations == []
    assert capability._metrics.enabled is (metric is not None)


def test_projection_usage_conversion_does_not_mutate_authoritative_usage() -> None:
    usage = RequestUsage(input_tokens=8, output_tokens=0, cache_write_tokens=5)
    measurements = boundary._ModelMetricProjection.usage_measurements(usage)
    assert {item.name: item.value for item in measurements} == {"input_tokens": 8, "cache_write_tokens": 5}
    assert (usage.input_tokens, usage.output_tokens, usage.cache_write_tokens) == (8, 0, 5)
    assert boundary._ModelMetricProjection.usage_measurements(RequestUsage()) == ()
