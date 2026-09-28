#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-request model usage projection into Runtime trace."""

import asyncio
from types import SimpleNamespace

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._capabilities import (
    _AgentRunPersistenceCapability,
)
from linktools.ai.runtime._history_projection import _trace_item
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage
from linktools.ai.runtime.state._steps import (
    StagingAgentRunStore,
)
from linktools.ai.runtime.state._step_contracts import (
    EventKind,
    StepEvent,
)


async def _text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del messages, info
    return ModelResponse(parts=[TextPart("done")])


async def _usage_model(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    del messages, info
    return ModelResponse(
        parts=[TextPart("done")],
        usage=_response_usage(
            input_tokens=101,
            output_tokens=202,
            cache_read_tokens=303,
            cache_write_tokens=404,
        ),
    )


def _response_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> RequestUsage:
    return RequestUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _project_event(event: StepEvent, ordinal: int = 0) -> dict[str, object] | None:
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        ordinal,
        event,
    )
    assert item is not None
    assert item.payload["kind"] == "MODEL_RESPONSE"
    return item.payload["token_usage"]


def _project_usage(usage: RequestUsage) -> dict[str, object] | None:
    return _project_event(
        StepEvent(
            agent_run_id="run",
            kind="model_request_completed",
            step_index=1,
            metadata=_model_usage_metadata(ModelResponse(parts=(), usage=usage)),
        )
    )


def _model_usage_metadata(response: ModelResponse) -> dict[str, str]:
    usage = response.usage
    return {
        "linktools.ai.model_usage.input_tokens": str(usage.input_tokens),
        "linktools.ai.model_usage.output_tokens": str(usage.output_tokens),
        "linktools.ai.model_usage.cache_read_tokens": str(
            usage.cache_read_tokens
        ),
        "linktools.ai.model_usage.cache_write_tokens": str(
            usage.cache_write_tokens
        ),
    }


def _persistence(
    store: StagingAgentRunStore,
    agent_run_id: str,
    *,
    reverse_registration: bool = False,
) -> CombinedCapability[object]:
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        agent_run_id=agent_run_id,
    )
    run_recorder = AgentRunRecorder(
        store,
        execution_id=None,
        agent_run_id=agent_run_id,
    )
    persistence = _AgentRunPersistenceCapability(
        recorder=run_recorder,
        agent_id="usage-test",
        agent_run_id=agent_run_id,
    )
    observation = ModelObservationCapability(
        None,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        agent_run_id=agent_run_id,
        agent_id="usage-test",
        journal=journal,
        interaction_recorder=run_recorder,
    )
    capabilities = [observation, persistence]
    if reverse_registration:
        capabilities.reverse()
    return CombinedCapability(capabilities)


async def _completed_usage(
    store: StagingAgentRunStore, agent_run_id: str
) -> list[dict[str, object]]:
    events = await store.list_events(agent_run_id=agent_run_id)
    values = [
        _project_event(event, ordinal)
        for ordinal, event in enumerate(events)
        if event.kind == "model_request_completed"
    ]
    assert all(value is not None for value in values)
    return [value for value in values if value is not None]


@pytest.mark.asyncio
async def test_model_usage_trace_does_not_depend_on_registration_order() -> None:
    store = StagingAgentRunStore()
    agent_run_id = "reverse-registration-run"
    agent = Agent(
        FunctionModel(_usage_model),
        capabilities=[
            _persistence(
                store,
                agent_run_id,
                reverse_registration=True,
            )
        ],
    )

    await agent.run("hello")

    events = await store.list_events(agent_run_id=agent_run_id)
    started = [
        event
        for event in events
        if event.kind == "model_request_started"
    ]
    assert len(started) == 1
    assert started[0].metadata["linktools.ai.request_sequence"] == "1"
    assert started[0].metadata["linktools.ai.request_purpose"] == "agent"

    completed = [
        event
        for event in events
        if event.kind == "model_request_completed"
    ]
    assert len(completed) == 1
    assert completed[0].metadata["linktools.ai.request_sequence"] == "1"
    assert completed[0].metadata["linktools.ai.request_purpose"] == "agent"
    assert _project_event(completed[0]) == {
        "input_tokens": 101,
        "output_tokens": 202,
        "cache_read_tokens": 303,
        "cache_write_tokens": 404,
    }


@pytest.mark.asyncio
async def test_asyncio_model_cancellation_preserves_cancelled_status() -> None:
    store = StagingAgentRunStore()

    async def cancelled_model(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        del messages, info
        raise asyncio.CancelledError

    persistence = _persistence(store, "cancelled-model-run")
    agent = Agent(FunctionModel(cancelled_model), capabilities=[persistence])

    with pytest.raises(asyncio.CancelledError):
        await agent.run("hello")

    events = await store.list_events(agent_run_id="cancelled-model-run")
    model_events = [
        event.kind for event in events if event.kind.startswith("model_request_")
    ]
    assert model_events == ["model_request_started", "model_request_cancelled"]
    cancelled = next(
        event for event in events if event.kind == "model_request_cancelled"
    )
    assert cancelled.metadata["linktools.ai.request_sequence"] == "1"
    assert cancelled.metadata["linktools.ai.request_purpose"] == "agent"
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        cancelled,
    )
    assert item is not None
    assert item.payload["status"] == "CANCELLED"
    assert item.payload["token_usage"] is None


def _assert_token_sum(values: list[dict[str, object]], usage: RunUsage) -> None:
    assert sum(int(value["input_tokens"]) for value in values) == usage.input_tokens
    assert sum(int(value["output_tokens"]) for value in values) == usage.output_tokens
    assert (
        sum(int(value["cache_read_tokens"]) for value in values)
        == usage.cache_read_tokens
    )
    assert (
        sum(int(value["cache_write_tokens"]) for value in values)
        == usage.cache_write_tokens
    )


def test_model_response_trace_contains_request_usage() -> None:
    assert _project_usage(
        _response_usage(
            input_tokens=1234,
            output_tokens=256,
            cache_read_tokens=1000,
            cache_write_tokens=0,
        )
    ) == {
        "input_tokens": 1234,
        "output_tokens": 256,
        "cache_read_tokens": 1000,
        "cache_write_tokens": 0,
    }


def test_model_response_trace_defaults_missing_cache_usage_to_zero() -> None:
    assert _project_usage(RequestUsage(input_tokens=100, output_tokens=20)) == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def test_model_response_trace_keeps_each_request_usage_separate() -> None:
    first = _response_usage(
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=80,
        cache_write_tokens=0,
    )
    second = _response_usage(
        input_tokens=140,
        output_tokens=30,
        cache_read_tokens=0,
        cache_write_tokens=40,
    )

    first_trace = _project_usage(first)
    second_trace = _project_usage(second)
    assert first_trace is not None
    assert second_trace is not None
    assert first_trace != second_trace

    total = RunUsage()
    total.incr(first)
    total.incr(second)
    assert (
        total.input_tokens == first_trace["input_tokens"] + second_trace["input_tokens"]
    )
    assert (
        total.output_tokens
        == first_trace["output_tokens"] + second_trace["output_tokens"]
    )
    assert (
        total.cache_read_tokens
        == first_trace["cache_read_tokens"] + second_trace["cache_read_tokens"]
    )
    assert (
        total.cache_write_tokens
        == first_trace["cache_write_tokens"] + second_trace["cache_write_tokens"]
    )


def test_successful_model_response_trace_allows_missing_usage_fact() -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_completed",
        step_index=1,
        metadata={},
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["token_usage"] is None


def test_successful_model_response_trace_allows_partial_usage_fact() -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_completed",
        step_index=1,
        metadata={
            "linktools.ai.model_usage.input_tokens": "10",
            "linktools.ai.model_usage.output_tokens": "2",
        },
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["token_usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
    }


def test_model_response_trace_rejects_invalid_usage_value() -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_completed",
        step_index=1,
        metadata={"linktools.ai.model_usage.input_tokens": "-1"},
    )
    with pytest.raises(AIError) as error:
        _trace_item(
            SimpleNamespace(execution_id="execution"),
            1,
            0,
            0,
            event,
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    "kind",
    ("tool_call_started", "tool_call_completed", "tool_call_failed"),
)
def test_tool_trace_accepts_request_sequence_without_request_purpose(
    kind: EventKind,
) -> None:
    event = StepEvent(
        agent_run_id="run",
        kind=kind,
        step_index=1,
        tool_call_id="call",
        tool_name="lookup",
        metadata={"linktools.ai.request_sequence": "1"},
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["request_sequence"] == 1
    assert "purpose" not in item.payload


def test_model_trace_accepts_sparse_request_lineage() -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_started",
        step_index=1,
        metadata={"linktools.ai.request_sequence": "1"},
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["request_sequence"] == 1
    assert "purpose" not in item.payload


def test_model_trace_preserves_unknown_request_purpose() -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_started",
        step_index=1,
        metadata={"linktools.ai.request_purpose": "future"},
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["purpose"] == "future"


def test_legacy_cancelled_model_response_trace_is_normalized() -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_failed",
        step_index=1,
        error=ErrorCode.EXECUTION_CANCELLED.value,
        metadata={},
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["status"] == "CANCELLED"
    assert item.payload["token_usage"] is None


@pytest.mark.parametrize(
    "error_code",
    (None, ErrorCode.INTERNAL_ERROR.value),
)
def test_failed_model_response_trace_has_no_request_usage(
    error_code: str | None,
) -> None:
    event = StepEvent(
        agent_run_id="run",
        kind="model_request_failed",
        step_index=1,
        error=error_code,
        metadata={},
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["status"] == "FAILED"
    assert item.payload["token_usage"] is None


@pytest.mark.asyncio
async def test_model_retry_records_each_request_usage() -> None:
    store = StagingAgentRunStore()
    agent = Agent(
        FunctionModel(_text_model),
        capabilities=[_persistence(store, "retry-run")],
    )
    attempts = 0

    @agent.output_validator
    def retry_once(output: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelRetry("retry once")
        return output

    result = await agent.run("hello")
    values = await _completed_usage(store, "retry-run")

    assert attempts == 2
    assert len(values) == result.usage.requests == 2
    _assert_token_sum(values, result.usage)


@pytest.mark.asyncio
async def test_tool_loop_records_usage_before_and_after_tool() -> None:
    store = StagingAgentRunStore()
    agent = Agent(
        TestModel(),
        capabilities=[_persistence(store, "tool-run")],
    )

    @agent.tool_plain
    async def echo(text: str) -> str:
        return text

    result = await agent.run("hello")
    values = await _completed_usage(store, "tool-run")

    assert result.usage.tool_calls == 1
    assert len(values) == result.usage.requests == 2
    _assert_token_sum(values, result.usage)


@pytest.mark.asyncio
async def test_streaming_records_completed_request_usage() -> None:
    store = StagingAgentRunStore()
    agent = Agent(
        TestModel(custom_output_text="streamed"),
        capabilities=[_persistence(store, "stream-run")],
    )

    async with agent.run_stream("hello") as result:
        assert await result.get_output() == "streamed"
        usage = result.usage

    values = await _completed_usage(store, "stream-run")
    assert len(values) == usage.requests == 1
    _assert_token_sum(values, usage)
