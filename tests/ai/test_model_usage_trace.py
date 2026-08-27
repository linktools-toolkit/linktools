#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-request model usage projection into Runtime trace."""

from types import SimpleNamespace

import pytest
from linktools.ai.runtime import ExecutionTraceItem
from linktools.ai.runtime._capabilities import (
    ToolOperationDecision,
    _RuntimeStepPersistence,
    _model_usage_metadata,
)
from linktools.ai.runtime._history import _trace_item
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepEvent


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


async def _text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del messages, info
    return ModelResponse(parts=[TextPart("done")])


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
            run_id="run",
            kind="model_request_completed",
            step_index=1,
            metadata=_model_usage_metadata(ModelResponse(parts=(), usage=usage)),
        )
    )


def _persistence(store: InMemoryStepStore, run_id: str) -> _RuntimeStepPersistence:
    return _RuntimeStepPersistence(
        store=store,
        agent_name="usage-test",
        run_id=run_id,
        tool_operations=_ToolOperations(),
    )


async def _completed_usage(store: InMemoryStepStore, run_id: str) -> list[dict[str, object]]:
    events = await store.list_events(run_id=run_id)
    values = [
        _project_event(event, ordinal)
        for ordinal, event in enumerate(events)
        if event.kind == "model_request_completed"
    ]
    assert all(value is not None for value in values)
    return [value for value in values if value is not None]


def _assert_token_sum(values: list[dict[str, object]], usage: RunUsage) -> None:
    assert sum(int(value["input_tokens"]) for value in values) == usage.input_tokens
    assert sum(int(value["output_tokens"]) for value in values) == usage.output_tokens
    assert sum(int(value["cache_read_tokens"]) for value in values) == usage.cache_read_tokens
    assert sum(int(value["cache_write_tokens"]) for value in values) == usage.cache_write_tokens


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
    assert total.input_tokens == first_trace["input_tokens"] + second_trace["input_tokens"]
    assert total.output_tokens == first_trace["output_tokens"] + second_trace["output_tokens"]
    assert total.cache_read_tokens == first_trace["cache_read_tokens"] + second_trace["cache_read_tokens"]
    assert total.cache_write_tokens == first_trace["cache_write_tokens"] + second_trace["cache_write_tokens"]


def test_legacy_model_response_trace_without_usage_returns_null() -> None:
    event = StepEvent(
        run_id="run",
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

    cached = ExecutionTraceItem(
        "execution",
        1,
        {"kind": "MODEL_RESPONSE", "status": "SUCCEEDED"},
    )
    assert cached.payload["token_usage"] is None


def test_failed_model_response_trace_has_no_request_usage() -> None:
    event = StepEvent(
        run_id="run",
        kind="model_request_failed",
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


@pytest.mark.asyncio
async def test_model_retry_records_each_request_usage() -> None:
    store = InMemoryStepStore()
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
    store = InMemoryStepStore()
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
    store = InMemoryStepStore()
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
