#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-request model usage projection into Runtime trace."""

from types import SimpleNamespace

from linktools.ai.adapter._history import _trace_item
from linktools.ai.agent._capabilities import _model_usage_metadata
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness.step_persistence import StepEvent


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


def _project_usage(usage: RequestUsage) -> dict[str, object] | None:
    event = StepEvent(
        run_id="run",
        kind="model_request_completed",
        step_index=1,
        metadata=_model_usage_metadata(ModelResponse(parts=(), usage=usage)),
    )
    item = _trace_item(
        SimpleNamespace(execution_id="execution"),
        1,
        0,
        0,
        event,
    )
    assert item is not None
    assert item.payload["kind"] == "MODEL_RESPONSE"
    return item.payload["token_usage"]


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
