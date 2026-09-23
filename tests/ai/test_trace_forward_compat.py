#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward-compatibility coverage for public execution trace projections."""

from linktools.ai.runtime.service_api import ExecutionTraceItem


def _usage(**extra: object) -> dict[str, object]:
    return {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_tokens": 3,
        "cache_write_tokens": 4,
        **extra,
    }


def test_trace_model_response_allows_additive_usage_fields() -> None:
    payload = {
        "kind": "MODEL_RESPONSE",
        "status": "SUCCEEDED",
        "token_usage": _usage(reasoning_tokens=5),
    }

    item = ExecutionTraceItem("execution", 1, payload)

    assert item.payload == payload


def test_trace_dto_does_not_revalidate_usage_payload() -> None:
    payload = {
        "kind": "MODEL_RESPONSE",
        "status": "SUCCEEDED",
        "token_usage": _usage(input_tokens="bad"),
    }

    item = ExecutionTraceItem("execution", 1, payload)

    assert item.payload == payload


def test_trace_model_response_preserves_unknown_future_status() -> None:
    payload = {
        "kind": "MODEL_RESPONSE",
        "status": "PARTIAL",
        "future_payload": {"revision": 2},
    }

    item = ExecutionTraceItem("execution", 1, payload)

    assert item.payload == payload
