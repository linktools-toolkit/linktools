#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalized model usage measurement regressions."""

from linktools.ai.runtime._metric_capability import _provider_usage_measurements
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage


def test_model_usage_records_normalized_zero_cache_fields() -> None:
    response = ModelResponse(
        parts=(),
        usage=RequestUsage(input_tokens=12, output_tokens=4),
    )

    measurements = _provider_usage_measurements(response)

    assert [(item.name, item.value) for item in measurements] == [
        ("input_tokens", 12),
        ("output_tokens", 4),
        ("cache_read_tokens", 0),
        ("cache_write_tokens", 0),
    ]


def test_model_usage_records_all_normalized_zero_fields() -> None:
    response = ModelResponse(parts=(), usage=RequestUsage())

    measurements = _provider_usage_measurements(response)

    assert [(item.name, item.value) for item in measurements] == [
        ("input_tokens", 0),
        ("output_tokens", 0),
        ("cache_read_tokens", 0),
        ("cache_write_tokens", 0),
    ]
