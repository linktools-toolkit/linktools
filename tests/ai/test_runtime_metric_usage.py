#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model usage measurement availability regressions."""

from linktools.ai.runtime._metric_capability import _provider_usage_measurements
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage


def test_model_usage_omits_unreported_zero_cache_fields() -> None:
    response = ModelResponse(
        parts=(),
        usage=RequestUsage(input_tokens=12, output_tokens=4),
    )

    measurements = _provider_usage_measurements(response)

    assert [(item.name, item.value) for item in measurements] == [
        ("input_tokens", 12),
        ("output_tokens", 4),
    ]


def test_model_usage_keeps_explicit_zero_detail_measurement() -> None:
    response = ModelResponse(
        parts=(),
        usage=RequestUsage(details={"cache_read_tokens": 0}),
    )

    measurements = _provider_usage_measurements(response)

    assert [(item.name, item.value) for item in measurements] == [
        ("cache_read_tokens", 0),
    ]
