#!/usr/bin/env python3
"""Streaming observations remain cumulative across requests and interruption."""

from linktools.ai.execution.snapshots import (
    ModelRequestUsageObservation,
    RequestUsage,
    RunUsageCapture,
)
from linktools.ai.execution.domain import RunUsage


def test_two_streaming_rounds_do_not_replace_cumulative_usage_with_request_delta():
    capture = RunUsageCapture()
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="stream-1",
            usage=RequestUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="stream-2",
            usage=RequestUsage(input_tokens=3, output_tokens=1, total_tokens=4),
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )

    assert capture.snapshot() == RunUsage(input_tokens=13, output_tokens=3, total_tokens=16)


def test_interrupted_stream_observes_prior_plus_request_local_usage():
    request = RequestUsage(input_tokens=3, output_tokens=1, total_tokens=4)
    capture = RunUsageCapture()
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="interrupted-stream",
            usage=request,
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )

    assert capture.snapshot().input_tokens == 3
    assert capture.snapshot().output_tokens == 1
