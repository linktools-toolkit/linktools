#!/usr/bin/env python3
"""Streaming observations remain cumulative across requests and interruption."""

from linktools.ai.execution.domain import RunUsage
from linktools.ai.execution.snapshots import (
    ModelUsageObservation,
    RequestUsage,
    RunUsageCapture,
)


def test_two_streaming_rounds_do_not_replace_cumulative_usage_with_request_delta():
    capture = RunUsageCapture()
    capture.observe(
        ModelUsageObservation(
            request_usage=RequestUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            cumulative_usage=RunUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            resolved_model_id=None,
        )
    )
    capture.observe(
        ModelUsageObservation(
            request_usage=RequestUsage(input_tokens=3, output_tokens=1, total_tokens=4),
            cumulative_usage=RunUsage(input_tokens=13, output_tokens=3, total_tokens=16),
            resolved_model_id=None,
        )
    )

    assert capture.snapshot() == RunUsage(input_tokens=13, output_tokens=3, total_tokens=16)


def test_interrupted_stream_observes_prior_plus_request_local_usage():
    prior = RunUsage(input_tokens=10, output_tokens=2, total_tokens=12)
    request = RequestUsage(input_tokens=3, output_tokens=1, total_tokens=4)
    capture = RunUsageCapture()
    capture.observe(
        ModelUsageObservation(
            request_usage=request,
            cumulative_usage=RunUsage(
                input_tokens=prior.input_tokens + request.input_tokens,
                output_tokens=prior.output_tokens + request.output_tokens,
                total_tokens=prior.total_tokens + request.total_tokens,
            ),
            resolved_model_id=None,
        )
    )

    assert capture.snapshot().input_tokens == 13
    assert capture.snapshot().output_tokens == 3
