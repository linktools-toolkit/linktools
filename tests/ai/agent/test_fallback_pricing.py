#!/usr/bin/env python3
"""Usage pricing must follow the model that actually served each request."""

from decimal import Decimal

from linktools.ai.execution.domain import RunUsage
from linktools.ai.execution.snapshots import (
    ModelUsageObservation,
    RequestUsage,
    RunUsageCapture,
)
from linktools.ai.model.pricing import ModelPricing


def test_capture_prices_actual_model_without_using_primary_model():
    actual = ModelPricing(
        "provider/actual",
        input_cost_per_token=Decimal("0.02"),
        output_cost_per_token=Decimal("0.03"),
    )
    primary = ModelPricing(
        "provider/primary",
        input_cost_per_token=Decimal("9"),
        output_cost_per_token=Decimal("9"),
    )
    capture = RunUsageCapture()

    first_request = RequestUsage(
        input_tokens=2,
        output_tokens=3,
        total_cost=actual.cost(input_tokens=2, output_tokens=3),
    )
    capture.observe(
        ModelUsageObservation(
            request_usage=first_request,
            cumulative_usage=RunUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            resolved_model_id="provider/actual",
        )
    )
    second_request = RequestUsage(
        input_tokens=1,
        output_tokens=1,
        total_cost=actual.cost(input_tokens=1, output_tokens=1),
    )
    capture.observe(
        ModelUsageObservation(
            request_usage=second_request,
            cumulative_usage=RunUsage(input_tokens=3, output_tokens=4, total_tokens=7),
            resolved_model_id="provider/actual",
        )
    )

    assert capture.snapshot().total_cost == Decimal("0.18")
    assert primary.cost(input_tokens=2, output_tokens=3) != first_request.total_cost


def test_capture_keeps_cost_unknown_until_authoritative_cost_arrives():
    capture = RunUsageCapture()
    capture.observe(
        ModelUsageObservation(
            request_usage=RequestUsage(input_tokens=4, output_tokens=1),
            cumulative_usage=RunUsage(input_tokens=4, output_tokens=1, total_tokens=5),
            resolved_model_id=None,
        )
    )
    assert capture.snapshot().total_cost is None

    capture.observe(
        ModelUsageObservation(
            request_usage=RequestUsage(),
            cumulative_usage=RunUsage(input_tokens=4, output_tokens=1, total_tokens=5),
            resolved_model_id=None,
            provider_total_cost=Decimal("0.21"),
        )
    )
    assert capture.snapshot().total_cost == Decimal("0.21")
