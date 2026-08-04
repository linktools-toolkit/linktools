#!/usr/bin/env python3
"""Usage pricing must follow the model that actually served each request."""

from decimal import Decimal

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage as PydanticRequestUsage

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import AgentInput
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.execution.cancellation import CancellationToken
from linktools.ai.execution.context import RunContext
from linktools.ai.execution.snapshots import (
    ModelRequestUsageObservation,
    RequestUsage,
    RunUsageCapture,
)
from linktools.ai.execution.domain import RunnableType
from linktools.ai.execution.live_events import (
    NoopRunLiveEventSink,
    NoopSecurityEventSink,
)
from linktools.ai.model.pricing import ModelPricing
from linktools.ai.model.pricing import StaticModelPricingProvider
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver


class _AsyncCaptureSink:
    def __init__(self, capture):
        self.capture = capture

    async def observe_request(self, observation, *, pricing):
        return self.capture.observe_request(observation, pricing=pricing)

    def snapshot(self):
        return self.capture.snapshot()


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
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="request-1",
            usage=first_request,
            provider_name="provider",
            response_model_name="actual",
        ),
        pricing=actual,
    )
    second_request = RequestUsage(
        input_tokens=1,
        output_tokens=1,
        total_cost=actual.cost(input_tokens=1, output_tokens=1),
    )
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="request-2",
            usage=second_request,
            provider_name="provider",
            response_model_name="actual",
        ),
        pricing=actual,
    )

    assert capture.snapshot().total_cost == Decimal("0.18")
    assert primary.cost(input_tokens=2, output_tokens=3) != first_request.total_cost


def test_capture_keeps_cost_unknown_until_authoritative_cost_arrives():
    capture = RunUsageCapture()
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="request-unknown",
            usage=RequestUsage(input_tokens=4, output_tokens=1),
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )
    assert capture.snapshot().total_cost is None
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="request-authoritative",
            usage=RequestUsage(total_cost=Decimal("0.21")),
            provider_name="provider",
            response_model_name="actual",
        ),
        pricing=None,
    )
    assert capture.snapshot().total_cost is None


@pytest.mark.asyncio
async def test_agent_engine_records_actual_fallback_response_usage_once():
    def primary(messages, info: AgentInfo):
        raise ModelHTTPError(503, "primary", None)

    def fallback(messages, info: AgentInfo):
        return ModelResponse(
            parts=[TextPart(content="ok")],
            usage=PydanticRequestUsage(input_tokens=4, output_tokens=2),
            model_name="fallback",
            provider_name=None,
        )

    registry = ModelRegistry()
    registry.register("primary", model=FunctionModel(primary))
    registry.register("fallback", model=FunctionModel(fallback))
    spec = AgentSpec(
        id="pricing-agent",
        name="pricing-agent",
        model=ModelPolicy(primary="primary", fallbacks=("fallback",)),
        instructions=PromptSpec(instructions="answer"),
    )
    compiled = await AgentCompiler(
        model_resolver=ModelResolver(registry=registry)
    ).compile(spec)
    capture = RunUsageCapture()
    sink = _AsyncCaptureSink(capture)
    engine = AgentEngine(
        pricing_provider=StaticModelPricingProvider(
            {
                "fallback": ModelPricing(
                    "fallback",
                    input_cost_per_token=Decimal("0.1"),
                    output_cost_per_token=Decimal("0.2"),
                )
            }
        )
    )
    for run_id in ("run-1", "run-2"):
        await engine.execute_pure(
            compiled,
            AgentInput(prompt="hello"),
            RunContext(
                run_id,
                run_id,
                None,
                "session",
                "pricing-agent",
                RunnableType.AGENT,
                None,
                None,
                None,
            ),
            cancellation=CancellationToken(),
            live_events=NoopRunLiveEventSink(),
            security_events=NoopSecurityEventSink(),
            usage_sink=sink,
        )

    assert capture.snapshot().input_tokens == 8
    assert capture.snapshot().output_tokens == 4
    assert capture.snapshot().total_cost == Decimal("1.6")
    assert len(capture.observations) == 2
