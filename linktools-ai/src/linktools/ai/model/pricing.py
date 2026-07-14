#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model cost pricing + budget enforcement.

ModelPricing is the per-token cost for one model id; ModelPricingProvider looks
it up. AgentRunner asks the provider for the model's pricing, multiplies by the
RunUsage token counts, and -- when ModelPolicy.budget is set -- raises
ModelPolicyExceededError once the cumulative cost crosses it. Decimal
throughout (float would lose precision on small per-token costs).

A budget set without a pricing provider is a configuration error: the run
refuses to start rather than silently running without a cost limit (§18.6)."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token cost for one model. Decimal so fractional per-token costs
    (e.g. 0.000001) round-trip exactly."""

    model_id: str
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    currency: str = "USD"

    def cost(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        return self.input_cost_per_token * Decimal(input_tokens) + (
            self.output_cost_per_token * Decimal(output_tokens)
        )


@runtime_checkable
class ModelPricingProvider(Protocol):
    """Resolve ModelPricing for a model id. Returns None when the model is not
    priced (free / unknown) -- a budget set against an unpriced model is a
    configuration error the caller raises on."""

    async def get_pricing(self, model_id: str) -> "ModelPricing | None": ...


class StaticModelPricingProvider:
    """A pricing provider backed by a fixed {model_id: ModelPricing} mapping --
    the test/default implementation. Production wires one that reads a price
    sheet."""

    def __init__(self, pricing: "dict[str, ModelPricing]") -> None:
        self._pricing = dict(pricing)

    async def get_pricing(self, model_id: str) -> "ModelPricing | None":
        return self._pricing.get(model_id)
