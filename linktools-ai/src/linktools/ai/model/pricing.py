#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model cost pricing + budget enforcement.

ModelPricing is the per-token cost for one model id; ModelPricingProvider looks
it up. AgentEngine asks the provider for the model's pricing, multiplies by the
RunUsage token counts, and -- when ModelPolicy.budget is set -- raises
ModelPolicyExceededError once the cumulative cost crosses it. Decimal
throughout (float would lose precision on small per-token costs).

A budget set without a pricing provider is a configuration error: the run
refuses to start rather than silently running without a cost limit."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token cost for one model. Decimal so fractional per-token costs
    (e.g. 0.000001) round-trip exactly.

    Prompt-cache pricing is separate when the provider bills cache reads/writes
    differently from a standard input token (Anthropic: cache_read ~0.1x,
    cache_write ~1.25x). When a model has no cache concept, leave the cache
    rates at their defaults (equal to the standard input rate) and pass
    cache_*_tokens=0 -- cost() is then identical to input/output-only."""

    model_id: str
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    currency: str = "USD"
    cache_read_cost_per_token: "Decimal | None" = None
    cache_write_cost_per_token: "Decimal | None" = None

    def cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Decimal:
        # Cache rates default to the standard input rate so an unconfigured
        # model (no separate cache pricing) bills cache tokens as regular input.
        read_rate = self.cache_read_cost_per_token if self.cache_read_cost_per_token is not None else self.input_cost_per_token
        write_rate = self.cache_write_cost_per_token if self.cache_write_cost_per_token is not None else self.input_cost_per_token
        return (
            self.input_cost_per_token * Decimal(input_tokens)
            + self.output_cost_per_token * Decimal(output_tokens)
            + read_rate * Decimal(cache_read_tokens)
            + write_rate * Decimal(cache_write_tokens)
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
