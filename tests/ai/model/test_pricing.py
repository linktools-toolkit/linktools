#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ModelPricing: per-token cost including prompt-cache billing.

Cache reads/writes bill at provider-specific rates (Anthropic: cache_read
~0.1x, cache_write ~1.25x of the standard input rate). When a model has no
separate cache pricing, cache tokens bill at the standard input rate so
cost() stays backward-compatible."""

from decimal import Decimal

from linktools.ai.model.pricing import ModelPricing


def _pricing(*, read=None, write=None):
    return ModelPricing(
        "m",
        Decimal("0.001"),
        Decimal("0.002"),
        cache_read_cost_per_token=read,
        cache_write_cost_per_token=write,
    )


def test_cost_without_cache_tokens_matches_input_output_only():
    p = _pricing()
    assert p.cost(input_tokens=100, output_tokens=50) == Decimal("0.200")


def test_cache_tokens_bill_at_standard_input_rate_when_unconfigured():
    # No separate cache rate -> cache tokens billed as regular input.
    p = _pricing()
    assert p.cost(input_tokens=100, output_tokens=50, cache_read_tokens=200, cache_write_tokens=100) == Decimal("0.500")


def test_cache_separate_rates_billed_independently():
    p = _pricing(read=Decimal("0.0001"), write=Decimal("0.00125"))
    # 100*0.001 + 50*0.002 + 200*0.0001 + 100*0.00125 = 0.1 + 0.1 + 0.02 + 0.125
    assert p.cost(input_tokens=100, output_tokens=50, cache_read_tokens=200, cache_write_tokens=100) == Decimal("0.34500")


def test_zero_cache_tokens_when_no_cache_concept():
    # A non-cache provider passes 0 for cache tokens -> standard cost.
    p = _pricing(read=Decimal("0.0001"), write=Decimal("0.00125"))
    assert p.cost(input_tokens=100, output_tokens=50) == Decimal("0.200")
