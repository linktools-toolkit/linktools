#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import replace
from decimal import Decimal

import pytest

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.core import UsageMetrics
from linktools.ai.evaluation import EvaluationPolicy, ModelUsage, PriceTable, estimate_model_budget
from linktools.ai.storage import StorageEntryRevision
from linktools.ai.task import TaskGraphLimits


def _table() -> dict:
    return {
        "currency": "USD", "unit_tokens": 1_000_000,
        "models": [{
            "provider": "provider", "model": "model-2026-10-01",
            "input_tokens": "2", "output_tokens": "8",
            "cache_read_tokens": "0.5", "cache_write_tokens": "3",
            "input_includes_cache": True,
        }],
    }


def _cost_policy(limit: str = "1", **kwargs: object) -> EvaluationPolicy:
    ref = AssetVersionRef(AssetKey("price_table", "prices"), "layer", StorageEntryRevision(1), "a" * 64, 0)
    return EvaluationPolicy(cost_limit=limit, currency="USD", price_table=ref, **kwargs)


def _usage(**kwargs: object) -> ModelUsage:
    return ModelUsage("provider", "model-2026-10-01", UsageMetrics(model_requests=1, **kwargs), True)


def test_all_attempts_are_counted_with_decimal_cache_prices() -> None:
    table = PriceTable.from_mapping(_table(), currency="USD")
    attempt = _usage(input_tokens=100, output_tokens=50, cache_read_tokens=20, cache_write_tokens=10)
    judge = _usage(input_tokens=50, output_tokens=25)
    result = estimate_model_budget(_cost_policy(), (attempt, attempt, judge), price_table=table)
    assert result.tokens == 375
    assert Decimal(result.cost) == Decimal("0.00146")
    assert result.currency == "USD"
    assert not result.stop and result.reason is None


def test_exclusive_input_prices_do_not_subtract_cache_twice() -> None:
    value = _table()
    value["models"][0]["input_includes_cache"] = False
    result = estimate_model_budget(
        _cost_policy(), (_usage(input_tokens=70, output_tokens=50, cache_read_tokens=20, cache_write_tokens=10),),
        price_table=PriceTable.from_mapping(value),
    )
    assert Decimal(result.cost) == Decimal("0.00058")


@pytest.mark.parametrize("field,value", [
    ("input_tokens", "NaN"), ("output_tokens", "Infinity"),
    ("cache_read_tokens", "-1"), ("cache_write_tokens", 0.1),
    ("input_tokens", None), ("input_tokens", "not-a-number"),
    ("input_includes_cache", 1), ("model", ""),
])
def test_invalid_price_fields_are_rejected(field: str, value: object) -> None:
    payload = _table()
    payload["models"][0][field] = value
    with pytest.raises(ValueError):
        PriceTable.from_mapping(payload)


@pytest.mark.parametrize("field,value", [("unit_tokens", 0), ("unit_tokens", True), ("currency", ""),
                                          ("models", {}), ("models", ["invalid"])])
def test_invalid_price_table_structure_is_rejected(field: str, value: object) -> None:
    payload = _table()
    payload[field] = value
    with pytest.raises(ValueError):
        PriceTable.from_mapping(payload)


def test_price_table_rejects_duplicate_models_and_mixed_currency() -> None:
    payload = _table()
    payload["models"].append(dict(payload["models"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        PriceTable.from_mapping(payload)
    with pytest.raises(ValueError, match="currency"):
        PriceTable.from_mapping(_table(), currency="EUR")
    table = PriceTable.from_mapping({**_table(), "currency": "EUR"})
    with pytest.raises(ValueError, match="currency"):
        estimate_model_budget(_cost_policy(), (), price_table=table)


@pytest.mark.parametrize("missing", ("rate", "provider", "actual_model", "table", "cache_count"))
def test_unknown_cost_is_not_zero_or_guessed(missing: str) -> None:
    payload = _table()
    usage = _usage(input_tokens=100, output_tokens=50)
    if missing == "rate":
        del payload["models"][0]["output_tokens"]
    elif missing == "provider":
        usage = replace(usage, provider="another-provider")
    elif missing == "actual_model":
        usage = replace(usage, model="model")
    elif missing == "cache_count":
        usage = _usage(input_tokens=10, cache_read_tokens=11)
    table = None if missing == "table" else PriceTable.from_mapping(payload)
    stopped = estimate_model_budget(_cost_policy(), (usage,), price_table=table)
    assert stopped.cost is None and stopped.stop and stopped.reason == "unknown_usage"
    continued = estimate_model_budget(_cost_policy(unknown_usage="continue"), (usage,), price_table=table)
    assert continued.cost is None and not continued.stop


def test_token_budget_does_not_require_model_identity_or_prices() -> None:
    usage = ModelUsage("", "", UsageMetrics(input_tokens=100, output_tokens=50), True)
    result = estimate_model_budget(EvaluationPolicy(token_limit=151), (usage,))
    assert result.tokens == 150 and result.cost is None and not result.stop
    reached = estimate_model_budget(EvaluationPolicy(token_limit=150), (usage,))
    assert reached.stop and reached.reason == "token_limit"


def test_unknown_tokens_stop_or_continue_but_known_subtotals_still_stop() -> None:
    usage = replace(_usage(input_tokens=100, output_tokens=50), complete=False)
    policy = EvaluationPolicy(token_limit=151)
    result = estimate_model_budget(policy, (usage,))
    assert result.tokens is None and result.cost is None and result.reason == "unknown_usage"
    policy = replace(policy, unknown_usage="continue")
    assert not estimate_model_budget(policy, (usage,)).stop
    reached = estimate_model_budget(replace(policy, token_limit=150), (usage,))
    assert reached.tokens is None and reached.stop and reached.reason == "token_limit"


def test_decimal_cost_limit_stops_at_equality_including_partial_usage() -> None:
    table = PriceTable.from_mapping(_table())
    usage = _usage(input_tokens=50, output_tokens=25)
    result = estimate_model_budget(_cost_policy("0.0003"), (usage,), price_table=table)
    assert result.stop and result.reason == "cost_limit"
    partial = estimate_model_budget(
        _cost_policy("0.0003", unknown_usage="continue"), (usage,),
        usage_complete=False, price_table=table,
    )
    assert partial.tokens is None and partial.cost is None and partial.reason == "cost_limit"


def test_no_model_budget_does_not_block_unknown_usage_or_mix_graph_limits() -> None:
    policy = EvaluationPolicy(target_graph_limits=TaskGraphLimits(max_budget=1))
    assert not estimate_model_budget(policy, (_usage(input_tokens=10000),)).stop
    unknown = estimate_model_budget(policy, (), usage_complete=False)
    assert unknown.tokens is None and unknown.cost is None and not unknown.stop
    empty = estimate_model_budget(policy, ())
    assert empty.tokens == 0 and empty.cost == "0" and not empty.stop
    reached = estimate_model_budget(EvaluationPolicy(token_limit=0), ())
    assert reached.stop and reached.reason == "token_limit"
