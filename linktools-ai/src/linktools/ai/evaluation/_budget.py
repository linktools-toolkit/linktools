#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure observed-usage estimates for evaluation soft budgets."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from ..core import JsonValue, UsageMetrics
from ._contracts import EvaluationPolicy
from ._evidence import ModelUsage


_RATE_NAMES = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def _rate(value: JsonValue) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("model prices must be decimal strings")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("model prices must be decimal strings") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("model prices must be finite and nonnegative")
    return amount


@dataclass(frozen=True, slots=True)
class _ModelPrice:
    rates: tuple[Decimal | None, ...]
    input_includes_cache: bool

    def cost(self, usage: UsageMetrics, unit_tokens: int) -> Decimal | None:
        ordinary_input = usage.input_tokens
        if self.input_includes_cache:
            ordinary_input -= usage.cache_read_tokens + usage.cache_write_tokens
        if ordinary_input < 0 or any(rate is None for rate in self.rates):
            return None
        counts = (ordinary_input, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens)
        return sum((count * rate for count, rate in zip(counts, self.rates)
                    if rate is not None), Decimal(0)) / unit_tokens


@dataclass(frozen=True, slots=True)
class PriceTable:
    currency: str
    unit_tokens: int
    models: Mapping[tuple[str, str], _ModelPrice]

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue], *, currency: str | None = None) -> "PriceTable":
        if not isinstance(value, Mapping):
            raise ValueError("price table must be an object")
        table_currency = value.get("currency")
        if not isinstance(table_currency, str) or not table_currency.strip():
            raise ValueError("price table requires a currency")
        if currency is not None and table_currency != currency:
            raise ValueError("price table currency differs from budget currency")
        unit_tokens = value.get("unit_tokens")
        if isinstance(unit_tokens, bool) or not isinstance(unit_tokens, int) or unit_tokens <= 0:
            raise ValueError("price table token unit must be a positive integer")
        entries = value.get("models")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("price table requires model entries")
        models: dict[tuple[str, str], _ModelPrice] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("price table model entry must be an object")
            provider, model = entry.get("provider"), entry.get("model")
            if any(not isinstance(item, str) or not item.strip() for item in (provider, model)):
                raise ValueError("price table requires exact provider and model identities")
            key = (provider, model)
            if key in models:
                raise ValueError("price table contains duplicate provider/model entries")
            includes_cache = entry.get("input_includes_cache")
            if not isinstance(includes_cache, bool):
                raise ValueError("price table must declare whether input includes cache tokens")
            models[key] = _ModelPrice(tuple(_rate(entry[name]) if name in entry else None
                                           for name in _RATE_NAMES), includes_cache)
        return cls(table_currency, unit_tokens, MappingProxyType(models))


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    tokens: int | None
    cost: str | None
    currency: str | None
    stop: bool
    reason: str | None


def estimate_model_budget(
    policy: EvaluationPolicy,
    model_usage: Iterable[ModelUsage],
    *,
    usage_complete: bool = True,
    price_table: PriceTable | None = None,
) -> BudgetEstimate:
    """Count every supplied attempt; callers own collection and deduplication.

    Empty complete input means no model requests. Missing observations must be
    marked incomplete. Observed lower bounds can reach a limit even when the
    final total is unknown; this never reserves or gates individual requests.
    """
    if price_table is not None and policy.currency is not None and price_table.currency != policy.currency:
        raise ValueError("price table currency differs from budget currency")
    tokens = 0
    cost = Decimal(0)
    tokens_complete = costs_complete = usage_complete
    for item in model_usage:
        tokens += item.usage.total_tokens
        tokens_complete = tokens_complete and item.complete
        price = None if price_table is None else price_table.models.get((item.provider, item.model))
        amount = None if price is None else price.cost(item.usage, price_table.unit_tokens)
        costs_complete = costs_complete and item.complete and amount is not None
        if amount is not None:
            cost += amount
    reason = None
    if policy.token_limit is not None and tokens >= policy.token_limit:
        reason = "token_limit"
    elif policy.cost_limit is not None and cost >= Decimal(policy.cost_limit):
        reason = "cost_limit"
    elif policy.unknown_usage == "stop" and (
        policy.token_limit is not None and not tokens_complete
        or policy.cost_limit is not None and not costs_complete
    ):
        reason = "unknown_usage"
    return BudgetEstimate(
        tokens if tokens_complete else None,
        str(cost) if costs_complete else None,
        price_table.currency if price_table is not None else policy.currency,
        reason is not None,
        reason,
    )


__all__ = ["BudgetEstimate", "PriceTable", "estimate_model_budget"]
