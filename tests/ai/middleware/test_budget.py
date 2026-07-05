#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/middleware/test_budget.py"""
from dataclasses import dataclass

import pytest

from linktools.ai.budget.tracker import BudgetExceededError, BudgetTracker
from linktools.ai.middleware.budget import BudgetMiddleware


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


class _FakeResponse:
    def __init__(self, usage: _FakeUsage) -> None:
        self.usage = usage


@pytest.mark.asyncio
async def test_after_model_records_usage_against_tracker():
    tracker = BudgetTracker(budget_usd=10.0, cost_per_1k_input_tokens=1.0, cost_per_1k_output_tokens=2.0)
    middleware = BudgetMiddleware(tracker=tracker)
    response = _FakeResponse(usage=_FakeUsage(input_tokens=1000, output_tokens=1000))
    result = await middleware.after_model(context=None, response=response)
    assert result is response
    assert tracker.spent_usd == pytest.approx(3.0)  # 1000/1000*1.0 + 1000/1000*2.0


@pytest.mark.asyncio
async def test_after_model_raises_once_budget_exceeded():
    tracker = BudgetTracker(budget_usd=1.0, cost_per_1k_input_tokens=10.0)
    middleware = BudgetMiddleware(tracker=tracker)
    response = _FakeResponse(usage=_FakeUsage(input_tokens=1000, output_tokens=0))
    with pytest.raises(BudgetExceededError):
        await middleware.after_model(context=None, response=response)
