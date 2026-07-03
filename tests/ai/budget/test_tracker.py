import pytest
from pydantic_ai.usage import RequestUsage

from linktools.ai.budget.tracker import BudgetExceededError, BudgetTracker


def test_starts_at_zero_spend():
    tracker = BudgetTracker(budget_usd=1.0)
    assert tracker.spent_usd == 0.0
    tracker.check()  # must not raise


def test_record_accrues_cost_from_usage():
    tracker = BudgetTracker(budget_usd=10.0, cost_per_1k_input_tokens=0.01, cost_per_1k_output_tokens=0.03)
    tracker.record(RequestUsage(input_tokens=1000, output_tokens=1000))
    assert tracker.spent_usd == pytest.approx(0.04)


def test_check_raises_once_budget_exceeded():
    tracker = BudgetTracker(budget_usd=0.05, cost_per_1k_input_tokens=0.01, cost_per_1k_output_tokens=0.0)
    tracker.record(RequestUsage(input_tokens=6000, output_tokens=0))  # 0.06 spent, cap is 0.05
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_check_does_not_raise_when_under_budget():
    tracker = BudgetTracker(budget_usd=1.0, cost_per_1k_input_tokens=0.01, cost_per_1k_output_tokens=0.0)
    tracker.record(RequestUsage(input_tokens=1000, output_tokens=0))  # 0.01 spent
    tracker.check()  # must not raise
