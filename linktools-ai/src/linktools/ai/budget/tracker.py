#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BudgetTracker: accrue $ cost from model usage and enforce a hard cap.

Cost rates are supplied by the caller (per-model $/1k-token pricing isn't
something pydantic-ai exposes, and hardcoding a pricing table here would go
stale immediately) — a tracker constructed with rate 0.0 (the default) never
accrues cost and never trips the budget, which is the correct behavior when
the caller doesn't know or care about $ cost and just wants the toggle to
exist without side effects.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import AgentRunError

if TYPE_CHECKING:
    from pydantic_ai.usage import RequestUsage


class BudgetExceededError(AgentRunError):
    pass


@dataclass
class BudgetTracker:
    budget_usd: float
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    spent_usd: float = field(default=0.0)

    def record(self, usage: "RequestUsage") -> None:
        self.spent_usd += (usage.input_tokens / 1000) * self.cost_per_1k_input_tokens
        self.spent_usd += (usage.output_tokens / 1000) * self.cost_per_1k_output_tokens

    def check(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"budget exceeded: spent ${self.spent_usd:.4f} of ${self.budget_usd:.4f}"
            )
