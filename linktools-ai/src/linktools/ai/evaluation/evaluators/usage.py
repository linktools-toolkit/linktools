#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UsageEvaluator: score based on token/cost efficiency."""

from ..models import EvalCase, EvalExecution, EvalScore


class UsageEvaluator:
    @property
    def name(self) -> str:
        return "usage"

    async def evaluate(
        self,
        case: EvalCase,
        execution: EvalExecution,
        snapshot: "object | None" = None,
    ) -> EvalScore:
        limits = case.metadata.get("usage_limits", {})
        max_tokens = limits.get("max_tokens")
        max_cost = limits.get("max_cost")
        # Usage comes from the executor-captured model_usage (RunResult.
        # token_usage), not the run's free-form output. Fall back to a legacy
        # output-dict shape only when no usage was captured.
        usage = dict(execution.model_usage or {})
        if not usage and isinstance(execution.output, dict):
            usage = execution.output
        tokens = usage.get("total_tokens") or 0
        cost = float(usage.get("total_cost") or 0)
        over = []
        if max_tokens is not None and tokens > max_tokens:
            over.append("tokens")
        if max_cost is not None and cost > max_cost:
            over.append("cost")
        if over:
            return EvalScore(
                evaluator_name=self.name,
                score=0.0,
                details={"over_budget": over, "tokens": tokens, "cost": cost},
            )
        return EvalScore(evaluator_name=self.name, score=1.0)


__all__: "list[str]" = ["UsageEvaluator"]
