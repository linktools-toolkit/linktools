#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExactMatchEvaluator: score 1.0 if output == expected, else 0.0."""

from ..models import EvalCase, EvalExecution, EvalScore


class ExactMatchEvaluator:
    @property
    def name(self) -> str:
        return "exact_match"

    async def evaluate(
        self,
        case: EvalCase,
        execution: EvalExecution,
        snapshot: "object | None" = None,
    ) -> EvalScore:
        output = execution.output
        # Expected comes from the case metadata (simplified for direct mode;
        # full artifact-resolved expected_ref lands with RunSnapshot).
        expected = case.metadata.get("expected")
        if expected is not None and output == expected:
            return EvalScore(evaluator_name=self.name, score=1.0)
        return EvalScore(evaluator_name=self.name, score=0.0)


__all__: "list[str]" = ["ExactMatchEvaluator"]
