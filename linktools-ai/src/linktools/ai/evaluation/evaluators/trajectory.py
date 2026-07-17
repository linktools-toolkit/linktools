#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TrajectoryEvaluator: check required/forbidden actions in the trajectory."""

from ..models import EvalCase, EvalExecution, EvalScore


class TrajectoryEvaluator:
    @property
    def name(self) -> str:
        return "trajectory"

    async def evaluate(
        self,
        case: EvalCase,
        execution: EvalExecution,
        snapshot: "object | None" = None,
    ) -> EvalScore:
        config = case.metadata.get("trajectory", {})
        required = set(config.get("required_actions", []))
        forbidden = set(config.get("forbidden_actions", []))
        max_calls = config.get("max_action_calls")
        trajectory = execution.output if isinstance(execution.output, dict) else {}
        actions = set()
        call_count = 0
        if isinstance(trajectory, dict):
            actions = set(trajectory.get("actions", []))
            call_count = trajectory.get("total_calls", 0)
        missing = required - actions
        violations = forbidden & actions
        over_limit = max_calls is not None and call_count > max_calls
        if missing or violations or over_limit:
            return EvalScore(
                evaluator_name=self.name,
                score=0.0,
                details={
                    "missing_required": sorted(missing),
                    "forbidden_used": sorted(violations),
                    "over_limit": over_limit,
                },
            )
        return EvalScore(evaluator_name=self.name, score=1.0)


__all__: "list[str]" = ["TrajectoryEvaluator"]
