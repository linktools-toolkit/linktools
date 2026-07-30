#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SchemaEvaluator: score 1.0 if output matches a JSON schema, else 0.0."""

from ..models import EvalCase, EvalExecution, EvalScore


class SchemaEvaluator:
    @property
    def name(self) -> str:
        return "schema"

    async def evaluate(
        self,
        case: EvalCase,
        execution: EvalExecution,
        snapshot: "object | None" = None,
    ) -> EvalScore:
        schema = case.metadata.get("schema")
        if schema is None:
            return EvalScore(evaluator_name=self.name, score=1.0)
        output = execution.output
        if isinstance(output, dict) and isinstance(schema, dict):
            required = schema.get("required", [])
            missing = [k for k in required if k not in output]
            if missing:
                return EvalScore(
                    evaluator_name=self.name,
                    score=0.0,
                    details={"missing_keys": missing},
                )
            return EvalScore(evaluator_name=self.name, score=1.0)
        return EvalScore(evaluator_name=self.name, score=0.0)


__all__: "list[str]" = ["SchemaEvaluator"]
