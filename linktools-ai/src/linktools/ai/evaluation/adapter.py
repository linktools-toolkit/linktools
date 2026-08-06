#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adapter for official Pydantic Evals datasets/evaluators."""


class PydanticEvalsAdapter:
    """Adapt duck-typed official Dataset/Evaluator objects without running Agents."""

    def __init__(self, dataset: object, evaluator: object) -> None:
        self._dataset = dataset
        self._evaluator = evaluator

    def list_cases(self) -> "tuple[object, ...]":
        try:
            cases = self._dataset.cases
        except AttributeError:
            cases = self._dataset
        return tuple(cases)

    def score_case(self, case: object, result: object) -> float:
        score = self._evaluator.evaluate(case, result)
        return float(score if isinstance(score, (int, float)) else score.value)


__all__ = ["PydanticEvalsAdapter"]
