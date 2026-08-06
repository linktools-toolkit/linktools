#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation persistence and scoring protocols."""

from typing import Protocol


class EvaluationRepository(Protocol):
    async def create(self, evaluation: object) -> object: ...
    async def record_case(self, case: object) -> object: ...
    async def get(self, evaluation_id: str) -> "object | None": ...


class EvaluationRunner(Protocol):
    def list_cases(self) -> "tuple[object, ...]": ...
    def score_case(self, case: object, result: object) -> float: ...


__all__ = ["EvaluationRepository", "EvaluationRunner"]
