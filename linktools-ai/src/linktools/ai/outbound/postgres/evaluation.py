#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL evaluation persistence adapter boundary."


class EvaluationStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def create(self, evaluation: object) -> object:
        return await self._operation.create(evaluation)

    async def record_case(self, case: object) -> object:
        return await self._operation.record_case(case)

    async def get(self, evaluation_id: str) -> "object | None":
        return await self._operation.get(evaluation_id)

    def list_cases(self) -> "tuple[object, ...]":
        return self._operation.list_cases()

    def score_case(self, case: object, result: object) -> float:
        return self._operation.score_case(case, result)


__all__ = ["EvaluationStore"]
