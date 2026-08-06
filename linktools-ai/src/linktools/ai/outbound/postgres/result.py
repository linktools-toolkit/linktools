#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL result persistence adapter boundary."


class ResultStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def commit(self, result: object) -> object:
        return await self._operation.commit(result)

    async def get(self, execution_id: str) -> "object | None":
        return await self._operation.get(execution_id)


__all__ = ["ResultStore"]
