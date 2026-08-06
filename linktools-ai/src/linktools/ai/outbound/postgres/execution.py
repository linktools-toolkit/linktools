#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL execution persistence adapter boundary."


class ExecutionStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def upsert_projection(self, view: object) -> object:
        return await self._operation.upsert_projection(view)

    async def get(self, execution_id: str) -> "object | None":
        return await self._operation.get(execution_id)

    async def repair(self, execution_id: str) -> object:
        return await self._operation.repair(execution_id)


__all__ = ["ExecutionStore"]
