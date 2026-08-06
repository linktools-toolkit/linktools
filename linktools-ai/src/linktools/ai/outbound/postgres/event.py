#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL event persistence adapter boundary."


class EventStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def append(self, event: object) -> object:
        return await self._operation.append(event)

    async def list_after(self, execution_id: str, after_sequence: int, limit: int) -> object:
        return await self._operation.list_after(execution_id, after_sequence, limit)


__all__ = ["EventStore"]
