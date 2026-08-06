#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"SQLite transcript persistence adapter boundary."


class TranscriptStore:
    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def append(self, segment: object) -> object:
        return await self._operation.append(segment)

    async def list(self, execution_id: str) -> "tuple[object, ...]":
        return await self._operation.list(execution_id)

    async def search(self, execution_id: str, query: str) -> "tuple[object, ...]":
        return await self._operation.search(execution_id, query)


__all__ = ["TranscriptStore"]
