#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"SQLite memory persistence adapter boundary."


class MemoryStore:
    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def get(self, namespace: str, key: str) -> "object | None":
        return await self._operation.get(namespace, key)

    async def put(self, namespace: str, key: str, value: object) -> None:
        await self._operation.put(namespace, key, value)


__all__ = ["MemoryStore"]
