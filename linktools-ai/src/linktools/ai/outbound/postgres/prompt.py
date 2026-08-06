#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL prompt persistence adapter boundary."


class PromptStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def save(self, prompt: object) -> object:
        return await self._operation.save(prompt)

    async def get(self, snapshot_id: str) -> "object | None":
        return await self._operation.get(snapshot_id)


__all__ = ["PromptStore"]
