#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL deletion persistence adapter boundary."


class DeletionStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def create(self, job: object) -> object:
        return await self._operation.create(job)

    async def advance(self, deletion_id: str, job: object) -> object:
        return await self._operation.advance(deletion_id, job)

    async def get(self, deletion_id: str) -> "object | None":
        return await self._operation.get(deletion_id)


__all__ = ["DeletionStore"]
