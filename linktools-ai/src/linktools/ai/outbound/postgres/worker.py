#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL worker persistence adapter boundary."


class WorkerStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def resolve(self, bundle_id: str) -> "object | None":
        return await self._operation.resolve(bundle_id)

    async def heartbeat(self, bundle_id: str) -> object:
        return await self._operation.heartbeat(bundle_id)


__all__ = ["WorkerStore"]
