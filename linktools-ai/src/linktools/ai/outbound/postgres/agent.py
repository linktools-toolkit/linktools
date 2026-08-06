#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL agent persistence adapter boundary."


class AgentStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def get(self, agent_id: str, revision: int) -> "object | None":
        return await self._operation.get(agent_id, revision)

    async def get_enabled(self, agent_id: str) -> "object | None":
        return await self._operation.get_enabled(agent_id)

    async def save(self, release: object) -> object:
        return await self._operation.save(release)


__all__ = ["AgentStore"]
