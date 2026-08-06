#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"PostgreSQL conversation persistence adapter boundary."


class ConversationStore:
    "Map a domain port to an injected async session operation."

    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def create(self, conversation: object) -> object:
        return await self._operation.create(conversation)

    async def get(self, conversation_id: str) -> "object | None":
        return await self._operation.get(conversation_id)


__all__ = ["ConversationStore"]
