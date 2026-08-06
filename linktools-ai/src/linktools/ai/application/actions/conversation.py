#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation creation and query actions."""


class CreateConversation:
    def __init__(self, repository: object) -> None: self._repository = repository
    async def execute(self, conversation: object) -> object: return await self._repository.create(conversation)


class InspectConversation:
    def __init__(self, repository: object) -> None: self._repository = repository
    async def execute(self, conversation_id: str) -> object: return await self._repository.get(conversation_id)


__all__ = ["CreateConversation", "InspectConversation"]
