#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation persistence protocol."""

from typing import Protocol


class ConversationRepository(Protocol):
    async def create(self, conversation: object) -> object: ...
    async def get(self, conversation_id: str) -> "object | None": ...


__all__ = ["ConversationRepository"]
