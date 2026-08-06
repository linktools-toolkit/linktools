#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Managed Prompt client boundary."""


class ManagedPromptClient:
    def __init__(self, client: object) -> None:
        self._client = client

    async def resolve(self, prompt_name: str) -> str:
        return await self._client.resolve(prompt_name)


__all__ = ["ManagedPromptClient"]
