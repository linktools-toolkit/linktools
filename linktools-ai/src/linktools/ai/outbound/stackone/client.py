#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""StackOne account and action boundary."""


class StackOneClient:
    """Expose only the account and action operations supplied by the client."""

    def __init__(self, client: object) -> None:
        self._client = client

    async def accounts(self, request: object) -> object:
        return await self._client.accounts(request)

    async def actions(self, request: object) -> object:
        return await self._client.actions(request)


__all__ = ["StackOneClient"]
