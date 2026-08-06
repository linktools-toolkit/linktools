#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Exa retrieval client boundary."""


class ExaClient:
    def __init__(self, client: object) -> None:
        self._client = client

    async def search(self, query: str, limit: int) -> "tuple[object, ...]":
        return tuple(await self._client.search(query, limit=limit))


__all__ = ["ExaClient"]
