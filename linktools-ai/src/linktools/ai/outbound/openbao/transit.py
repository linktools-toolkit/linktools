#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenBao Transit KeyManagement adapter."""


class OpenBaoTransitAdapter:
    def __init__(self, client: object) -> None:
        self._client = client

    async def wrap(self, key: bytes) -> object: return await self._client.wrap(key)
    async def unwrap(self, reference: object) -> bytes: return await self._client.unwrap(reference)
    async def rotate(self, reference: object) -> object: return await self._client.rotate(reference)


__all__ = ["OpenBaoTransitAdapter"]
