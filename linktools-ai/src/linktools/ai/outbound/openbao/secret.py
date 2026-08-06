#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OpenBao SecretProvider adapter."""


class OpenBaoSecretProvider:
    def __init__(self, client: object) -> None:
        self._client = client

    async def resolve(self, reference: str) -> object:
        return await self._client.read(reference)


__all__ = ["OpenBaoSecretProvider"]
