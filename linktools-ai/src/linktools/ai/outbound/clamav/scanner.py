#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ClamAV content scanner adapter."""


class ClamAVScanner:
    def __init__(self, client: object) -> None:
        self._client = client

    async def scan(self, digest: str, content: bytes) -> object:
        return await self._client.scan_stream(content)


__all__ = ["ClamAVScanner"]
