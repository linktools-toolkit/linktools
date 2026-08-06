#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source revision invalidation coordination."""


class SourceService:
    def __init__(self, source: object, index: object) -> None:
        self._source = source
        self._index = index

    async def refresh(self) -> object:
        return await self._index.refresh()


__all__ = ["SourceService"]
