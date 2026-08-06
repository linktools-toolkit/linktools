#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Macroscope CLI boundary."""


class MacroscopeClient:
    def __init__(self, runner: object) -> None:
        self._runner = runner

    async def inspect(self, root: str) -> object:
        return await self._runner(root)


__all__ = ["MacroscopeClient"]
