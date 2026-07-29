#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt a text loader to the generic specification source protocol."""

from __future__ import annotations

from .contracts import SpecSource
from .parsing import SpecLoader


class SpecLoaderSource:
    """SpecSource adapter over a SpecLoader."""

    def __init__(self, loader: SpecLoader) -> None:
        self._loader = loader

    async def revision(self) -> str:
        return str(await self._loader.revision())

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._loader.list_ids(suffix)

    async def read(self, path: str) -> str:
        return await self._loader.read(path)


__all__: "list[str]" = ["SpecLoaderSource"]
