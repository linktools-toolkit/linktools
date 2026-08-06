#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adapters from loaders to the asset source protocol."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parsing import AssetLoader


class AssetLoaderSource:
    def __init__(self, loader: "AssetLoader") -> None:
        self._loader = loader

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._loader.list_ids(suffix)

    async def read(self, path: str) -> str:
        return await self._loader.read(path)

    def identity(self, raw: str) -> str:
        return self._loader.identity(raw)


__all__ = ["AssetLoaderSource"]
