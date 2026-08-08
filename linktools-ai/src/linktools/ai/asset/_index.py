#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-identity keyed index for parsed text assets."""

from typing import Generic, TypeVar

from ._codec import AssetCodec
from ._parsing import AssetLoader, AssetLoaderSource
from .domain import AssetSource

TAsset = TypeVar("TAsset")


class AssetIndex(Generic[TAsset]):
    def __init__(self, source: AssetSource, codec: AssetCodec[TAsset], *, suffix: str) -> None:
        self._suffix = suffix
        self._codec = codec
        self._source = source
        self._cache: dict[tuple[str, str], TAsset] = {}

    @classmethod
    def source_from_loader(cls, loader: AssetLoader) -> AssetSource:
        return AssetLoaderSource(loader)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._source.list_ids(self._suffix)

    async def get(self, item_id: str) -> TAsset:
        raw = await self._source.read(f"{item_id}{self._suffix}")
        key = (item_id, self._source.identity(raw))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._codec.decode(item_id, raw)
        self._cache[key] = value
        return value


__all__ = ["AssetIndex"]
