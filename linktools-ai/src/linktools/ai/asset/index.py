#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content-identity keyed parsed asset index."""

from typing import Generic, TYPE_CHECKING, TypeVar

from .contracts import AssetCodec
from .source import AssetLoaderSource

if TYPE_CHECKING:
    from .contracts import AssetSource
    from .parsing import AssetLoader

T = TypeVar("T")


class AssetIndex(Generic[T]):
    def __init__(self, source: "AssetSource", codec: "AssetCodec[T]", *, suffix: str) -> None:
        self._suffix = suffix
        self._codec = codec
        self._source = source
        self._cache: "dict[tuple[str, str], T]" = {}

    @classmethod
    def source_from_loader(cls, loader: "AssetLoader") -> "AssetSource":
        return AssetLoaderSource(loader)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._source.list_ids(self._suffix)

    async def get(self, item_id: str) -> T:
        raw = await self._source.read(f"{item_id}{self._suffix}")
        key = (item_id, self._source.identity(raw))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._codec.decode(item_id, raw)
        self._cache[key] = value
        return value


__all__ = ["AssetIndex"]
