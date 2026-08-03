#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Content-addressed parsed index shared by every specification domain."""

from typing import Generic, TypeVar
from .contracts import SpecCodec
from .source import SpecLoaderSource

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import SpecSource
    from .parsing import SpecLoader

T = TypeVar("T")


class SpecIndex(Generic[T]):
    """A parsed-object cache over a spec source. A change in one document
    re-parses only that document; the others keep their cached values."""

    def __init__(
        self,
        source: "SpecSource",
        codec: "SpecCodec[T]",
        *,
        suffix: str,
    ) -> None:
        self._suffix = suffix
        self._codec = codec
        self._source = source
        self._cache: "dict[tuple[str, str], T]" = {}

    @classmethod
    def source_from_loader(cls, loader: "SpecLoader") -> "SpecSource":
        return SpecLoaderSource(loader)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._source.list_ids(self._suffix)

    async def get(self, item_id: str) -> T:
        raw = await self._source.read(f"{item_id}{self._suffix}")
        # Key by (item_id, content identity). A changed document re-parses; an
        # unchanged one returns the cached object.
        key = (item_id, self._source.identity(raw))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._codec.decode(item_id, raw)
        self._cache[key] = value
        return value


__all__ = ["SpecIndex"]
