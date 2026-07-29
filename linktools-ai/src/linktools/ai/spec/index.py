"""Revision-aware index shared by every specification domain."""

from __future__ import annotations

from typing import Generic, TypeVar

from .revision import RevisionCache
from .contracts import SpecCodec, SpecSource
from .parsing import SpecLoader
from .source import SpecLoaderSource

T = TypeVar("T")


class SpecIndex(Generic[T]):
    def __init__(
        self,
        source: SpecSource,
        codec: SpecCodec[T],
        *,
        suffix: str,
        source_name: str | None = None,
    ) -> None:
        self._cache = RevisionCache(
            source,
            codec,
            suffix=suffix,
            source_name=source_name,
        )

    @classmethod
    def source_from_loader(cls, loader: SpecLoader) -> SpecSource:
        return SpecLoaderSource(loader)

    async def list_ids(self) -> tuple[str, ...]:
        return await self._cache.list_ids()

    async def get(self, item_id: str) -> T:
        return await self._cache.get(item_id)


__all__ = ["SpecIndex"]
