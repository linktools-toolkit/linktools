"""Specification storage assembled from independent storage features."""

from ..storage.cache import ContentCacheKey
from ..storage.composition import (
    StorageAdapter,
    StorageCacheAdapter,
    StorageComposition,
    StorageLayer,
)
from .document import SpecDocument, SpecDocumentInfo


class SpecStorageAdapter(
    StorageAdapter[str, SpecDocument, SpecDocumentInfo],
    StorageCacheAdapter[str, SpecDocument, SpecDocumentInfo],
):
    def info_key(self, info: SpecDocumentInfo) -> str:
        return info.path

    def value_info(self, value: SpecDocument) -> SpecDocumentInfo:
        return value.info

    def cache_key(self, key: str, info: SpecDocumentInfo) -> ContentCacheKey:
        # spec:{path}:{version}:{etag} -- content changes alter the etag, so a
        # stale cache entry can never be served against changed content.
        return f"spec:{key}:{info.version}:{info.etag}"

    def cache_content(self, value: SpecDocument) -> bytes:
        return value.content

    def from_cache(self, info: SpecDocumentInfo, content: bytes) -> SpecDocument:
        return SpecDocument(info, content)


class SpecStore:
    """Compose spec reads with optional writes, layers, and a content cache."""

    def __init__(
        self,
        primary,
        *,
        writer=None,
        layers: "tuple[StorageLayer[str, SpecDocument, SpecDocumentInfo], ...]" = (),
        cache=None,
    ) -> None:
        adapter = SpecStorageAdapter()
        self._storage = StorageComposition(
            primary,
            writer=writer,
            layers=layers,
            cache=cache,
            adapter=adapter,
            cache_adapter=adapter,
        )

    @property
    def writer(self):
        return self._storage.writer

    async def initialize_storage(self, *args: object) -> None:
        await self._storage.initialize(*args)

    async def stat(self, path: str) -> SpecDocumentInfo | None:
        state = await self._storage.refresh()
        return None if state is None else state.entries.get(path)

    async def list_active(
        self,
        kind: str | None = None,
        *,
        preload: bool = False,
    ) -> tuple[str, ...]:
        infos = await self._storage.list_info(preload=preload)
        return tuple(
            info.path
            for info in infos
            if info.active and (kind is None or info.kind == kind)
        )

    async def list_info(self, *, preload: bool = False) -> tuple[SpecDocumentInfo, ...]:
        return await self._storage.list_info(preload=preload)

    async def current_revision(self):
        return await self._storage.current_revision()

    async def get(self, path: str) -> SpecDocument | None:
        return await self._storage.get(path)

    async def get_many(self, paths: tuple[str, ...]) -> dict[str, SpecDocument]:
        return await self._storage.get_many(paths)

    async def put(self, document: SpecDocument) -> SpecDocument:
        return await self._storage.put(document)

    async def delete(self, path: str) -> None:
        await self._storage.delete(path)

    async def reset(self, documents: tuple[SpecDocument, ...]) -> None:
        await self._storage.reset(documents)


__all__ = ["SpecStore"]
