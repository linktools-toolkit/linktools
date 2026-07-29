"""Specification storage assembled from independent storage capabilities."""

from typing import Protocol

from .cache import ContentCache, ContentCacheKey
from .composition import (
    StorageAdapter,
    StorageCacheAdapter,
    StorageComposition,
)
from .multi import StorageLayer, StorageReader, StorageWriter
from .revision import (
    ChangeSource,
    Revision,
    RevisionSource,
)
from .document import SpecDocument, SpecDocumentChange, SpecDocumentInfo


class SpecReader(
    StorageReader[str, SpecDocument, SpecDocumentInfo],
    Protocol,
):
    pass


class SpecWriter(StorageWriter[str, SpecDocument], Protocol):
    pass


class SpecStorageAdapter(
    StorageAdapter[str, SpecDocument, SpecDocumentInfo, SpecDocumentChange],
    StorageCacheAdapter[str, SpecDocument, SpecDocumentInfo],
):
    def info_key(self, info: SpecDocumentInfo) -> str:
        return info.path

    def change_key(self, change: SpecDocumentChange) -> str:
        return change.path

    def change_value(
        self,
        change: SpecDocumentChange,
    ) -> SpecDocumentInfo | None:
        return change.info

    def value_info(self, value: SpecDocument) -> SpecDocumentInfo:
        return value.info

    def cache_key(
        self,
        key: str,
        info: SpecDocumentInfo,
    ) -> ContentCacheKey:
        return key, info.version, info.etag

    def cache_content(self, value: SpecDocument) -> bytes:
        return value.content

    def from_cache(
        self,
        info: SpecDocumentInfo,
        content: bytes,
    ) -> SpecDocument:
        return SpecDocument(info, content)


class SpecStore:
    """Compose spec reads with optional writes, overlays, revision, delta, and cache."""

    def __init__(
        self,
        reader: SpecReader,
        *,
        writer: SpecWriter | None = None,
        overlays: tuple[
            StorageLayer[str, SpecDocument, SpecDocumentInfo],
            ...,
        ] = (),
        revision: RevisionSource | None = None,
        changes: ChangeSource[SpecDocumentChange] | None = None,
        content_cache: ContentCache | None = None,
    ) -> None:
        adapter = SpecStorageAdapter()
        self._storage = StorageComposition(
            primary=reader,
            writer=writer,
            overlays=overlays,
            revision=revision,
            changes=changes,
            cache=content_cache,
            adapter=adapter,
            cache_adapter=adapter,
        )

    @property
    def backend(self) -> SpecReader:
        return self._storage.primary

    @property
    def writer(self) -> SpecWriter | None:
        return self._storage.writer

    async def initialize_storage(self, *args: object) -> None:
        await self._storage.initialize(*args)

    async def stat(self, path: str) -> SpecDocumentInfo | None:
        return await self._storage.get_info(path)

    async def list_active(
        self,
        kind: str | None = None,
        *,
        preload: bool = False,
    ) -> tuple[str, ...]:
        state = await self._storage.refresh(preload=preload)
        if state is None:
            return ()
        return tuple(
            sorted(
                path
                for path, info in state.entries.items()
                if info.active and (kind is None or info.kind == kind)
            )
        )

    async def list_info(
        self,
        *,
        preload: bool = False,
    ) -> tuple[SpecDocumentInfo, ...]:
        return await self._storage.list_info(preload=preload)

    async def current_revision(self) -> Revision | None:
        return await self._storage.current_revision()

    async def get(self, path: str) -> SpecDocument | None:
        return await self._storage.get(path)

    async def put(self, document: SpecDocument) -> SpecDocument:
        return await self._storage.require_writer().put(document)

    async def delete(self, path: str) -> None:
        await self._storage.require_writer().delete(path)

    async def reset(self, documents: tuple[SpecDocument, ...]) -> None:
        await self._storage.require_writer().reset(documents)


__all__ = ["SpecReader", "SpecStore", "SpecWriter"]
