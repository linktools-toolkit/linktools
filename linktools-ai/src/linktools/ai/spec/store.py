#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Specification storage assembled from independent storage features."""

from typing import TYPE_CHECKING
from ..storage.composition import (
    StorageAdapter,
    StorageCacheAdapter,
    StorageComposition,
    StorageLayer,
)
from ..storage.versioning import VersionedStorage, VersionSummary
from .document import SpecDocument, SpecDocumentInfo

if TYPE_CHECKING:
    from ..storage.cache import ContentCacheKey
    from ..storage.cache import ContentCache
    from ..storage.multi import StorageReader, StorageWriter
    from ..storage.revision import RevisionSource


class SpecStorageAdapter(
    StorageAdapter[str, SpecDocument, SpecDocumentInfo],
    StorageCacheAdapter[str, SpecDocument, SpecDocumentInfo],
):
    def info_key(self, info: SpecDocumentInfo) -> str:
        return info.path

    def value_info(self, value: SpecDocument) -> SpecDocumentInfo:
        return value.info

    def cache_key(self, key: str, info: SpecDocumentInfo) -> "ContentCacheKey":
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
        primary: "StorageReader[str, SpecDocument, SpecDocumentInfo]",
        *,
        writer: "StorageWriter[str, SpecDocument] | None" = None,
        layers: "tuple[StorageLayer[str, SpecDocument, SpecDocumentInfo], ...]" = (),
        cache: "ContentCache | None" = None,
        revision_source: "RevisionSource | None" = None,
    ) -> None:
        adapter = SpecStorageAdapter()
        self._storage = StorageComposition(
            primary,
            writer=writer,
            layers=layers,
            cache=cache,
            adapter=adapter,
            cache_adapter=adapter,
            revision_source=revision_source,
        )

    @property
    def writer(self) -> "StorageWriter[str, SpecDocument] | None":
        return self._storage.writer

    async def initialize_storage(self, *args: object) -> None:
        await self._storage.initialize(*args)

    async def stat(self, path: str) -> "SpecDocumentInfo | None":
        state = await self._storage.refresh()
        return None if state is None else state.entries.get(path)

    async def list_active(
        self,
        kind: "str | None" = None,
        *,
        preload: bool = False,
    ) -> "tuple[str, ...]":
        infos = await self._storage.list_info(preload=preload)
        return tuple(
            info.path
            for info in infos
            if info.active and (kind is None or info.kind == kind)
        )

    async def list_info(
        self, *, preload: bool = False
    ) -> "tuple[SpecDocumentInfo, ...]":
        return await self._storage.list_info(preload=preload)

    async def list_info_with_owners(
        self, *, preload: bool = False
    ) -> "tuple[tuple[SpecDocumentInfo, int], ...]":
        """list_info paired with each entry's owning layer index (0 = primary,
        >0 = a fallback layer). Use this to separate primary-managed entries
        from layer-provided ones (e.g. DB-customized vs builtin-default)."""
        return await self._storage.list_info_with_owners(preload=preload)

    async def current_revision(self) -> "int | str":
        return await self._storage.current_revision()

    async def get(self, path: str) -> "SpecDocument | None":
        return await self._storage.get(path)

    async def get_many(self, paths: "tuple[str, ...]") -> "dict[str, SpecDocument]":
        return await self._storage.get_many(paths)

    async def put(self, document: SpecDocument) -> SpecDocument:
        return await self._storage.put(document)

    async def delete(self, path: str) -> None:
        await self._storage.delete(path)

    async def reset(self, documents: "tuple[SpecDocument, ...]") -> None:
        await self._storage.reset(documents)

    async def apply_batch(
        self,
        puts: "tuple[SpecDocument, ...]",
        deletes: "tuple[str, ...]",
    ) -> None:
        await self._storage.apply_batch(puts, deletes)

    # ---- version history (optional backend capability) -----------------

    @property
    def _versioned_primary(
        self,
    ) -> "VersionedStorage[object, str, SpecDocument] | None":
        primary = self._storage.primary
        return primary if isinstance(primary, VersionedStorage) else None

    async def list_versions(self, path: str) -> "tuple[VersionSummary, ...]":
        """A path's history, newest first. Empty when the primary backend
        keeps no change log (e.g. a local directory backend)."""
        primary = self._versioned_primary
        return () if primary is None else await primary.list_versions(path)

    async def get_at_revision(
        self, path: str, revision: object
    ) -> "SpecDocument | None":
        """The version of ``path`` in effect at ``revision``. None when the
        primary backend keeps no change log, or the path had no content at
        that revision."""
        primary = self._versioned_primary
        return (
            None if primary is None else await primary.get_at_revision(path, revision)
        )

    async def get_at_version(self, path: str, version: int) -> "SpecDocument | None":
        """The record of ``path`` carrying the given declared ``version``
        number (e.g. ``SpecDocumentInfo.version``), not a history ordinal.
        None when the primary backend keeps no change log, no record of
        ``path`` carries that version, or the matching record is a
        deletion."""
        primary = self._versioned_primary
        return None if primary is None else await primary.get_at_version(path, version)


__all__ = ["SpecStore"]
