#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asset storage composed from the generic storage kernel."""

from typing import Any, TYPE_CHECKING

from ..storage.composition import StorageAdapter, StorageCacheAdapter, StorageComposition, StorageLayer
from ..storage.versioning import VersionedStorage, VersionSummary
from .content import AssetContent, AssetContentInfo

if TYPE_CHECKING:
    from ..storage.cache import ContentCache, ContentCacheKey
    from ..storage.multi import StorageReader, StorageWriter
    from ..storage.revision import RevisionSource


class AssetStorageAdapter(
    StorageAdapter[str, AssetContent, AssetContentInfo],
    StorageCacheAdapter[str, AssetContent, AssetContentInfo],
):
    def info_key(self, info: AssetContentInfo) -> str:
        return info.path

    def value_info(self, value: AssetContent) -> AssetContentInfo:
        return value.info

    def cache_key(self, key: str, info: AssetContentInfo) -> "ContentCacheKey":
        return f"asset:{key}:{info.version}:{info.etag}"

    def cache_content(self, value: AssetContent) -> bytes:
        return value.content

    def from_cache(self, info: AssetContentInfo, content: bytes) -> AssetContent:
        return AssetContent(info, content)


class AssetStore:
    def __init__(
        self,
        primary: "StorageReader[str, AssetContent, AssetContentInfo]",
        *,
        writer: "StorageWriter[str, AssetContent, Any] | None" = None,
        layers: "tuple[StorageLayer[str, AssetContent, AssetContentInfo], ...]" = (),
        cache: "ContentCache | None" = None,
        revision_source: "RevisionSource | None" = None,
    ) -> None:
        adapter = AssetStorageAdapter()
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
    def writer(self) -> "StorageWriter[str, AssetContent, Any] | None":
        return self._storage.writer

    async def initialize_storage(self, *args: object) -> None:
        await self._storage.initialize(*args)

    async def stat(self, path: str) -> "AssetContentInfo | None":
        state = await self._storage.refresh()
        return None if state is None else state.entries.get(path)

    async def list_active(self, kind: "str | None" = None, *, preload: bool = False) -> "tuple[str, ...]":
        return tuple(
            info.path
            for info in await self._storage.list_info(preload=preload)
            if info.active and (kind is None or info.kind == kind)
        )

    async def list_info(self, *, preload: bool = False) -> "tuple[AssetContentInfo, ...]":
        return await self._storage.list_info(preload=preload)

    async def list_info_with_owners(self, *, preload: bool = False) -> "tuple[tuple[AssetContentInfo, int], ...]":
        return await self._storage.list_info_with_owners(preload=preload)

    async def current_revision(self) -> "int | str":
        return await self._storage.current_revision()

    async def get(self, path: str) -> "AssetContent | None":
        return await self._storage.get(path)

    async def get_many(self, paths: "tuple[str, ...]") -> "dict[str, AssetContent]":
        return await self._storage.get_many(paths)

    async def put(self, content: AssetContent) -> AssetContent:
        return await self._storage.put(content)

    async def delete(self, path: str) -> None:
        await self._storage.delete(path)

    async def reset(self, contents: "tuple[AssetContent, ...]") -> None:
        await self._storage.reset(contents)

    async def apply_batch(self, puts: "tuple[AssetContent, ...]", deletes: "tuple[str, ...]") -> None:
        await self._storage.apply_batch(puts, deletes)

    def _versioned_primary(self) -> "VersionedStorage[object, str, AssetContent] | None":
        primary = self._storage.primary
        return primary if isinstance(primary, VersionedStorage) else None

    async def list_versions(self, path: str) -> "tuple[VersionSummary, ...]":
        primary = self._versioned_primary()
        return () if primary is None else await primary.list_versions(path)

    async def get_at_revision(self, path: str, revision: object) -> "AssetContent | None":
        primary = self._versioned_primary()
        return None if primary is None else await primary.get_at_revision(path, revision)

    async def get_at_version(self, path: str, version: int) -> "AssetContent | None":
        primary = self._versioned_primary()
        return None if primary is None else await primary.get_at_version(path, version)


__all__ = ["AssetStore"]
