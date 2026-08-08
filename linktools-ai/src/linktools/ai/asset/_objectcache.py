#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Version-aware decoded-object cache over text asset content."""

import asyncio
from typing import Generic, Protocol, TypeVar

from linktools.core import environ

from ._content import AssetContent, AssetContentInfo

TAsset = TypeVar("TAsset")
_logger = environ.get_logger("ai.asset.objectcache")


class AssetCacheStore(Protocol):
    async def get(self, path: str) -> "AssetContent | None": ...

    async def list_info(self) -> "tuple[AssetContentInfo, ...]": ...


class AssetCacheCodec(Generic[TAsset], Protocol):
    def decode(self, item_id: str, raw: str) -> TAsset: ...


class AssetObjectCache(Generic[TAsset]):
    def __init__(
        self,
        store: AssetCacheStore,
        codec: AssetCacheCodec[TAsset],
        *,
        prefix: str,
        suffix: str,
        source_name: "str | None" = None,
    ) -> None:
        self._store = store
        self._codec = codec
        self._prefix = prefix.strip("/")
        self._suffix = suffix
        self._source_name = source_name or type(store).__name__
        self._cache: dict[tuple[str, int, str], TAsset] = {}
        self._inflight: dict[tuple[str, int, str], asyncio.Task[TAsset]] = {}
        self._lock = asyncio.Lock()

    @property
    def source_name(self) -> str:
        return self._source_name

    def _full_path(self, item_id: str) -> str:
        joined = f"{self._prefix}/{item_id}" if self._prefix else item_id
        return f"{joined}{self._suffix}"

    async def list_ids(self) -> "tuple[str, ...]":
        prefix = f"{self._prefix}/" if self._prefix else ""
        infos = await self._store.list_info()
        ids = [
            info.path.rsplit("/", 1)[-1][: -len(self._suffix)]
            for info in infos
            if info.path.startswith(prefix) and info.path.endswith(self._suffix)
        ]
        return tuple(sorted(dict.fromkeys(ids)))

    async def get(self, item_id: str) -> TAsset:
        content = await self._store.get(self._full_path(item_id))
        if content is None:
            raise KeyError(item_id)
        key = (item_id, content.info.version, content.info.etag)
        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                _logger.debug("asset cache hit: item=%s source=%s", item_id, self._source_name)
                return cached
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._decode(item_id, key, content))
                self._inflight[key] = task
                _logger.debug("asset cache miss: item=%s source=%s", item_id, self._source_name)
        return await asyncio.shield(task)

    async def _decode(
        self,
        item_id: str,
        key: "tuple[str, int, str]",
        content: AssetContent,
    ) -> TAsset:
        try:
            value = self._codec.decode(item_id, content.content.decode("utf-8"))
            async with self._lock:
                self._cache[key] = value
            return value
        finally:
            async with self._lock:
                self._inflight.pop(key, None)


__all__ = ["AssetCacheCodec", "AssetCacheStore", "AssetObjectCache"]
