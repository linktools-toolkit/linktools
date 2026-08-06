#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Version-aware decoded-object cache over an asset store."""

import asyncio
from typing import Any, Generic, Protocol, TypeVar

from linktools.core import environ

T = TypeVar("T")
logger = environ.get_logger("ai.asset.cache")


class AssetCacheStore(Protocol):
    async def get(self, path: str) -> Any: ...

    async def list_info(self) -> "tuple[Any, ...]": ...


class AssetCacheCodec(Protocol, Generic[T]):
    def decode(self, item_id: str, raw: str) -> T: ...


class AssetObjectCache(Generic[T]):
    def __init__(
        self,
        store: AssetCacheStore,
        codec: "AssetCacheCodec[T]",
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
        self._cache: "dict[tuple[str, int, str], T]" = {}
        self._inflight: "dict[tuple[str, int, str], asyncio.Future[T]]" = {}
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

    async def get(self, item_id: str) -> T:
        path = self._full_path(item_id)
        async with self._lock:
            content = await self._store.get(path)
            if content is None:
                raise KeyError(item_id)
            key = (item_id, content.info.version, content.info.etag)
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("asset cache hit: item=%s source=%s", item_id, self._source_name)
                return cached
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._inflight[key] = future
                logger.debug("asset cache miss: item=%s source=%s", item_id, self._source_name)
                asyncio.create_task(self._decode(item_id, key, content, future))
        return await future

    async def _decode(
        self,
        item_id: str,
        key: "tuple[str, int, str]",
        content: Any,
        future: "asyncio.Future[T]",
    ) -> None:
        try:
            value = self._codec.decode(item_id, content.content.decode("utf-8"))
            async with self._lock:
                self._cache[key] = value
                self._inflight.pop(key, None)
            future.set_result(value)
        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
            future.set_exception(exc)


__all__ = ["AssetCacheCodec", "AssetCacheStore", "AssetObjectCache"]
