#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Per-document parsed-object cache over a spec store.

``SpecObjectCache`` caches decoded objects keyed by ``(item_id, version, etag)``.
``get`` reads a document once via ``store.get`` and obtains content, version,
and etag together -- there is no separate revision probe before the read (the
old design queried revision then re-read). When document A changes, only A's
cache entries age out; B and C keep their parsed values.

Inflight reads of the same key are coalesced so concurrent callers share one
decode. The cache is keyed by content identity, so a stale cache entry can
never be served against changed content."""


import asyncio
from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")


class SpecCacheStore(Protocol):
    async def get(self, path: str) -> Any: ...

    async def list_info(self) -> "tuple[Any, ...]": ...


class SpecCacheCodec(Protocol, Generic[T]):
    def decode(self, item_id: str, raw: str) -> T: ...


class SpecObjectCache(Generic[T]):
    def __init__(
        self,
        store: SpecCacheStore,
        codec: "SpecCacheCodec[T]",
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
        self._inflight: "dict[tuple[str, int, str], 'asyncio.Future[T]']" = {}
        self._ids: "tuple[str, ...] | None" = None

    @property
    def source_name(self) -> str:
        return self._source_name

    def _full_path(self, item_id: str) -> str:
        joined = (
            f"{self._prefix}/{item_id}" if self._prefix else item_id
        )
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
        document = await self._store.get(path)
        if document is None:
            raise KeyError(item_id)
        version = document.info.version
        etag = document.info.etag
        key = (item_id, version, etag)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        future = self._inflight.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
            try:
                value = self._codec.decode(
                    item_id, document.content.decode("utf-8")
                )
                self._cache[key] = value
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                self._inflight.pop(key, None)
        return await future


__all__ = ["SpecCacheCodec", "SpecCacheStore", "SpecObjectCache"]
