#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CachedObjectReader / CachedObjectHistoryReader: a ContentCache in front of
an object origin, keyed so a version change can never return stale bytes.

Cache key: ``object:{namespace}:{sha256(key)}:v{version}:{etag}`` -- version
AND etag are both embedded, so a v2 read can never be satisfied by a v1 (or a
same-version-different-etag) cache entry. ``CachedObjectHistoryReader`` uses
the SAME key format, so a given object version's content is cached exactly
once regardless of whether it was reached through the current-state reader or
the history reader.

When ``cache`` is None (or, for the current-state reader, when no ``index``
is wired), reads pass straight through to the origin -- caching is optional
infrastructure a caller can wire in later without changing this class."""

from __future__ import annotations

from hashlib import sha256

from ..object.models import StorageKey, StoredObject


def _cache_key(namespace: str, key: str, *, version: int, etag: str) -> str:
    return f"object:{namespace}:{sha256(key.encode('utf-8')).hexdigest()}:v{version}:{etag}"


class CachedObjectReader:
    def __init__(self, *, origin, cache, index=None, namespace: str = "default") -> None:
        self._origin = origin
        self._cache = cache
        self._index = index
        self._namespace = namespace

    async def get_versioned(self, key: str, *, version: int, etag: str) -> StoredObject:
        if self._cache is None:
            return await self._origin.get_versioned(key, version=version, etag=etag)
        cache_key = _cache_key(self._namespace, key, version=version, etag=etag)
        cached_bytes = await self._cache.get(cache_key)
        if cached_bytes is not None and self._index is not None:
            info = await self._index.stat(StorageKey(key))
            if info is not None and info.version == version and info.etag == etag:
                return StoredObject(info=info, content=cached_bytes)
        obj = await self._origin.get_versioned(key, version=version, etag=etag)
        await self._cache.put(cache_key, obj.content)
        return obj


class CachedObjectHistoryReader:
    """Reads a specific past version through the same versioned cache key
    ``CachedObjectReader`` uses, so history and current-state reads of the
    same object version share one cached copy."""

    def __init__(self, *, origin, cache, namespace: str = "default") -> None:
        self._origin = origin
        self._cache = cache
        self._namespace = namespace

    async def get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        stored = await self._origin.raw_get_version(key, version)
        if stored is None or self._cache is None:
            return stored
        cache_key = _cache_key(self._namespace, key.value, version=version, etag=stored.info.etag)
        cached_bytes = await self._cache.get(cache_key)
        if cached_bytes is not None:
            return StoredObject(info=stored.info, content=cached_bytes)
        await self._cache.put(cache_key, stored.content)
        return stored


__all__: "list[str]" = ["CachedObjectReader", "CachedObjectHistoryReader"]
