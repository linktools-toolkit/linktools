#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic best-effort caches for immutable versioned content.

The cache key is a stable domain-identity string (e.g.
``spec:{path}:{version}:{etag}``); filesystem cache file names are its SHA-256.
All caches are best-effort: any read/write error is swallowed and treated as a
miss/failed write, never propagated as the origin's result.

``contains_many`` answers whether keys already exist WITHOUT reading content --
preload uses it to avoid reading whole files just to decide a cache miss."""


import asyncio
import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, TypeAlias


ContentCacheKey: TypeAlias = str


class ContentCache(Protocol):
    async def get(self, key: ContentCacheKey) -> "bytes | None": ...

    async def put(self, key: ContentCacheKey, content: bytes) -> None: ...

    async def contains_many(
        self,
        keys: "tuple[ContentCacheKey, ...]",
    ) -> "frozenset[ContentCacheKey]": ...


async def read_cache(cache: "ContentCache | None", key: ContentCacheKey) -> "bytes | None":
    if cache is None:
        return None
    try:
        return await cache.get(key)
    except Exception:
        return None


async def write_cache(
    cache: "ContentCache | None",
    key: ContentCacheKey,
    content: bytes,
) -> bool:
    if cache is None:
        return False
    try:
        await cache.put(key, content)
        return True
    except Exception:
        return False


async def contains_many(
    cache: "ContentCache | None",
    keys: "tuple[ContentCacheKey, ...]",
) -> "frozenset[ContentCacheKey]":
    if cache is None or not keys:
        return frozenset()
    try:
        return await cache.contains_many(keys)
    except Exception:
        return frozenset()


class MemoryContentCache:
    """Bounded LRU content cache. ``contains_many`` checks all keys under one
    lock and does NOT touch LRU order (existence is not an access)."""

    def __init__(self, *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._items: "OrderedDict[ContentCacheKey, bytes]" = OrderedDict()
        self._size = 0
        self._lock = asyncio.Lock()

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        async with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        if len(content) > self.max_bytes:
            return
        async with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._size -= len(previous)
            self._items[key] = content
            self._size += len(content)
            while self._size > self.max_bytes and self._items:
                _, removed = self._items.popitem(last=False)
                self._size -= len(removed)

    async def contains_many(
        self,
        keys: "tuple[ContentCacheKey, ...]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        async with self._lock:
            return frozenset(key for key in keys if key in self._items)


class FilesystemContentCache:
    """Bounded second-tier cache. Reads happen OUTSIDE the lock (a slow file
    never blocks other keys); only the LRU metadata update takes the short
    lock. ``contains_many`` checks all target paths in ONE ``to_thread`` call,
    reading no content and not scanning the whole root.

    The eviction index is built once, on first write. Reads of an un-indexed
    root simply miss (a cache, not the source of truth)."""

    def __init__(self, root: "str | Path", *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._lock = asyncio.Lock()
        self._indexed = False
        self._entries: "dict[str, tuple[int, int]]" = {}
        self._total = 0
        self._clock = 0

    @staticmethod
    def _name(key: ContentCacheKey) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _path(self, key: ContentCacheKey) -> Path:
        return self.root / self._name(key)

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        path = self._path(key)
        try:
            value = await asyncio.to_thread(path.read_bytes)
        except (FileNotFoundError, OSError):
            await self._forget(path.name)
            return None
        # Metadata touch under a short lock; the read itself was unlocked.
        async with self._lock:
            self._clock += 1
            self._entries[path.name] = (len(value), self._clock)
        return value

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        if len(content) > self.max_bytes:
            return
        async with self._lock:
            await self._ensure_index()
            await asyncio.to_thread(self._put_sync, key, content)
            name = self._name(key)
            previous = self._entries.get(name)
            if previous is not None:
                self._total -= previous[0]
            self._clock += 1
            self._entries[name] = (len(content), self._clock)
            self._total += len(content)
            while self._total > self.max_bytes and self._entries:
                victim = min(self._entries, key=lambda item: self._entries[item][1])
                size, _ = self._entries.pop(victim)
                self._total -= size
                await asyncio.to_thread((self.root / victim).unlink, missing_ok=True)

    async def contains_many(
        self,
        keys: "tuple[ContentCacheKey, ...]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        paths = {self._name(key): key for key in keys}

        def _existing() -> "set[str]":
            return {name for name in paths if (self.root / name).exists()}

        names = await asyncio.to_thread(_existing)
        return frozenset(paths[name] for name in names)

    async def _ensure_index(self) -> None:
        if self._indexed:
            return
        self._indexed = True
        try:
            entries = await asyncio.to_thread(self._scan_index)
        except OSError:
            self._entries.clear()
            self._total = 0
            return
        for name, size in entries:
            self._clock += 1
            self._entries[name] = (size, self._clock)
            self._total += size

    def _scan_index(self) -> "tuple[tuple[str, int], ...]":
        if not self.root.exists():
            return ()
        return tuple(
            (item.name, item.stat().st_size)
            for item in self.root.iterdir()
            if item.is_file()
        )

    async def _forget(self, name: str) -> None:
        async with self._lock:
            previous = self._entries.pop(name, None)
            if previous is not None:
                self._total -= previous[0]

    def _put_sync(self, key: ContentCacheKey, content: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class TieredContentCache:
    """L1/L2 cache composition. ``get`` promotes an L2 hit into L1; failures are
    misses. ``contains_many`` is satisfied if EITHER tier holds the key, and
    does NOT promote L2 content into L1 (existence is not an access)."""

    def __init__(self, l1: ContentCache, l2: "ContentCache | None" = None) -> None:
        self.l1 = l1
        self.l2 = l2

    async def get(self, key: ContentCacheKey) -> "bytes | None":
        try:
            value = await self.l1.get(key)
            if value is not None:
                return value
            if self.l2 is None:
                return None
            value = await self.l2.get(key)
            if value is not None:
                await self.l1.put(key, value)
            return value
        except Exception:
            return None

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        for cache in (self.l1, self.l2):
            if cache is None:
                continue
            try:
                await cache.put(key, content)
            except Exception:
                continue

    async def contains_many(
        self,
        keys: "tuple[ContentCacheKey, ...]",
    ) -> "frozenset[ContentCacheKey]":
        if not keys:
            return frozenset()
        present = await self.l1.contains_many(keys)
        if self.l2 is None:
            return present
        missing = tuple(key for key in keys if key not in present)
        if missing:
            present |= await self.l2.contains_many(missing)
        return present


__all__ = [
    "ContentCache",
    "ContentCacheKey",
    "FilesystemContentCache",
    "MemoryContentCache",
    "TieredContentCache",
    "contains_many",
    "read_cache",
    "write_cache",
]
