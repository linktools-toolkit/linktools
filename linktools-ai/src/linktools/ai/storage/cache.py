"""Generic best-effort caches for immutable versioned content."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, TypeAlias


ContentCacheKey: TypeAlias = tuple[str, int, str]


class ContentCache(Protocol):
    async def get(self, key: ContentCacheKey) -> bytes | None: ...
    async def put(self, key: ContentCacheKey, content: bytes) -> None: ...


class MemoryContentCache:
    """Bounded LRU content cache used as the first tier."""

    def __init__(self, *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._items: OrderedDict[ContentCacheKey, bytes] = OrderedDict()
        self._size = 0
        self._lock = asyncio.Lock()

    async def get(self, key: ContentCacheKey) -> bytes | None:
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


class FilesystemContentCache:
    """Bounded second-tier cache with one atomically replaced file per key."""

    def __init__(self, root: str | Path, *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._lock = asyncio.Lock()

    @staticmethod
    def _name(key: ContentCacheKey) -> str:
        return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()

    def _path(self, key: ContentCacheKey) -> Path:
        return self.root / self._name(key)

    async def get(self, key: ContentCacheKey) -> bytes | None:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            return None
        except OSError:
            return None

    async def put(self, key: ContentCacheKey, content: bytes) -> None:
        if len(content) > self.max_bytes:
            return
        async with self._lock:
            await asyncio.to_thread(self._put_sync, key, content)

    def _put_sync(self, key: ContentCacheKey, content: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
            os.replace(temporary, path)
            files = sorted(self.root.iterdir(), key=lambda item: item.stat().st_mtime_ns)
            total = sum(item.stat().st_size for item in files if item.is_file())
            for item in files:
                if total <= self.max_bytes:
                    break
                if item.is_file():
                    total -= item.stat().st_size
                    item.unlink()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class TieredContentCache:
    """L1/L2 cache composition; failures are misses and never origin errors."""

    def __init__(self, l1: ContentCache, l2: ContentCache | None = None) -> None:
        self.l1 = l1
        self.l2 = l2

    async def get(self, key: ContentCacheKey) -> bytes | None:
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


__all__ = [
    "ContentCache",
    "ContentCacheKey",
    "FilesystemContentCache",
    "MemoryContentCache",
    "TieredContentCache",
]
