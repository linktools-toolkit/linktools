#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryContentCache: a bounded, asyncio-safe in-process ContentCache.

Bounded by ``max_entries`` and/or ``max_bytes`` (LRU eviction of the oldest
entry); an item over ``max_item_bytes`` is silently not admitted rather than
evicting everything else to make room for it. ``corrupt()`` is a test-only
seam (not part of the ContentCache Protocol) that simulates a damaged cache
entry: the checksum stored alongside the content lets :meth:`get` detect and
self-heal a corrupted entry (delete it, report a miss) instead of ever
handing back bad bytes."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256


@dataclass
class _Entry:
    content: bytes
    checksum: str


class MemoryContentCache:
    def __init__(
        self,
        *,
        max_entries: "int | None" = None,
        max_bytes: "int | None" = None,
        max_item_bytes: "int | None" = None,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._max_item_bytes = max_item_bytes
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._total_bytes = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> "bytes | None":
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.checksum != sha256(entry.content).hexdigest():
                self._remove(key)
                return None
            self._entries.move_to_end(key)
            return entry.content

    async def put(self, key: str, content: bytes) -> None:
        if self._max_item_bytes is not None and len(content) > self._max_item_bytes:
            return
        async with self._lock:
            self._remove(key)
            self._entries[key] = _Entry(content=content, checksum=sha256(content).hexdigest())
            self._total_bytes += len(content)
            self._enforce_bounds()

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._remove(key)

    def corrupt(self, key: str) -> None:
        """Test-only: flip the stored checksum so the next :meth:`get`
        detects corruption and self-heals (delete + miss)."""
        entry = self._entries.get(key)
        if entry is not None:
            entry.checksum = "corrupted"

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._total_bytes -= len(entry.content)

    def _enforce_bounds(self) -> None:
        while self._max_entries is not None and len(self._entries) > self._max_entries:
            self._evict_oldest()
        while self._max_bytes is not None and self._total_bytes > self._max_bytes:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        if not self._entries:
            return
        _, entry = self._entries.popitem(last=False)
        self._total_bytes -= len(entry.content)


__all__: "list[str]" = ["MemoryContentCache"]
