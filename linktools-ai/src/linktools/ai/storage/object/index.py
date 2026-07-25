#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RevisionedObjectIndex: a revision-gated in-memory metadata cache over one
prefix of a ``RevisionedObjectReader`` source, so a point ``stat`` does not
re-scan the source on every call.

``ensure_fresh()`` compares the source's current revision against the
revision the index was last built at; on a match it is a no-op, on a mismatch
it refreshes under a lock (re-checking the revision once inside the lock, so
concurrent callers that lost the race to acquire it don't refresh twice) by
paging the full prefix listing and atomically swapping in the new entry map.

The index only ever caches :class:`ObjectInfo` (metadata), never content, and
never serves metadata it cannot prove is current: if the revision check
itself fails, ``stat`` bypasses the index and asks the source directly rather
than returning a possibly-stale cached entry. ``list`` always bypasses the
index -- it is a point-read accelerator only, not a listing cache (a listing
cache reintroduces exactly the per-Catalog-get full-scan cost it exists to
avoid)."""

from __future__ import annotations

import asyncio

from .models import Depth, ObjectInfo, ObjectPage, StorageKey


class RevisionedObjectIndex:
    def __init__(self, source, *, prefix: StorageKey, depth: "Depth" = Depth.INFINITY) -> None:
        self._source = source
        self._prefix = prefix
        self._depth = depth
        self._lock = asyncio.Lock()
        self._cached_revision: "str | None" = None
        self._entries: "dict[str, ObjectInfo]" = {}

    async def ensure_fresh(self) -> bool:
        """Returns True when the index is known-current (revision matched, or
        a refresh just succeeded); False when freshness could not be proven
        (the revision check itself failed) -- callers must not trust cached
        entries in that case."""
        try:
            current = await self._source.revision()
        except Exception:
            return False
        if current == self._cached_revision:
            return True
        async with self._lock:
            try:
                current = await self._source.revision()
            except Exception:
                return False
            if current == self._cached_revision:
                return True
            entries: "dict[str, ObjectInfo]" = {}
            cursor: "str | None" = None
            while True:
                page: ObjectPage = await self._source.list(
                    self._prefix, depth=self._depth, limit=200, cursor=cursor
                )
                for info in page.items:
                    entries[info.key.value] = info
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            self._entries = entries
            self._cached_revision = current
            return True

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        if not await self.ensure_fresh():
            return await self._source.stat(key)
        return self._entries.get(key.value)

    async def list(self, prefix: StorageKey, **kwargs) -> ObjectPage:
        return await self._source.list(prefix, **kwargs)


__all__: "list[str]" = ["RevisionedObjectIndex"]
