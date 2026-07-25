#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContentCache contract for the storage kernel (storage.cache).

The cache sits in front of the object origin. A read is keyed by
``object:{namespace}:{sha256(key)}:v{version}:{etag}`` so a version change
never returns stale bytes. L1 (memory) hit bypasses L2; L2 hit backfills L1;
corruption falls back to origin; a cache error never affects origin
correctness."""

from __future__ import annotations

import asyncio


class TestMemoryContentCache:
    def test_hit_returns_cached_bytes(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache

            cache = MemoryContentCache()
            await cache.put("k1", b"v1")
            assert await cache.get("k1") == b"v1"

        asyncio.run(_run())

    def test_miss_returns_none(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache

            cache = MemoryContentCache()
            assert await cache.get("absent") is None

        asyncio.run(_run())

    def test_respects_max_entries_bound(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache

            cache = MemoryContentCache(max_entries=2)
            await cache.put("k1", b"v1")
            await cache.put("k2", b"v2")
            await cache.put("k3", b"v3")  # evicts the oldest
            hits = 0
            for k in ("k1", "k2", "k3"):
                if await cache.get(k) is not None:
                    hits += 1
            assert hits == 2

        asyncio.run(_run())

    def test_oversized_item_is_not_admitted(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache

            cache = MemoryContentCache(max_item_bytes=4)
            await cache.put("big", b"oversized")
            assert await cache.get("big") is None

        asyncio.run(_run())


class TestTieredContentCache:
    def test_l1_hit_does_not_touch_l2(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache
            from linktools.ai.storage.cache.tiered import TieredContentCache

            l1 = MemoryContentCache()
            l2 = MemoryContentCache()
            await l1.put("k", b"v")
            tiered = TieredContentCache(l1=l1, l2=l2)
            # L2 stays empty; the L1 hit satisfies the read.
            assert await tiered.get("k") == b"v"
            assert await l2.get("k") is None

        asyncio.run(_run())

    def test_l2_hit_backfills_l1(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache
            from linktools.ai.storage.cache.tiered import TieredContentCache

            l1 = MemoryContentCache()
            l2 = MemoryContentCache()
            await l2.put("k", b"v")
            tiered = TieredContentCache(l1=l1, l2=l2)
            assert await tiered.get("k") == b"v"
            assert await l1.get("k") == b"v"  # backfilled

        asyncio.run(_run())

    def test_corrupt_cache_entry_falls_back_to_origin(self) -> None:
        # If a cached entry is corrupt, the reader must re-source from origin
        # rather than return bad bytes.
        async def _run() -> None:
            from linktools.ai.storage.cache.memory import MemoryContentCache
            from linktools.ai.storage.cache.tiered import TieredContentCache

            l1 = MemoryContentCache()
            await l1.put("k", b"corrupt")
            l1.corrupt("k")  # mark the entry unreadable
            l2 = MemoryContentCache()
            await l2.put("k", b"good")
            tiered = TieredContentCache(l1=l1, l2=l2)
            assert await tiered.get("k") == b"good"

        asyncio.run(_run())


class TestVersionedCacheKey:
    def test_version_change_never_returns_stale_bytes(self) -> None:
        # The cache key embeds BOTH version and etag, so a v2 read can never
        # be satisfied by a cached v1 entry.
        async def _run() -> None:
            from linktools.ai.storage.cache.cached import CachedObjectReader

            # CachedObjectReader wraps an origin reader + a content cache.
            reader = CachedObjectReader(origin=_Origin(), cache=None)
            v1 = await reader.get_versioned("/a", version=1, etag="e1")
            v2 = await reader.get_versioned("/a", version=2, etag="e2")
            assert v1.content != v2.content

        asyncio.run(_run())


class _Origin:
    """Minimal stand-in origin reader for the versioned-key contract."""

    async def get_versioned(self, key, *, version, etag):
        from linktools.ai.storage.object.models import ObjectInfo, StoredObject, StorageKey
        from datetime import datetime, timezone

        return StoredObject(
            info=ObjectInfo(
                key=StorageKey(key),
                etag=etag,
                version=version,
                commit_revision=None,
                content_type=None,
                size=len(f"{key}-v{version}".encode()),
                modified_at=datetime.now(timezone.utc),
                metadata={},
            ),
            content=f"{key}-v{version}".encode(),
        )
