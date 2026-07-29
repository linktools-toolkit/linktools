import pytest

from linktools.ai.spec.cache import (
    FilesystemContentCache,
    MemoryContentCache,
    TieredContentCache,
)


@pytest.mark.asyncio
async def test_tiered_cache_is_storage_generic(tmp_path):
    key = ("document", 1, "etag")
    cache = TieredContentCache(
        MemoryContentCache(max_bytes=100),
        FilesystemContentCache(tmp_path, max_bytes=100),
    )
    await cache.put(key, b"content")
    assert await cache.get(key) == b"content"


@pytest.mark.asyncio
async def test_memory_cache_is_bounded_lru():
    cache = MemoryContentCache(max_bytes=3)
    await cache.put(("a", 1, ""), b"aa")
    await cache.put(("b", 1, ""), b"b")
    assert await cache.get(("a", 1, "")) == b"aa"
    await cache.put(("c", 1, ""), b"c")
    assert await cache.get(("b", 1, "")) is None
    assert await cache.get(("a", 1, "")) == b"aa"
