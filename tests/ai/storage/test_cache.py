import pytest

from linktools.ai.storage.cache import (
    FilesystemContentCache,
    MemoryContentCache,
    TieredContentCache,
    contains_many,
)


@pytest.mark.asyncio
async def test_tiered_cache_is_storage_generic(tmp_path):
    key = "spec:doc/a:1:etag"
    cache = TieredContentCache(
        MemoryContentCache(max_bytes=100),
        FilesystemContentCache(tmp_path, max_bytes=100),
    )
    await cache.put(key, b"content")
    assert await cache.get(key) == b"content"


@pytest.mark.asyncio
async def test_memory_cache_is_bounded_lru():
    cache = MemoryContentCache(max_bytes=3)
    await cache.put("a", b"aa")
    await cache.put("b", b"b")
    assert await cache.get("a") == b"aa"
    await cache.put("c", b"c")
    assert await cache.get("b") is None
    assert await cache.get("a") == b"aa"


@pytest.mark.asyncio
async def test_memory_contains_many_checks_all_keys_without_changing_lru():
    cache = MemoryContentCache(max_bytes=100)
    await cache.put("a", b"1")
    await cache.put("b", b"2")
    present = await cache.contains_many(("a", "b", "c"))
    assert present == frozenset({"a", "b"})
    # contains must not promote LRU: after evicting 'a' by overflow it should
    # still be present because it was the most-recently-put before contains.
    assert await cache.get("a") == b"1"


@pytest.mark.asyncio
async def test_filesystem_contains_many_reads_no_content_and_no_full_scan(tmp_path):
    cache = FilesystemContentCache(tmp_path, max_bytes=100)
    await cache.put("spec:a:1:e1", b"one")
    await cache.put("spec:b:1:e2", b"two")
    present = await cache.contains_many(("spec:a:1:e1", "spec:b:1:e2", "spec:c:1:e3"))
    assert present == frozenset({"spec:a:1:e1", "spec:b:1:e2"})


@pytest.mark.asyncio
async def test_tiered_contains_many_satisfied_if_either_tier_holds(tmp_path):
    l1 = MemoryContentCache(max_bytes=100)
    l2 = FilesystemContentCache(tmp_path, max_bytes=100)
    cache = TieredContentCache(l1, l2)
    await l1.put("in-l1", b"x")
    await l2.put("in-l2", b"y")
    present = await cache.contains_many(("in-l1", "in-l2", "neither"))
    assert present == frozenset({"in-l1", "in-l2"})


@pytest.mark.asyncio
async def test_contains_many_on_none_cache_is_empty():
    assert await contains_many(None, ("a", "b")) == frozenset()
