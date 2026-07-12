#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/resource/test_memory_backend.py"""

import pytest

from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.models import Found, Missing, Masked, Depth
from linktools.ai.storage.resource.path import ResourcePath


@pytest.mark.asyncio
async def test_put_then_get_roundtrip():
    backend = MemoryResourceBackend()
    info = await backend.raw_put(
        ResourcePath("/a/b.txt"), b"hello", content_type="text/plain", metadata={}
    )
    assert info.size == 5
    assert info.version == 1

    lookup = await backend.raw_get(ResourcePath("/a/b.txt"))
    assert isinstance(lookup, Found)
    assert lookup.resource.content == b"hello"
    assert lookup.resource.info.etag == info.etag


@pytest.mark.asyncio
async def test_get_missing_returns_missing():
    backend = MemoryResourceBackend()
    lookup = await backend.raw_get(ResourcePath("/nope"))
    assert isinstance(lookup, Missing)


@pytest.mark.asyncio
async def test_put_same_content_and_metadata_bumps_version_each_call():
    # NOTE: idempotency (no-version-bump-on-identical-PUT) is a ResourceStore-level
    # concern (scenario), not a backend concern -- the raw backend always applies the write.
    backend = MemoryResourceBackend()
    first = await backend.raw_put(
        ResourcePath("/a/b.txt"), b"hello", content_type=None, metadata={}
    )
    second = await backend.raw_put(
        ResourcePath("/a/b.txt"), b"hello", content_type=None, metadata={}
    )
    assert second.version == first.version + 1


@pytest.mark.asyncio
async def test_delete_existing_returns_info_and_masks():
    backend = MemoryResourceBackend()
    await backend.raw_put(
        ResourcePath("/a/b.txt"), b"hello", content_type=None, metadata={}
    )
    removed = await backend.raw_delete(ResourcePath("/a/b.txt"))
    assert removed is not None
    lookup = await backend.raw_get(ResourcePath("/a/b.txt"))
    assert isinstance(lookup, Masked)


@pytest.mark.asyncio
async def test_delete_never_written_still_masks_for_overlay_shadowing():
    backend = MemoryResourceBackend()
    removed = await backend.raw_delete(ResourcePath("/never/here"))
    assert removed is None
    lookup = await backend.raw_get(ResourcePath("/never/here"))
    assert isinstance(lookup, Masked)


@pytest.mark.asyncio
async def test_revision_increments_on_write():
    backend = MemoryResourceBackend()
    r0 = await backend.revision()
    await backend.raw_put(ResourcePath("/a"), b"x", content_type=None, metadata={})
    r1 = await backend.revision()
    assert r1 > r0


@pytest.mark.asyncio
async def test_propfind_lists_under_prefix_with_depth_one():
    backend = MemoryResourceBackend()
    await backend.raw_put(
        ResourcePath("/agents/a.md"), b"x", content_type=None, metadata={}
    )
    await backend.raw_put(
        ResourcePath("/agents/b.md"), b"y", content_type=None, metadata={}
    )
    await backend.raw_put(
        ResourcePath("/other/c.md"), b"z", content_type=None, metadata={}
    )
    page = await backend.raw_propfind(
        ResourcePath("/agents"), depth=Depth.ONE, limit=100, cursor=None
    )
    paths = {info.path.value for info in page.items}
    assert paths == {"/agents/a.md", "/agents/b.md"}


@pytest.mark.asyncio
async def test_idempotency_record_roundtrip():
    from linktools.ai.storage.resource.models import IdempotencyRecord

    backend = MemoryResourceBackend()
    assert await backend.get_idempotency("k1") is None
    record = IdempotencyRecord(key="k1", request_hash="h1", result=None)
    await backend.put_idempotency(record)
    fetched = await backend.get_idempotency("k1")
    assert fetched == record


@pytest.mark.asyncio
async def test_version_continues_monotonically_across_delete_and_recreate():
    backend = MemoryResourceBackend()
    first = await backend.raw_put(
        ResourcePath("/a.txt"), b"one", content_type=None, metadata={}
    )
    assert first.version == 1
    await backend.raw_delete(ResourcePath("/a.txt"))
    recreated = await backend.raw_put(
        ResourcePath("/a.txt"), b"two", content_type=None, metadata={}
    )
    assert recreated.version == 3
