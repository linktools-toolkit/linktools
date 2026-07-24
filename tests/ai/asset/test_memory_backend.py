#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/asset/test_memory_backend.py"""

import pytest

from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.models import Found, Missing, Masked, Depth
from linktools.ai.asset.path import AssetPath


@pytest.mark.asyncio
async def test_put_then_get_roundtrip():
    backend = MemoryAssetBackend()
    info = await backend.raw_put(
        AssetPath("/a/b.txt"), b"hello", content_type="text/plain", metadata={}
    )
    assert info.size == 5
    assert info.version == 1

    lookup = await backend.raw_get(AssetPath("/a/b.txt"))
    assert isinstance(lookup, Found)
    assert lookup.asset.content == b"hello"
    assert lookup.asset.info.etag == info.etag


@pytest.mark.asyncio
async def test_get_missing_returns_missing():
    backend = MemoryAssetBackend()
    lookup = await backend.raw_get(AssetPath("/nope"))
    assert isinstance(lookup, Missing)


@pytest.mark.asyncio
async def test_put_same_content_and_metadata_bumps_version_each_call():
    # NOTE: idempotency (no-version-bump-on-identical-PUT) is a AssetStore-level
    # concern (scenario), not a backend concern -- the raw backend always applies the write.
    backend = MemoryAssetBackend()
    first = await backend.raw_put(
        AssetPath("/a/b.txt"), b"hello", content_type=None, metadata={}
    )
    second = await backend.raw_put(
        AssetPath("/a/b.txt"), b"hello", content_type=None, metadata={}
    )
    assert second.version == first.version + 1


@pytest.mark.asyncio
async def test_delete_existing_returns_info_and_masks():
    backend = MemoryAssetBackend()
    await backend.raw_put(
        AssetPath("/a/b.txt"), b"hello", content_type=None, metadata={}
    )
    removed = await backend.raw_delete(AssetPath("/a/b.txt"))
    assert removed is not None
    lookup = await backend.raw_get(AssetPath("/a/b.txt"))
    assert isinstance(lookup, Masked)


@pytest.mark.asyncio
async def test_delete_never_written_still_masks_for_overlay_shadowing():
    backend = MemoryAssetBackend()
    removed = await backend.raw_delete(AssetPath("/never/here"))
    assert removed is None
    lookup = await backend.raw_get(AssetPath("/never/here"))
    assert isinstance(lookup, Masked)


@pytest.mark.asyncio
async def test_revision_increments_on_write():
    backend = MemoryAssetBackend()
    r0 = int(await backend.revision())
    await backend.raw_put(AssetPath("/a"), b"x", content_type=None, metadata={})
    r1 = int(await backend.revision())
    assert r1 > r0


@pytest.mark.asyncio
async def test_list_lists_under_prefix_with_depth_one():
    backend = MemoryAssetBackend()
    await backend.raw_put(
        AssetPath("/agents/a.md"), b"x", content_type=None, metadata={}
    )
    await backend.raw_put(
        AssetPath("/agents/b.md"), b"y", content_type=None, metadata={}
    )
    await backend.raw_put(
        AssetPath("/other/c.md"), b"z", content_type=None, metadata={}
    )
    page = await backend.raw_list(
        AssetPath("/agents"), depth=Depth.ONE, limit=100, cursor=None
    )
    paths = {info.path.value for info in page.items}
    assert paths == {"/agents/a.md", "/agents/b.md"}


@pytest.mark.asyncio
async def test_checked_put_is_idempotent_within_backend():
    # Idempotency lives INSIDE the checked write, not on a separate reader/
    # writer method: a replay with the same key + request hash returns the
    # cached result without re-mutating, and a different body under the same
    # key conflicts.
    from linktools.ai.asset.models import WriteOptions
    from linktools.ai.errors import IdempotencyConflictError

    backend = MemoryAssetBackend()
    path = AssetPath("/a.txt")
    opts = WriteOptions(idempotency_key="k1")
    first = await backend.raw_put_checked(path, b"one", options=opts, request_hash="h1")
    second = await backend.raw_put_checked(path, b"one", options=opts, request_hash="h1")
    assert second.info.version == first.info.version
    assert backend._revision == 1  # replay did not mutate again
    with pytest.raises(IdempotencyConflictError):
        await backend.raw_put_checked(path, b"two", options=opts, request_hash="h2")


@pytest.mark.asyncio
async def test_version_continues_monotonically_across_delete_and_recreate():
    backend = MemoryAssetBackend()
    first = await backend.raw_put(
        AssetPath("/a.txt"), b"one", content_type=None, metadata={}
    )
    assert first.version == 1
    await backend.raw_delete(AssetPath("/a.txt"))
    recreated = await backend.raw_put(
        AssetPath("/a.txt"), b"two", content_type=None, metadata={}
    )
    assert recreated.version == 3
