#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReadOnlyAssetBackend: read-only-ness is structural (no write methods), not a
runtime flag. The wrapper delegates reads, exposes no write surface, and is
rejected as an AssetStore primary because it does not satisfy
AssetWriterBackend."""

import pytest

from linktools.ai.asset import ReadOnlyAssetBackend
from linktools.ai.asset.backend import AssetReaderBackend, AssetWriterBackend
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.models import Depth
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.errors import AssetReadOnlyError


@pytest.mark.asyncio
async def test_readonly_backend_delegates_reads_and_has_no_write_methods():
    writable = MemoryAssetBackend()
    await writable.raw_put(
        AssetPath("/a/b.txt"), b"data", content_type=None, metadata={}
    )
    ro = ReadOnlyAssetBackend(writable)

    # It is a Reader...
    assert isinstance(ro, AssetReaderBackend)
    # ...and NOT a Writer (no write methods -> fails the structural check).
    assert not isinstance(ro, AssetWriterBackend)
    for write_method in (
        "raw_put_checked",
        "raw_delete_checked",
        "raw_move_checked",
    ):
        assert not hasattr(ro, write_method), (
            f"ReadOnlyAssetBackend must not expose {write_method}"
        )
    # Idempotency is not part of the read surface -- the wrapper exposes no
    # idempotency method either.
    assert not hasattr(ro, "get_idempotency")
    assert not hasattr(ro, "put_idempotency")
    # The Reader backend_id is forwarded from the inner backend.
    assert ro.backend_id == writable.backend_id

    # Reads delegate to the inner writable backend.
    lookup = await ro.raw_get(AssetPath("/a/b.txt"))
    assert lookup.asset.content == b"data"
    page = await ro.raw_list(AssetPath("/a"), depth=Depth.ONE, limit=10, cursor=None)
    assert [i.path.value for i in page.items] == ["/a/b.txt"]


@pytest.mark.asyncio
async def test_readonly_backend_as_primary_is_rejected_at_construction():
    # The structural check at AssetStore construction rejects a read-only primary
    # with a clear error (no runtime bool involved).
    ro = ReadOnlyAssetBackend(MemoryAssetBackend())
    with pytest.raises(AssetReadOnlyError):
        AssetStore(primary=ro)


@pytest.mark.asyncio
async def test_readonly_backend_serves_as_overlay():
    writable = MemoryAssetBackend()
    await writable.raw_put(
        AssetPath("/builtin.md"), b"overlay", content_type=None, metadata={}
    )
    overlay = ReadOnlyAssetBackend(writable)
    # A writable primary + a read-only overlay is the intended composition.
    store = AssetStore(primary=MemoryAssetBackend(), overlays=(overlay,))
    assert (await store.get(AssetPath("/builtin.md"))).content == b"overlay"
