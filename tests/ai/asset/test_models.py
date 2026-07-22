#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/asset/test_models.py"""

import json
from datetime import datetime, timezone

from linktools.ai.asset.models import (
    AssetKind,
    AssetInfo,
    Asset,
    Found,
    Missing,
    Masked,
    Depth,
    AssetPage,
    WriteOptions,
    IdempotencyRecord,
)
from linktools.ai.asset.path import AssetPath


def _info(path="/a/b.txt", size=3) -> AssetInfo:
    return AssetInfo(
        path=AssetPath(path),
        kind=AssetKind.FILE,
        etag="etag-1",
        version=1,
        content_type="text/plain",
        size=size,
        modified_at=datetime.now(timezone.utc),
        metadata={"k": "v"},
    )


def test_resource_kind_values():
    assert AssetKind.FILE == "file"
    assert AssetKind.COLLECTION == "collection"


def test_resource_text_and_from_text():
    info = _info()
    r = Asset.from_text(info, "abc")
    assert r.content == b"abc"
    assert r.text() == "abc"


def test_resource_json_and_from_json():
    info = _info()
    r = Asset.from_json(info, {"x": 1})
    assert json.loads(r.content) == {"x": 1}
    assert r.json() == {"x": 1}


def test_lookup_variants_are_distinct_types():
    found = Found(asset=Asset.from_text(_info(), "abc"))
    missing = Missing()
    masked = Masked(path=AssetPath("/a/b.txt"), version=2)
    assert isinstance(found, Found)
    assert isinstance(missing, Missing)
    assert isinstance(masked, Masked)
    assert masked.version == 2


def test_depth_values():
    assert Depth.ZERO == "0"
    assert Depth.ONE == "1"
    assert Depth.INFINITY == "infinity"


def test_depth_infinity_list_returns_all_descendants(tmp_path):
    # INFINITY: target + every descendant (the full subtree).
    import asyncio

    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.path import AssetPath
    from linktools.ai.asset.store import AssetStore

    store = AssetStore(primary=MemoryAssetBackend())

    async def _run():
        for p in ("/r/a", "/r/a/b", "/r/a/b/c", "/r/c"):
            await store.put(AssetPath(p), b"x")
        return await store.list(
            AssetPath("/r"), depth=Depth.INFINITY, limit=100, cursor=None
        )

    page = asyncio.run(_run())
    paths = {info.path.value for info in page.items}
    # INFINITY returns the full subtree (direct children AND deeper);
    # Depth.ONE would stop at /r/a and /r/c.
    assert {"/r/a", "/r/a/b", "/r/a/b/c", "/r/c"} <= paths, paths


def test_resource_page():
    page = AssetPage(items=(_info(),), cursor="next-token")
    assert len(page.items) == 1
    assert page.cursor == "next-token"


def test_write_options_defaults():
    opts = WriteOptions()
    assert opts.idempotency_key is None
    assert opts.if_match is None
    assert opts.if_none_match is False
    assert opts.content_type is None
    assert dict(opts.metadata) == {}
    assert opts.actor is None


def test_idempotency_record():
    rec = IdempotencyRecord(key="k1", request_hash="h1", result=_info())
    assert rec.key == "k1"
    assert rec.result.path == AssetPath("/a/b.txt")
