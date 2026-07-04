#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/resource/test_models.py"""
import json
from datetime import datetime, timezone

from linktools.ai.storage.resource.models import (
    ResourceKind,
    ResourceInfo,
    Resource,
    Found,
    Missing,
    Masked,
    Depth,
    ResourcePage,
    WriteOptions,
    IdempotencyRecord,
)
from linktools.ai.storage.resource.path import ResourcePath


def _info(path="/a/b.txt", size=3) -> ResourceInfo:
    return ResourceInfo(
        path=ResourcePath(path),
        kind=ResourceKind.FILE,
        etag="etag-1",
        version=1,
        content_type="text/plain",
        size=size,
        modified_at=datetime.now(timezone.utc),
        metadata={"k": "v"},
    )


def test_resource_kind_values():
    assert ResourceKind.FILE == "file"
    assert ResourceKind.COLLECTION == "collection"


def test_resource_text_and_from_text():
    info = _info()
    r = Resource.from_text(info, "abc")
    assert r.content == b"abc"
    assert r.text() == "abc"


def test_resource_json_and_from_json():
    info = _info()
    r = Resource.from_json(info, {"x": 1})
    assert json.loads(r.content) == {"x": 1}
    assert r.json() == {"x": 1}


def test_lookup_variants_are_distinct_types():
    found = Found(resource=Resource.from_text(_info(), "abc"))
    missing = Missing()
    masked = Masked(path=ResourcePath("/a/b.txt"), version=2)
    assert isinstance(found, Found)
    assert isinstance(missing, Missing)
    assert isinstance(masked, Masked)
    assert masked.version == 2


def test_depth_values():
    assert Depth.ZERO == "0"
    assert Depth.ONE == "1"


def test_resource_page():
    page = ResourcePage(items=(_info(),), cursor="next-token")
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
    assert rec.result.path == ResourcePath("/a/b.txt")
