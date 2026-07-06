#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/resource/test_file_backend.py"""
import pytest

from linktools.ai.errors import InvalidResourcePathError
from linktools.ai.storage.resource.file import FileResourceBackend, _filename
from linktools.ai.storage.resource.models import Found, Missing, Masked, Depth, IdempotencyRecord
from linktools.ai.storage.resource.path import ResourcePath


@pytest.mark.asyncio
async def test_put_then_get_roundtrip_persists_to_disk(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    info = await backend.raw_put(ResourcePath("/a/b.txt"), b"hello", content_type="text/plain", metadata={"k": "v"})
    assert info.version == 1

    reopened = FileResourceBackend(root=tmp_path)
    lookup = await reopened.raw_get(ResourcePath("/a/b.txt"))
    assert isinstance(lookup, Found)
    assert lookup.resource.content == b"hello"
    assert lookup.resource.info.metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_get_missing_returns_missing(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    assert isinstance(await backend.raw_get(ResourcePath("/nope")), Missing)


@pytest.mark.asyncio
async def test_delete_masks_and_survives_reopen(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.raw_put(ResourcePath("/a/b.txt"), b"hello", content_type=None, metadata={})
    await backend.raw_delete(ResourcePath("/a/b.txt"))

    reopened = FileResourceBackend(root=tmp_path)
    assert isinstance(await reopened.raw_get(ResourcePath("/a/b.txt")), Masked)


@pytest.mark.asyncio
async def test_readonly_backend_still_supports_reads(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.raw_put(ResourcePath("/a.txt"), b"x", content_type=None, metadata={})

    ro = FileResourceBackend(root=tmp_path, readonly=True)
    assert ro.readonly is True
    lookup = await ro.raw_get(ResourcePath("/a.txt"))
    assert isinstance(lookup, Found)


@pytest.mark.asyncio
async def test_atomic_replace_leaves_no_temp_file_on_disk(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.raw_put(ResourcePath("/a.txt"), b"x", content_type=None, metadata={})
    leftovers = list((tmp_path / "data").glob("*.tmp*"))
    assert leftovers == []


@pytest.mark.asyncio
async def test_revision_persists_across_reopen(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.raw_put(ResourcePath("/a.txt"), b"x", content_type=None, metadata={})
    reopened = FileResourceBackend(root=tmp_path)
    assert await reopened.revision() == 1


@pytest.mark.asyncio
async def test_propfind_depth_one(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.raw_put(ResourcePath("/agents/a.md"), b"1", content_type=None, metadata={})
    await backend.raw_put(ResourcePath("/agents/b.md"), b"2", content_type=None, metadata={})
    await backend.raw_put(ResourcePath("/other/c.md"), b"3", content_type=None, metadata={})
    page = await backend.raw_propfind(ResourcePath("/agents"), depth=Depth.ONE, limit=100, cursor=None)
    assert {i.path.value for i in page.items} == {"/agents/a.md", "/agents/b.md"}


@pytest.mark.asyncio
async def test_idempotency_record_persists_across_reopen(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.put_idempotency(IdempotencyRecord(key="k1", request_hash="h1", result=None))
    reopened = FileResourceBackend(root=tmp_path)
    fetched = await reopened.get_idempotency("k1")
    assert fetched.key == "k1" and fetched.request_hash == "h1"


@pytest.mark.asyncio
async def test_paths_with_double_underscore_do_not_collide_with_nested_paths(tmp_path):
    backend = FileResourceBackend(root=tmp_path)
    await backend.raw_put(ResourcePath("/a/b"), b"nested", content_type=None, metadata={})
    await backend.raw_put(ResourcePath("/a__b"), b"flat-with-underscores", content_type=None, metadata={})
    nested = await backend.raw_get(ResourcePath("/a/b"))
    flat = await backend.raw_get(ResourcePath("/a__b"))
    assert isinstance(nested, Found) and isinstance(flat, Found)
    assert nested.resource.content == b"nested"
    assert flat.resource.content == b"flat-with-underscores"


@pytest.mark.asyncio
async def test_symlink_escape_is_denied(tmp_path):
    root = tmp_path / "backend-root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    backend = FileResourceBackend(root=root)
    # Put something so the metadata dir exists, then plant a symlink inside it
    # pointing outside the backend root, using the exact filename that
    # ResourcePath("/evil__link") would resolve to.
    await backend.raw_put(ResourcePath("/a.txt"), b"x", content_type=None, metadata={})
    evil_name = _filename(ResourcePath("/evil__link")) + ".json"
    evil_link = root / ".resource" / "metadata" / evil_name
    evil_link.symlink_to(outside / "does-not-exist.json")
    with pytest.raises(InvalidResourcePathError):
        await backend.raw_get(ResourcePath("/evil__link"))
