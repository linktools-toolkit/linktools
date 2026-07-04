#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/resource/test_store.py"""
import pytest

from linktools.ai.errors import IdempotencyConflictError, ResourcePreconditionFailedError, ResourceReadOnlyError
from linktools.ai.storage.resource.file import FileResourceBackend
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.models import Depth, WriteOptions
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.resource.store import ResourceStore


def _memory_backend(**kwargs):
    return MemoryResourceBackend(**kwargs)


def _file_backend(tmp_path, **kwargs):
    return FileResourceBackend(root=tmp_path, **kwargs)


@pytest.fixture(params=["memory", "file"])
def backend_factory(request, tmp_path):
    if request.param == "memory":
        return lambda **kw: _memory_backend(**kw)
    counter = {"n": 0}

    def factory(**kw):
        counter["n"] += 1
        return _file_backend(tmp_path / f"backend-{counter['n']}", **kw)

    return factory


@pytest.mark.asyncio
async def test_primary_only_put_get_roundtrip(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.put(ResourcePath("/a.txt"), b"hello")
    resource = await store.get(ResourcePath("/a.txt"))
    assert resource.content == b"hello"


@pytest.mark.asyncio
async def test_get_missing_returns_none(backend_factory):
    store = ResourceStore(primary=backend_factory())
    assert await store.get(ResourcePath("/nope")) is None


@pytest.mark.asyncio
async def test_overlay_fallback_when_primary_missing(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/builtin.md"), b"builtin content", content_type=None, metadata={})
    store = ResourceStore(primary=backend_factory(), overlays=(overlay,))
    resource = await store.get(ResourcePath("/builtin.md"))
    assert resource.content == b"builtin content"


@pytest.mark.asyncio
async def test_primary_shadows_overlay(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/shared.md"), b"overlay version", content_type=None, metadata={})
    primary = backend_factory()
    store = ResourceStore(primary=primary, overlays=(overlay,))
    await store.put(ResourcePath("/shared.md"), b"primary version")
    resource = await store.get(ResourcePath("/shared.md"))
    assert resource.content == b"primary version"


@pytest.mark.asyncio
async def test_whiteout_prevents_overlay_resurrection(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/builtin.md"), b"builtin content", content_type=None, metadata={})
    store = ResourceStore(primary=backend_factory(), overlays=(overlay,))
    assert (await store.get(ResourcePath("/builtin.md"))) is not None
    await store.delete(ResourcePath("/builtin.md"))
    assert (await store.get(ResourcePath("/builtin.md"))) is None


@pytest.mark.asyncio
async def test_write_to_readonly_primary_raises(backend_factory):
    store = ResourceStore(primary=backend_factory(readonly=True))
    with pytest.raises(ResourceReadOnlyError):
        await store.put(ResourcePath("/a.txt"), b"x")


@pytest.mark.asyncio
async def test_put_same_content_and_metadata_does_not_bump_version(backend_factory):
    store = ResourceStore(primary=backend_factory())
    first = await store.put(ResourcePath("/a.txt"), b"same", options=WriteOptions(metadata={"k": "v"}))
    second = await store.put(ResourcePath("/a.txt"), b"same", options=WriteOptions(metadata={"k": "v"}))
    assert first.info.version == second.info.version


@pytest.mark.asyncio
async def test_put_different_content_bumps_version(backend_factory):
    store = ResourceStore(primary=backend_factory())
    first = await store.put(ResourcePath("/a.txt"), b"one")
    second = await store.put(ResourcePath("/a.txt"), b"two")
    assert second.info.version == first.info.version + 1


@pytest.mark.asyncio
async def test_delete_missing_path_is_a_no_op_success(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.delete(ResourcePath("/never/existed"))  # must not raise


@pytest.mark.asyncio
async def test_conditional_put_if_none_match_rejects_existing(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.put(ResourcePath("/a.txt"), b"x")
    with pytest.raises(ResourcePreconditionFailedError):
        await store.put(ResourcePath("/a.txt"), b"y", options=WriteOptions(if_none_match=True))


@pytest.mark.asyncio
async def test_conditional_put_if_match_wrong_etag_rejects(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.put(ResourcePath("/a.txt"), b"x")
    with pytest.raises(ResourcePreconditionFailedError):
        await store.put(ResourcePath("/a.txt"), b"y", options=WriteOptions(if_match="wrong-etag"))


@pytest.mark.asyncio
async def test_conditional_put_if_match_correct_etag_succeeds(backend_factory):
    store = ResourceStore(primary=backend_factory())
    first = await store.put(ResourcePath("/a.txt"), b"x")
    updated = await store.put(ResourcePath("/a.txt"), b"y", options=WriteOptions(if_match=first.info.etag))
    assert updated.content == b"y"


@pytest.mark.asyncio
async def test_idempotent_put_same_key_and_hash_replays_first_result(backend_factory):
    store = ResourceStore(primary=backend_factory())
    first = await store.put(ResourcePath("/a.txt"), b"x", options=WriteOptions(idempotency_key="k1"))
    second = await store.put(ResourcePath("/a.txt"), b"x", options=WriteOptions(idempotency_key="k1"))
    assert first.info.version == second.info.version


@pytest.mark.asyncio
async def test_idempotent_put_same_key_different_hash_conflicts(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.put(ResourcePath("/a.txt"), b"x", options=WriteOptions(idempotency_key="k1"))
    with pytest.raises(IdempotencyConflictError):
        await store.put(ResourcePath("/a.txt"), b"different", options=WriteOptions(idempotency_key="k1"))


@pytest.mark.asyncio
async def test_idempotent_delete_same_key_replays(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.put(ResourcePath("/a.txt"), b"x")
    await store.delete(ResourcePath("/a.txt"), options=WriteOptions(idempotency_key="d1"))
    await store.delete(ResourcePath("/a.txt"), options=WriteOptions(idempotency_key="d1"))  # must not raise


@pytest.mark.asyncio
async def test_move_shadows_overlay_source_after_move(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/src.md"), b"overlay content", content_type=None, metadata={})
    store = ResourceStore(primary=backend_factory(), overlays=(overlay,))
    moved = await store.move(ResourcePath("/src.md"), ResourcePath("/dst.md"))
    assert moved.content == b"overlay content"
    assert (await store.get(ResourcePath("/src.md"))) is None
    assert (await store.get(ResourcePath("/dst.md"))).content == b"overlay content"


@pytest.mark.asyncio
async def test_propfind_merges_primary_and_overlay_primary_wins(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/agents/shared.md"), b"overlay", content_type=None, metadata={})
    await overlay.raw_put(ResourcePath("/agents/only-overlay.md"), b"overlay-only", content_type=None, metadata={})
    primary = backend_factory()
    store = ResourceStore(primary=primary, overlays=(overlay,))
    await store.put(ResourcePath("/agents/shared.md"), b"primary")
    page = await store.propfind(ResourcePath("/agents"), depth=Depth.ONE, limit=100, cursor=None)
    by_path = {i.path.value: i for i in page.items}
    assert set(by_path) == {"/agents/shared.md", "/agents/only-overlay.md"}
    shared = await store.get(ResourcePath("/agents/shared.md"))
    assert shared.content == b"primary"


@pytest.mark.asyncio
async def test_put_identical_to_overlay_content_still_writes_primary(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/x.txt"), b"same", content_type=None, metadata={})
    primary = backend_factory()
    store = ResourceStore(primary=primary, overlays=(overlay,))
    await store.put(ResourcePath("/x.txt"), b"same", options=WriteOptions(metadata={}))
    primary_lookup = await primary.raw_get(ResourcePath("/x.txt"))
    from linktools.ai.storage.resource.models import Found
    assert isinstance(primary_lookup, Found)
    assert primary_lookup.resource.content == b"same"


@pytest.mark.asyncio
async def test_propfind_hides_deleted_overlay_only_path(backend_factory):
    overlay = backend_factory(readonly=True)
    await overlay.raw_put(ResourcePath("/agents/only-overlay.md"), b"overlay-only", content_type=None, metadata={})
    store = ResourceStore(primary=backend_factory(), overlays=(overlay,))
    await store.delete(ResourcePath("/agents/only-overlay.md"))
    page = await store.propfind(ResourcePath("/agents"), depth=Depth.ONE, limit=100, cursor=None)
    assert "/agents/only-overlay.md" not in {i.path.value for i in page.items}


@pytest.mark.asyncio
async def test_move_forwards_if_none_match_to_destination_write(backend_factory):
    store = ResourceStore(primary=backend_factory())
    await store.put(ResourcePath("/src.txt"), b"data")
    await store.put(ResourcePath("/dst.txt"), b"already here")
    with pytest.raises(ResourcePreconditionFailedError):
        await store.move(ResourcePath("/src.txt"), ResourcePath("/dst.txt"), options=WriteOptions(if_none_match=True))
