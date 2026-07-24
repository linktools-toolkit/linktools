#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root-path contract: AssetPath("/") is the namespace root -- a synthetic
directory, not a persistable Asset. list/stat/revision accept it; get/put/
delete/move reject it with AssetRootMutationError. The same cases run against
Memory, Filesystem, and SqlAlchemy so the three backends cannot drift."""

import pytest

from linktools.ai.asset.models import AssetKind, Depth, WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.errors import AssetRootMutationError


def test_root_path_parts_name_parent_namespace():
    root = AssetPath("/")
    assert root.parts == ()
    assert root.name == ""
    assert root.parent is None
    assert root.namespace is None


def test_non_root_path_parent_and_name():
    assert AssetPath("/a/b").name == "b"
    assert AssetPath("/a/b").parent == AssetPath("/a")
    assert AssetPath("/a").parent == AssetPath("/")
    assert AssetPath("/a").namespace == "a"


@pytest.mark.asyncio
async def test_stat_root_returns_synthetic_directory(make_store):
    store = make_store()
    info = await store.stat(AssetPath("/"))
    assert info is not None
    assert info.path == AssetPath("/")
    assert info.kind is AssetKind.DIRECTORY
    assert info.synthetic is True


@pytest.mark.asyncio
async def test_list_root_includes_synthetic_root_entry(make_store):
    store = make_store()
    page = await store.list(AssetPath("/"), depth=Depth.ZERO, limit=100, cursor=None)
    assert [info.path.value for info in page.items] == ["/"]
    assert page.items[0].synthetic is True


@pytest.mark.asyncio
async def test_get_root_is_rejected(make_store):
    store = make_store()
    with pytest.raises(AssetRootMutationError):
        await store.get(AssetPath("/"))


@pytest.mark.asyncio
async def test_put_root_is_rejected(make_store):
    store = make_store()
    with pytest.raises(AssetRootMutationError):
        await store.put(AssetPath("/"), b"x")


@pytest.mark.asyncio
async def test_delete_root_is_rejected(make_store):
    store = make_store()
    with pytest.raises(AssetRootMutationError):
        await store.delete(AssetPath("/"))


@pytest.mark.asyncio
async def test_move_root_source_is_rejected(make_store):
    store = make_store()
    with pytest.raises(AssetRootMutationError):
        await store.move(AssetPath("/"), AssetPath("/a"))


@pytest.mark.asyncio
async def test_move_root_target_is_rejected(make_store):
    store = make_store()
    with pytest.raises(AssetRootMutationError):
        await store.move(AssetPath("/a"), AssetPath("/"))


@pytest.mark.asyncio
async def test_backend_raw_get_root_is_rejected_directly(make_backend):
    backend = make_backend()
    with pytest.raises(AssetRootMutationError):
        await backend.raw_get(AssetPath("/"))


@pytest.mark.asyncio
async def test_backend_raw_put_checked_root_is_rejected_directly(make_backend):
    backend = make_backend()
    with pytest.raises(AssetRootMutationError):
        await backend.raw_put_checked(
            AssetPath("/"), b"x", options=WriteOptions(), request_hash="h"
        )
