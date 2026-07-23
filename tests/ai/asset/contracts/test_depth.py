#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Depth-contract: the same ZERO/ONE/INFINITY cases run against the Memory,
Filesystem, and SqlAlchemy asset backends so the three cannot drift. Encodes
the fixed semantics:

    ZERO       -> the target asset itself (empty when it is not stored)
    ONE        -> target + direct children
    INFINITY   -> target + every descendant

plus stable keyset pagination (no duplicates, no omissions at limit=1), whiteout
exclusion, and overlay contribution under the AssetStore merge."""

import pytest

from linktools.ai.asset.models import Depth
from linktools.ai.asset.path import (
    AssetPath,
    _relative_depth,
    matches_asset_depth,
    relative_asset_depth,
)
from linktools.ai.asset.store import AssetStore
from linktools.ai.errors import InvalidAssetPathError


def _paths(page) -> "list[str]":
    return [info.path.value for info in page.items]


# --------------------------------------------------------------------------
# helper unit tests (backend-independent)
# --------------------------------------------------------------------------


def test_relative_asset_depth_branches():
    base = AssetPath("/a")
    assert relative_asset_depth(base, AssetPath("/a")) == 0
    assert relative_asset_depth(base, AssetPath("/a/file.txt")) == 1
    assert relative_asset_depth(base, AssetPath("/a/dir")) == 1
    assert relative_asset_depth(base, AssetPath("/a/dir/deep.txt")) == 2
    assert relative_asset_depth(base, AssetPath("/a/x/y/z")) == 3
    assert relative_asset_depth(base, AssetPath("/b.txt")) is None
    # sibling-ish prefix that is NOT a path descendant
    assert relative_asset_depth(base, AssetPath("/ab")) is None
    assert relative_asset_depth(base, AssetPath("/agents")) is None


def test_relative_depth_root_namespace():
    # The depth logic must be correct for the root namespace (base == "/"):
    # every top-level path is a direct child.
    assert _relative_depth("/", "/") == 0
    assert _relative_depth("/", "/a") == 1
    assert _relative_depth("/", "/a/b") == 2
    assert _relative_depth("/", "/agents") == 1
    assert _relative_depth("/", "/a/b/c") == 3


def test_root_asset_path_constructs():
    # "/" is a valid path: the root namespace, against which every path matches
    # and whose direct children a ONE-depth list enumerates.
    assert AssetPath("/").value == "/"


@pytest.mark.asyncio
async def test_one_at_root_lists_only_top_level_paths(make_store):
    store = make_store()
    page = await store.list(AssetPath("/"), depth=Depth.ONE, limit=100, cursor=None)
    # Root's direct children are the top-level paths only (/a, /b.txt); the
    # deeper descendants (/a/dir, /a/file.txt, /a/dir/deep.txt) are NOT direct
    # children of root and must not appear.
    assert _paths(page) == ["/a", "/b.txt"]


def test_matches_asset_depth_branches():
    base = AssetPath("/a")
    target = AssetPath("/a")
    child = AssetPath("/a/file.txt")
    grandchild = AssetPath("/a/dir/deep.txt")
    outsider = AssetPath("/b.txt")
    for depth in (Depth.ZERO, Depth.ONE, Depth.INFINITY):
        assert matches_asset_depth(base, outsider, depth) is False
    assert matches_asset_depth(base, target, Depth.ZERO) is True
    assert matches_asset_depth(base, child, Depth.ZERO) is False
    assert matches_asset_depth(base, grandchild, Depth.ZERO) is False
    assert matches_asset_depth(base, target, Depth.ONE) is True
    assert matches_asset_depth(base, child, Depth.ONE) is True
    assert matches_asset_depth(base, grandchild, Depth.ONE) is False
    assert matches_asset_depth(base, target, Depth.INFINITY) is True
    assert matches_asset_depth(base, child, Depth.INFINITY) is True
    assert matches_asset_depth(base, grandchild, Depth.INFINITY) is True


# --------------------------------------------------------------------------
# per-backend contract (parametrized over memory / filesystem / sqlalchemy)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_returns_only_the_target(make_store):
    store = make_store()
    page = await store.list(AssetPath("/a"), depth=Depth.ZERO, limit=100, cursor=None)
    assert _paths(page) == ["/a"]


@pytest.mark.asyncio
async def test_one_returns_target_and_direct_children(make_store):
    store = make_store()
    page = await store.list(AssetPath("/a"), depth=Depth.ONE, limit=100, cursor=None)
    assert _paths(page) == ["/a", "/a/dir", "/a/file.txt"]


@pytest.mark.asyncio
async def test_infinity_returns_target_and_all_descendants(make_store):
    store = make_store()
    page = await store.list(
        AssetPath("/a"), depth=Depth.INFINITY, limit=100, cursor=None
    )
    assert _paths(page) == ["/a", "/a/dir", "/a/dir/deep.txt", "/a/file.txt"]


@pytest.mark.asyncio
async def test_zero_on_missing_target_is_empty(make_store):
    store = make_store()
    page = await store.list(
        AssetPath("/missing"), depth=Depth.ZERO, limit=100, cursor=None
    )
    assert _paths(page) == []


@pytest.mark.asyncio
async def test_zero_does_not_synthesize_an_unstored_target(make_backend):
    # Only a child exists; the target itself was never stored, so ZERO must not
    # invent it.
    primary = make_backend()
    store = AssetStore(primary=primary)
    await store.put(AssetPath("/a/child.txt"), b"x")
    page = await store.list(AssetPath("/a"), depth=Depth.ZERO, limit=100, cursor=None)
    assert _paths(page) == []
    # ONE still surfaces the stored child (target is simply absent, not faked).
    page = await store.list(AssetPath("/a"), depth=Depth.ONE, limit=100, cursor=None)
    assert _paths(page) == ["/a/child.txt"]


@pytest.mark.asyncio
async def test_one_on_a_subcollection(make_store):
    store = make_store()
    page = await store.list(
        AssetPath("/a/dir"), depth=Depth.ONE, limit=100, cursor=None
    )
    assert _paths(page) == ["/a/dir", "/a/dir/deep.txt"]


@pytest.mark.asyncio
async def test_infinity_on_a_leaf_target_returns_only_it(make_store):
    store = make_store()
    page = await store.list(
        AssetPath("/b.txt"), depth=Depth.INFINITY, limit=100, cursor=None
    )
    assert _paths(page) == ["/b.txt"]


@pytest.mark.asyncio
async def test_whiteout_excludes_the_deleted_path(make_store):
    store = make_store()
    await store.delete(AssetPath("/a/file.txt"))
    page = await store.list(AssetPath("/a"), depth=Depth.ONE, limit=100, cursor=None)
    assert _paths(page) == ["/a", "/a/dir"]


@pytest.mark.asyncio
async def test_overlay_contributes_paths_and_primary_shadows(make_backend):
    overlay = make_backend()
    overlay_store = AssetStore(primary=overlay)
    await overlay_store.put(AssetPath("/a/overlay.md"), b"overlay")
    await overlay_store.put(AssetPath("/a/file.txt"), b"overlay-version")

    primary = make_backend()
    primary_store = AssetStore(primary=primary)
    await primary_store.put(AssetPath("/a"), b"primary")
    await primary_store.put(AssetPath("/a/file.txt"), b"primary-version")

    store = AssetStore(primary=primary, overlays=(overlay,))
    page = await store.list(AssetPath("/a"), depth=Depth.ONE, limit=100, cursor=None)
    assert set(_paths(page)) == {"/a", "/a/file.txt", "/a/overlay.md"}
    # primary wins on the conflicting path
    assert (await store.get(AssetPath("/a/file.txt"))).content == b"primary-version"


@pytest.mark.asyncio
async def test_store_level_limit_one_pagination_has_no_gaps_or_duplicates(make_store):
    store = make_store()
    seen: "list[str]" = []
    cursor = None
    while True:
        page = await store.list(
            AssetPath("/a"), depth=Depth.INFINITY, limit=1, cursor=cursor
        )
        seen.extend(_paths(page))
        if page.cursor is None:
            break
        cursor = page.cursor
    assert seen == ["/a", "/a/dir", "/a/dir/deep.txt", "/a/file.txt"]
    assert len(seen) == len(set(seen)), "pagination duplicated a path"


@pytest.mark.asyncio
async def test_backend_direct_limit_one_pagination_has_no_gaps_or_duplicates(
    make_backend,
):
    # Drives raw_list directly: this is the case that proves the keyset cursor is
    # the last *returned* path (not the first unreturned one), so paging at
    # limit=1 neither skips the boundary item nor re-emits it.
    backend = make_backend()
    store = AssetStore(primary=backend)
    for path in ("/a", "/a/file.txt", "/a/dir", "/a/dir/deep.txt", "/b.txt"):
        await store.put(AssetPath(path), b"x")

    seen: "list[str]" = []
    cursor = None
    while True:
        page = await backend.raw_list(
            AssetPath("/a"), depth=Depth.INFINITY, limit=1, cursor=cursor
        )
        seen.extend(_paths(page))
        if page.cursor is None:
            break
        cursor = page.cursor
    assert seen == ["/a", "/a/dir", "/a/dir/deep.txt", "/a/file.txt"]
    assert len(seen) == len(set(seen)), "pagination duplicated a path"
