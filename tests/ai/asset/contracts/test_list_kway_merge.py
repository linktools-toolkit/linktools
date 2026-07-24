#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/asset/contracts/test_list_kway_merge.py

Regression battery for AssetStore.list()'s per-backend k-way merge cursor
(spec section 5.7): every backend paginates independently, same-path
priority is primary-then-overlay-registration-order, whiteouts shadow
overlay-only candidates, and the returned cursor is an opaque HMAC token
that fails closed on tamper/staleness rather than silently resuming."""

import pytest

from linktools.ai.errors import InvalidAssetCursorError, StaleAssetCursorError
from linktools.ai.asset.models import Depth, WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.readonly import ReadOnlyAssetBackend
from linktools.ai.asset.store import AssetStore


async def _drain(store, path, *, limit, depth=Depth.INFINITY):
    """Collect every item across every page, asserting no page exceeds
    limit and pagination terminates."""
    items = []
    cursor = None
    pages = 0
    while True:
        page = await store.list(path, depth=depth, limit=limit, cursor=cursor)
        assert len(page.items) <= limit
        items.extend(page.items)
        pages += 1
        assert pages < 1000, "list pagination did not terminate"
        if page.cursor is None:
            return items
        cursor = page.cursor


@pytest.mark.asyncio
async def test_worked_example_cross_page_result(make_backend):
    """spec section 5.7 worked example: overlay A has /a../e, primary
    whiteouts /a /b /c, overlay B has /z -- the final cross-page visible set
    must be exactly {/d, /e, /z}, at limit=2 (forcing multiple pages)."""
    primary = make_backend()
    overlay_a = make_backend()
    overlay_b = make_backend()
    for name in "abcde":
        await overlay_a.raw_put(
            AssetPath(f"/{name}"), name.encode(), content_type=None, metadata={}
        )
    await overlay_b.raw_put(AssetPath("/z"), b"z", content_type=None, metadata={})
    store = AssetStore(
        primary=primary,
        overlays=(ReadOnlyAssetBackend(overlay_a), ReadOnlyAssetBackend(overlay_b)),
    )
    for name in "abc":
        await store.put(AssetPath(f"/{name}"), name.encode())
        await store.delete(AssetPath(f"/{name}"))

    items = await _drain(store, AssetPath("/"), limit=2)
    paths = {item.path.value for item in items if not item.synthetic}
    assert paths == {"/d", "/e", "/z"}


@pytest.mark.asyncio
async def test_three_backends_paginate_independently(make_backend):
    """each backend advances through its OWN raw_list stream -- a fast
    backend's cursor must never cause a slower backend's own unscanned
    items to be skipped."""
    primary = make_backend()
    overlay_0 = make_backend()
    overlay_1 = make_backend()
    for i in range(3):
        await primary.raw_put_checked(
            AssetPath(f"/p/{i:02d}"), f"p{i}".encode(),
            options=WriteOptions(), request_hash=f"p{i}",
        )
    for i in range(7):
        await overlay_0.raw_put(
            AssetPath(f"/o0/{i:02d}"), f"o0-{i}".encode(), content_type=None, metadata={}
        )
    for i in range(1):
        await overlay_1.raw_put(
            AssetPath(f"/o1/{i:02d}"), f"o1-{i}".encode(), content_type=None, metadata={}
        )
    store = AssetStore(
        primary=primary,
        overlays=(ReadOnlyAssetBackend(overlay_0), ReadOnlyAssetBackend(overlay_1)),
    )

    items = await _drain(store, AssetPath("/"), limit=2)
    paths = {item.path.value for item in items if not item.synthetic}
    expected = (
        {f"/p/{i:02d}" for i in range(3)}
        | {f"/o0/{i:02d}" for i in range(7)}
        | {f"/o1/{i:02d}" for i in range(1)}
    )
    assert paths == expected
    assert len(items) == len(set(i.path.value for i in items)), "no duplicates across pages"


@pytest.mark.asyncio
async def test_same_path_shadowing_by_higher_priority_overlay(make_backend):
    """two overlays holding the SAME path: overlay:0 (registered first) must
    win over overlay:1, and primary (if it also held the path) must win over
    both."""
    primary = make_backend()
    overlay_0 = make_backend()
    overlay_1 = make_backend()
    await overlay_0.raw_put(
        AssetPath("/x"), b"from-overlay-0", content_type=None, metadata={}
    )
    await overlay_1.raw_put(
        AssetPath("/x"), b"from-overlay-1", content_type=None, metadata={}
    )
    store = AssetStore(
        primary=primary,
        overlays=(ReadOnlyAssetBackend(overlay_0), ReadOnlyAssetBackend(overlay_1)),
    )

    items = await _drain(store, AssetPath("/"), limit=10)
    (winner,) = [i for i in items if i.path.value == "/x"]
    expected = (await overlay_0.raw_stat(AssetPath("/x")))
    assert winner.etag == expected.etag

    # Now primary also holds the path -- primary must win over both overlays.
    await store.put(AssetPath("/x"), b"from-primary")
    items = await _drain(store, AssetPath("/"), limit=10)
    (winner,) = [i for i in items if i.path.value == "/x"]
    primary_info = await primary.raw_stat(AssetPath("/x"))
    assert winner.etag == primary_info.etag


@pytest.mark.asyncio
async def test_consecutive_multi_page_whiteouts(make_backend):
    """a long run of consecutive overlay-only paths, ALL whited-out by
    primary, spanning multiple internal fetch pages -- none may resurrect,
    and the items before/after the whited-out run must still be returned."""
    primary = make_backend()
    overlay = make_backend()
    for i in range(40):
        await overlay.raw_put(
            AssetPath(f"/m/{i:03d}"), f"v{i}".encode(), content_type=None, metadata={}
        )
    store = AssetStore(primary=primary, overlays=(ReadOnlyAssetBackend(overlay),))
    # Whiteout a long contiguous middle run (10..29) that spans multiple
    # fetch_size pages when limit is small.
    for i in range(10, 30):
        await store.put(AssetPath(f"/m/{i:03d}"), b"tombstoned")
        await store.delete(AssetPath(f"/m/{i:03d}"))

    items = await _drain(store, AssetPath("/m"), limit=3)
    paths = {item.path.value for item in items}
    expected = {f"/m/{i:03d}" for i in range(40) if not (10 <= i < 30)}
    assert paths == expected


@pytest.mark.asyncio
async def test_limit_one_pagination(make_backend):
    """limit=1 (the smallest possible page) must still visit every item
    across primary+overlay exactly once."""
    primary = make_backend()
    overlay = make_backend()
    for i in range(5):
        await primary.raw_put_checked(
            AssetPath(f"/p/{i}"), f"p{i}".encode(),
            options=WriteOptions(), request_hash=f"p{i}",
        )
    for i in range(5):
        await overlay.raw_put(
            AssetPath(f"/o/{i}"), f"o{i}".encode(), content_type=None, metadata={}
        )
    store = AssetStore(primary=primary, overlays=(ReadOnlyAssetBackend(overlay),))

    items = await _drain(store, AssetPath("/"), limit=1)
    paths = [item.path.value for item in items if not item.synthetic]
    expected = sorted({f"/p/{i}" for i in range(5)} | {f"/o/{i}" for i in range(5)})
    assert sorted(paths) == expected
    assert len(paths) == len(set(paths))


@pytest.mark.asyncio
async def test_cursor_tamper_rejected(make_backend):
    """a modified page token must be rejected as InvalidAssetCursorError,
    never silently decoded or resumed against a wrong/partial state."""
    primary = make_backend()
    for i in range(5):
        await primary.raw_put_checked(
            AssetPath(f"/a/{i}"), f"v{i}".encode(),
            options=WriteOptions(), request_hash=f"h{i}",
        )
    store = AssetStore(primary=primary)
    page = await store.list(AssetPath("/a"), limit=2)
    assert page.cursor is not None

    body, _, tag = page.cursor.rpartition(".")
    tampered = f"{body}x.{tag}"
    with pytest.raises(InvalidAssetCursorError):
        await store.list(AssetPath("/a"), limit=2, cursor=tampered)

    truncated = page.cursor[:-4]
    with pytest.raises(InvalidAssetCursorError):
        await store.list(AssetPath("/a"), limit=2, cursor=truncated)


@pytest.mark.asyncio
async def test_revision_change_invalidates_cursor(make_backend):
    """resuming a cursor after the backend's revision changed underneath it
    must raise StaleAssetCursorError, not silently continue against
    possibly-inconsistent state."""
    primary = make_backend()
    for i in range(5):
        await primary.raw_put_checked(
            AssetPath(f"/a/{i}"), f"v{i}".encode(),
            options=WriteOptions(), request_hash=f"h{i}",
        )
    store = AssetStore(primary=primary)
    page = await store.list(AssetPath("/a"), limit=2)
    assert page.cursor is not None

    await store.put(AssetPath("/a/new"), b"mutation")

    with pytest.raises(StaleAssetCursorError):
        await store.list(AssetPath("/a"), limit=2, cursor=page.cursor)


@pytest.mark.asyncio
async def test_backend_count_change_invalidates_cursor(make_backend):
    """a cursor minted against one backend set must be rejected if resumed
    against a store with a DIFFERENT backend set (here: an overlay added),
    even with the same cursor_secret."""
    secret = b"shared-secret-32-bytes-long!!!!!"
    primary = make_backend()
    for i in range(5):
        await primary.raw_put_checked(
            AssetPath(f"/a/{i}"), f"v{i}".encode(),
            options=WriteOptions(), request_hash=f"h{i}",
        )
    store_one_backend = AssetStore(primary=primary, cursor_secret=secret)
    page = await store_one_backend.list(AssetPath("/a"), limit=2)
    assert page.cursor is not None

    overlay = make_backend()
    store_two_backends = AssetStore(
        primary=primary, overlays=(ReadOnlyAssetBackend(overlay),), cursor_secret=secret
    )
    with pytest.raises(StaleAssetCursorError):
        await store_two_backends.list(AssetPath("/a"), limit=2, cursor=page.cursor)


@pytest.mark.asyncio
async def test_empty_backend_does_not_break_merge(make_backend):
    """an overlay with zero entries must not crash or hang the merge -- it
    is simply always-exhausted-with-an-empty-buffer, contributing nothing."""
    primary = make_backend()
    empty_overlay = make_backend()
    for i in range(4):
        await primary.raw_put_checked(
            AssetPath(f"/a/{i}"), f"v{i}".encode(),
            options=WriteOptions(), request_hash=f"h{i}",
        )
    store = AssetStore(primary=primary, overlays=(ReadOnlyAssetBackend(empty_overlay),))

    items = await _drain(store, AssetPath("/a"), limit=2)
    paths = {item.path.value for item in items}
    assert paths == {f"/a/{i}" for i in range(4)}


@pytest.mark.asyncio
async def test_all_candidates_whiteout_yields_empty_listing(make_backend):
    """when EVERY overlay-only candidate under a path is whited-out by
    primary, the listing must terminate cleanly with zero items, not error
    or loop forever."""
    primary = make_backend()
    overlay = make_backend()
    for i in range(6):
        await overlay.raw_put(
            AssetPath(f"/w/{i}"), f"v{i}".encode(), content_type=None, metadata={}
        )
    store = AssetStore(primary=primary, overlays=(ReadOnlyAssetBackend(overlay),))
    for i in range(6):
        await store.put(AssetPath(f"/w/{i}"), b"tombstoned")
        await store.delete(AssetPath(f"/w/{i}"))

    items = await _drain(store, AssetPath("/w"), limit=2)
    assert items == []


@pytest.mark.asyncio
async def test_same_cursor_replay_is_consistent(make_backend):
    """calling list() twice with the SAME cursor (no mutation in between)
    must return identical items and the identical next cursor -- a resumed
    page is deterministic, not a one-shot consumable."""
    primary = make_backend()
    overlay = make_backend()
    for i in range(6):
        await primary.raw_put_checked(
            AssetPath(f"/a/{i}"), f"v{i}".encode(),
            options=WriteOptions(), request_hash=f"h{i}",
        )
    for i in range(6):
        await overlay.raw_put(
            AssetPath(f"/b/{i}"), f"v{i}".encode(), content_type=None, metadata={}
        )
    store = AssetStore(primary=primary, overlays=(ReadOnlyAssetBackend(overlay),))

    first_page = await store.list(AssetPath("/"), limit=3)
    replay_a = await store.list(AssetPath("/"), limit=3, cursor=first_page.cursor)
    replay_b = await store.list(AssetPath("/"), limit=3, cursor=first_page.cursor)

    assert [i.path.value for i in replay_a.items] == [i.path.value for i in replay_b.items]
    assert replay_a.cursor == replay_b.cursor
