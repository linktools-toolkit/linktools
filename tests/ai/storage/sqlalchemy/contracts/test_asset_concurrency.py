#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemy asset concurrency contract, parametrized over the supported
dialects (SQLite always; MySQL/PostgreSQL when their TEST_*_DSN env vars are
set). Covers the four concurrency scenarios: two concurrent creates on
the same path, two concurrent missing deletes, a concurrent create vs delete on
the same path, and two concurrent CAS updates holding the same expected
version."""

import asyncio

import pytest

from linktools.ai.asset.models import Depth, WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.errors import AssetPreconditionFailedError

pytestmark = pytest.mark.asyncio


def _put_opts(**kw):
    return WriteOptions(**kw)


async def _put(store, path, content=b"x", **kw):
    return await store.put(path, content, options=_put_opts(**kw))


async def test_concurrent_create_same_path_one_wins_no_duplicate(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/race-create.txt")

    results = await asyncio.gather(_put(store, path, b"a"), _put(store, path, b"b"), return_exceptions=True)
    # No exception escaped: both unconditional puts succeed (one creates, one
    # updates) and exactly one row survives.
    assert all(not isinstance(r, Exception) for r in results), results
    asset = await store.get(path)
    assert asset.content in (b"a", b"b")
    page = await sql_asset_backend.raw_list(
        AssetPath("/contract"), depth=Depth.ONE, limit=10, cursor=None
    )
    assert [i.path.value for i in page.items].count(path.value) == 1


async def test_concurrent_missing_delete_is_idempotent(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/never-existed.txt")

    results = await asyncio.gather(store.delete(path), store.delete(path), return_exceptions=True)
    assert all(not isinstance(r, Exception) for r in results), results
    assert await store.get(path) is None


async def test_concurrent_create_and_delete_on_same_path(sql_asset_backend):
    # A create racing a delete on the same path must leave the store in a
    # consistent state: the delete of a not-yet-committed row is a no-op (or a
    # create-then-delete tombstone), and the create's row never duplicates. No
    # exception escapes either way.
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/race-create-delete.txt")

    results = await asyncio.gather(
        _put(store, path, b"created"),
        store.delete(path),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in results), results
    page = await sql_asset_backend.raw_list(
        AssetPath("/contract"), depth=Depth.ONE, limit=10, cursor=None
    )
    # At most one live (non-tombstone) row for the path -- no duplicate.
    live = [i for i in page.items if i.path.value == path.value and not i.is_tombstone]
    assert len(live) <= 1


async def test_concurrent_cas_same_expected_version_one_wins(sql_asset_backend):
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/race-cas.txt")
    await _put(store, path, b"v1")
    info = await sql_asset_backend.raw_stat(path)

    results = await asyncio.gather(
        store.put(path, b"a", options=_put_opts(if_match=info.etag)),
        store.put(path, b"b", options=_put_opts(if_match=info.etag)),
        return_exceptions=True,
    )
    precondition_failed = [r for r in results if isinstance(r, AssetPreconditionFailedError)]
    # Both held the same stale etag; the DB serializes them so exactly one wins
    # and the other sees a precondition failure.
    assert len(precondition_failed) == 1, results
    assert (await store.get(path)).content in (b"a", b"b")
