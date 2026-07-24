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


# -- Work package 3 (spec section 6.6): no-op version-fence regressions -----


async def test_noop_put_does_not_lose_a_concurrent_write(sql_asset_backend):
    """Lost-write scenario: T1 reads A (content matches -> would take the
    no-op short-circuit), T2 writes B and commits first, T1's no-op version
    fence must then fail (the row changed under it) and T1 must retry -- not
    silently return as if A were still current. The final state must be B,
    never A re-asserted by a stale no-op return."""
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/noop-lost-write.txt")
    await _put(store, path, b"A")

    # T1: a same-content PUT of "A" (would take the no-op branch) racing T2's
    # real content change to "B". Both issued concurrently; regardless of
    # scheduling, once both complete the live content must be exactly what
    # the LAST successful write produced, never a value silently reasserted
    # from a stale read.
    results = await asyncio.gather(
        _put(store, path, b"A"),
        _put(store, path, b"B"),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in results), results
    final = await store.get(path)
    assert final.content == b"B", (
        "a same-content no-op PUT raced against a real write must never win "
        "by returning a stale pre-race value"
    )


async def test_same_value_concurrent_put_succeeds_without_error(sql_asset_backend):
    """Two writers PUT the identical content concurrently: both must succeed
    with no exception, and the version-fence retry (if the no-op fence loses
    the race to the other writer's real first write) converges rather than
    deadlocking or raising a spurious conflict."""
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/noop-same-value.txt")

    results = await asyncio.gather(
        _put(store, path, b"same"),
        _put(store, path, b"same"),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in results), results
    final = await store.get(path)
    assert final.content == b"same"


async def test_noop_fence_does_not_bypass_stale_cas(sql_asset_backend):
    """An old ``expected_version``/etag must never pass through the no-op
    branch: a same-content PUT issued with an if_match that is no longer the
    live etag (because a real write already advanced it) must raise a
    precondition failure, not silently succeed via the version fence."""
    store = AssetStore(primary=sql_asset_backend)
    path = AssetPath("/contract/noop-stale-cas.txt")
    await _put(store, path, b"v1")
    stale_info = await sql_asset_backend.raw_stat(path)

    # Advance the row for real, invalidating stale_info.etag.
    await _put(store, path, b"v2")

    with pytest.raises(AssetPreconditionFailedError):
        # Same content as the (now stale) v1 read, but stamped with the old
        # etag -- must fail the precondition, never take the no-op fast path.
        await store.put(
            path, b"v1", options=_put_opts(if_match=stale_info.etag)
        )
    assert (await store.get(path)).content == b"v2"
