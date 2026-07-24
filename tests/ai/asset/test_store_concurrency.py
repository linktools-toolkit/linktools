#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TOCTOU + lost-update regression tests for AssetStore on SqlAlchemyAssetBackend.

The atomic ``raw_put_checked`` folds precondition-check +
idempotency-reservation + mutate into ONE transaction. The conditional UPDATE
WHERE version=expected (contract) and If-Match in the UPDATE WHERE (spec
contract) make lost updates impossible at the DB level. The atomic revision
increment UPDATE...RETURNING (contract) makes revision lost-update impossible
too. These tests prove each guarantee by firing concurrent writes that, under
the pre-fix Python read-then-write code, would have produced duplicate versions,
duplicate revisions, or both-pass-a-precondition.

This file is SqlAlchemy-only -- the File backend's atomicity is best-effort
under an in-process lock and the Memory backend has no real concurrency story,
so neither is exercised here."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import AssetPreconditionFailedError
from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.asset import SqlAlchemyAssetBackend


async def _make_store(tmp_path, db_name: str = "concurrency.db"):
    """Engine + schema + backend + store. connect_args timeout -> sqlite
    busy-timeout so a blocked writer waits for the lock holder to commit instead
    of raising "database is locked"."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / db_name}",
        connect_args={"timeout": 30.0},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    backend = SqlAlchemyAssetBackend(session_factory=session_factory)
    store = AssetStore(primary=backend)
    return engine, backend, store


@pytest.mark.asyncio
async def test_concurrent_put_if_none_match_exactly_one_succeeds(tmp_path):
    """Two concurrent puts with if_none_match=True on a fresh path: exactly one
    succeeds, the other raises AssetPreconditionFailedError. This is the
    TOCTOU guarantee -- without the atomic check+put the second could also pass
    the precondition (both read empty) and then one would hit a raw
    IntegrityError (or, pre-fix, both believe they created the asset)."""
    engine, backend, store = await _make_store(tmp_path)
    path = AssetPath("/concurrent.txt")

    results = {"success": 0, "conflict": 0}

    async def attempt(payload: bytes) -> None:
        try:
            await store.put(path, payload, options=WriteOptions(if_none_match=True))
            results["success"] += 1
        except AssetPreconditionFailedError:
            results["conflict"] += 1

    await asyncio.gather(attempt(b"payload-a"), attempt(b"payload-b"))

    assert results["success"] == 1, "exactly one concurrent put must succeed"
    assert results["conflict"] == 1, "the loser must surface as a precondition conflict"

    # Final state: the asset exists with exactly one of the two payloads.
    asset = await store.get(path)
    assert asset is not None
    assert asset.content in (b"payload-a", b"payload-b")

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_unconditional_puts_produce_distinct_versions(tmp_path):
    """contract lost-update guard: two concurrent UNCONDITIONAL puts on the same
    existing path must produce distinct, monotonic versions.

    Under the pre-fix Python read-then-write, both writers would SELECT the same
    version (1), both compute new_version=2, and both UPDATE -- the second
    overwrites the first (version stays 2, lost update). With the conditional
    UPDATE WHERE version=expected, only one writer can transition the row from
    v=1; the loser sees rowcount==0, retries against the new committed state,
    and lands on v=3. Net result: versions are distinct."""
    engine, backend, store = await _make_store(tmp_path)
    path = AssetPath("/counter.txt")

    # Seed the asset at version 1.
    first = await store.put(path, b"v0")
    assert first.info.version == 1

    # Two concurrent unconditional updates with different payloads.
    async def update(payload: bytes):
        return await store.put(path, payload)

    results = await asyncio.gather(update(b"a"), update(b"b"))
    versions = sorted(r.info.version for r in results)

    assert versions == [2, 3], (
        f"concurrent updates must produce distinct monotonic versions, got {versions}"
    )

    # Final content matches one of the two payloads; final version is the max.
    asset = await store.get(path)
    assert asset is not None
    assert asset.info.version == 3
    assert asset.content in (b"a", b"b")

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_put_with_same_if_match_exactly_one_succeeds(tmp_path):
    """contract If-Match-in-WHERE guard: two concurrent puts carrying the SAME
    If-Match (etag of the current asset) on an EXISTING asset -- exactly
    one succeeds, the other raises AssetPreconditionFailedError.

    Under the pre-fix Python pre-read check, both writers would SELECT the row,
    both see etag=X, both pass the Python check, and both UPDATE -- last write
    wins, the loser silently loses. Pushing If-Match into the UPDATE WHERE
    clause makes the etag check atomic with the write: the second UPDATE
    re-evaluates etag under the row lock and finds it no longer matches."""
    engine, backend, store = await _make_store(tmp_path)
    path = AssetPath("/ifmatch.txt")

    # Seed at version 1, capture its etag.
    initial = await store.put(path, b"original")
    etag = initial.info.etag

    results = {"success": 0, "conflict": 0}
    final_versions = []

    async def attempt(payload: bytes) -> None:
        try:
            r = await store.put(path, payload, options=WriteOptions(if_match=etag))
            results["success"] += 1
            final_versions.append(r.info.version)
        except AssetPreconditionFailedError:
            results["conflict"] += 1

    await asyncio.gather(attempt(b"payload-a"), attempt(b"payload-b"))

    assert results["success"] == 1, (
        f"exactly one concurrent if-match put must succeed, got {results}"
    )
    assert results["conflict"] == 1, (
        f"the loser must surface as a precondition conflict, got {results}"
    )
    # The winner bumped version exactly once (1 -> 2), not twice.
    assert final_versions == [2], (
        f"winner must produce version 2 (one bump from v=1), got {final_versions}"
    )

    # Final state: the asset exists with the winner's payload at version 2.
    asset = await store.get(path)
    assert asset is not None
    assert asset.info.version == 2
    assert asset.content in (b"payload-a", b"payload-b")

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_puts_produce_distinct_revisions(tmp_path):
    """contract atomic-revision guard: N concurrent puts to N distinct paths must
    produce exactly N revision increments -- no lost updates on the counter.

    Under the pre-fix Python read-then-write (``row.value += 1``), N concurrent
    transactions could each read the same counter value, each add 1, and each
    write back the same number -- losing N-1 increments. The atomic
    UPDATE...SET value = value + 1 RETURNING value makes each bump server-side
    and serial: the final counter == initial + N exactly."""
    engine, backend, store = await _make_store(tmp_path)

    n = 8
    paths = [AssetPath(f"/r/{i}.txt") for i in range(n)]

    # Establish baseline revision (should be 0 on a fresh DB).
    baseline = int(await backend.revision())
    assert baseline == 0

    # Fire N concurrent puts to distinct paths -- each must bump the revision
    # exactly once. Distinct paths avoid any path-lock contention, isolating
    # the test to the revision counter's atomicity.
    async def one_put(p: AssetPath) -> None:
        await store.put(p, b"x")

    await asyncio.gather(*(one_put(p) for p in paths))

    after = int(await backend.revision())
    assert after == n, (
        f"revision must advance by exactly {n} under {n} concurrent puts "
        f"(baseline={baseline}, after={after}); a smaller delta means the "
        "atomic UPDATE...RETURNING was bypassed by a Python read-then-write"
    )

    # Sanity: all N assets exist, each at version 1.
    for p in paths:
        r = await store.get(p)
        assert r is not None and r.info.version == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_revision_increments_monotonically_under_sequential_puts(tmp_path):
    """contract sanity: a sequence of puts (no concurrency) bumps the revision
    counter by exactly one per put. Catches regressions where the atomic
    UPDATE...RETURNING seeds id=1 with value=1 on the first call (which would
    read as revision=1 BEFORE any put, breaking monotonicity vs. the
    baseline=0 the test above asserts)."""
    engine, backend, store = await _make_store(tmp_path)

    assert int(await backend.revision()) == 0
    await store.put(AssetPath("/a.txt"), b"a")
    assert int(await backend.revision()) == 1
    await store.put(AssetPath("/b.txt"), b"b")
    assert int(await backend.revision()) == 2
    # Update an existing asset -- still bumps the revision (contract: a real
    # change is a change).
    await store.put(AssetPath("/a.txt"), b"a2")
    assert int(await backend.revision()) == 3

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_delete_with_same_if_match_exactly_one_succeeds(tmp_path):
    """contract If-Match guard on DELETE: two concurrent deletes of the same live
    asset, both with the same If-Match -- exactly one wins. The loser's
    conditional UPDATE...WHERE etag=:if_match AND deleted_at IS NULL misses
    (the row is now masked) and raises AssetPreconditionFailedError."""
    engine, backend, store = await _make_store(tmp_path)
    path = AssetPath("/del.txt")

    initial = await store.put(path, b"original")
    etag = initial.info.etag

    results = {"success": 0, "conflict": 0}

    async def attempt() -> None:
        try:
            await store.delete(path, options=WriteOptions(if_match=etag))
            results["success"] += 1
        except AssetPreconditionFailedError:
            results["conflict"] += 1

    await asyncio.gather(attempt(), attempt())

    assert results["success"] == 1, (
        f"exactly one concurrent if-match delete must succeed, got {results}"
    )
    assert results["conflict"] == 1, (
        f"the loser must surface as a precondition conflict, got {results}"
    )

    # Final state: the asset is gone from the reader's view.
    assert await store.get(path) is None

    await engine.dispose()


@pytest.mark.asyncio
async def test_revision_bump_uses_server_side_atomic_increment(tmp_path):
    """contract structural guard: the revision counter bump must be a single
    server-side atomic increment ``UPDATE ai_asset_revision SET value = value +
    1 WHERE id = ?`` (NOT a Python read-then-write like ``row.value += 1`` then
    an unconditional ``UPDATE ... SET value = ?``). The new value is read back
    with a separate SELECT rather than RETURNING, because MySQL lacks
    UPDATE...RETURNING -- the read-back runs inside the same transaction (the
    row lock is held), so it observes exactly this writer's increment.

    SQLite's deferred-transaction snapshot isolation masks the lost-update bug
    behaviorally (a concurrent reader's SELECT blocks behind a writer's lock, so
    it usually re-evaluates against the new committed value rather than reading
    stale). On MVCC databases (PostgreSQL et al.) the bug is silent and real.
    This structural test therefore catches the regression that the behavioral
    concurrency tests cannot reliably reproduce under SQLite."""
    from sqlalchemy import event

    engine, backend, store = await _make_store(tmp_path)
    captured: "list[str]" = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):
        captured.append(statement)

    # Seed the counter, then bump it via a real put.
    await store.put(AssetPath("/seed.txt"), b"x")
    captured.clear()
    await store.put(AssetPath("/bump.txt"), b"y")

    revision_bumps = [
        s
        for s in captured
        if "ai_asset_revision" in s
        and ("value + " in s.lower() or "value+" in s.lower())
    ]
    assert revision_bumps, "expected at least one atomic increment on ai_asset_revision"
    for stmt in revision_bumps:
        assert stmt.lower().startswith("update"), (
            f"revision bump must be an UPDATE (not a Python-computed write), "
            f"got: {stmt!r}"
        )
        assert "value + " in stmt.lower() or "value+" in stmt.lower(), (
            f"revision bump must use server-side atomic increment (value = value + 1), "
            f"got: {stmt!r}"
        )

    await engine.dispose()


@pytest.mark.asyncio
async def test_asset_update_uses_conditional_where_on_version(tmp_path):
    """contract structural guard: an update-existing PUT must emit a single
    ``UPDATE ai_assets SET ... WHERE path = :path AND version = :expected``
    -- NOT an unconditional ``UPDATE ... SET version = ?`` computed in Python.

    The behavioral test (test_concurrent_unconditional_puts_produce_distinct_versions)
    already catches this regression in SQLite; this structural test pins the
    exact SQL shape so the regression cannot hide behind SQLite scheduling
    variance on a future refactor."""
    from sqlalchemy import event

    engine, backend, store = await _make_store(tmp_path)
    captured: "list[str]" = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):
        captured.append(statement)

    # Seed at v=1, then update to v=2.
    await store.put(AssetPath("/u.txt"), b"v1")
    captured.clear()
    await store.put(AssetPath("/u.txt"), b"v2")

    asset_updates = [
        s for s in captured if "ai_assets" in s and s.upper().startswith("UPDATE")
    ]
    assert asset_updates, "expected at least one UPDATE on ai_assets"
    stmt = asset_updates[-1]
    assert "version" in stmt.lower() and "path" in stmt.lower(), (
        f"asset UPDATE must condition on path AND version, got: {stmt!r}"
    )
    # The WHERE clause must reference version (the optimistic-concurrency guard),
    # not just path. We check for the column name appearing in the WHERE region
    # of the statement -- a rough heuristic but sufficient to catch a regression
    # to an unconditional UPDATE-by-path.
    where_clause = stmt.split("WHERE", 1)[-1] if "WHERE" in stmt.upper() else ""
    assert "version" in where_clause.lower(), (
        f"asset UPDATE WHERE clause must include version, got: {stmt!r}"
    )

    await engine.dispose()
