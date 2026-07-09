#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_idempotency_store_contract.py — runs the
same IdempotencyStore contract against both FileIdempotencyStore and
SqlAlchemyIdempotencyStore (review doc §11 backend parity). The
parametrized ``store_factory`` fixture mirrors test_approval_store_contract.py
/ test_memory_store_contract.py (file + sqlalchemy branches, including
``_run_in_new_loop`` to bootstrap the SQL engine off the test loop).

The contract covers (per §11.2):
1. reserve -> complete -> get returns COMPLETED with cached result.
2. reserve same (scope, key) with a different request_hash -> IdempotencyConflictError.
3. Two fresh reservations with different (scope, key) coexist independently.
4. fail marks the record FAILED with the serialized error string.
5. RESERVED is the status reserve leaves a fresh record in (the caller's
   signal to proceed + complete/fail).

Uses ``def test_x(store_factory): asyncio.run(_run())`` style — sync test
wrapper driving its own event loop, no pytest-asyncio needed."""
import asyncio

import pytest

from linktools.ai.errors import IdempotencyConflictError
from linktools.ai.storage.file.idempotency import FileIdempotencyStore
from linktools.ai.tool.idempotency import IdempotencyStatus


# ---------------------------------------------------------------------------
# Parametrized store factory. The SQL branch (incl. ``_run_in_new_loop``) is
# copied verbatim from test_approval_store_contract.py / test_memory_store_contract.py.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileIdempotencyStore(root=tmp_path / f"idem-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.idempotency import SqlAlchemyIdempotencyStore

    counter = {"n": 0}
    engines = []

    def _run_in_new_loop(coro):
        # Called synchronously from inside an already-running pytest-asyncio
        # loop, so we cannot use asyncio.get_event_loop().run_until_complete()
        # here. Run the setup on a separate thread with its own fresh loop.
        import asyncio
        import threading

        outcome = {}

        def _runner():
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                outcome["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def sqlalchemy_factory():
        counter["n"] += 1
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/idem-db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                # ToolIdempotencyRow subclasses Base, so a single create_all
                # covers every table both backends need.
                await conn.run_sync(Base.metadata.create_all)
            # Dispose the pool so later ops open fresh connections on the
            # test's own loop instead of reusing one bound to this thread.
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyIdempotencyStore(session_factory=session_factory)

    def _dispose_engines():
        for engine in engines:
            _run_in_new_loop(engine.dispose())

    request.addfinalizer(_dispose_engines)

    return sqlalchemy_factory


# ---------------------------------------------------------------------------
# 1. reserve -> complete -> get: COMPLETED status, cached result round-trips,
#    completed_at is set + tz-aware.
# ---------------------------------------------------------------------------


def test_reserve_complete_get_roundtrip(store_factory):
    store = store_factory()

    async def _run():
        # Fresh reservation: reserve returns None (caller should proceed).
        existing = await store.reserve("scope-1", "key-1", "hash-1")
        assert existing is None, "fresh reserve must return None"
        # The persisted record is in the RESERVED state.
        record = await store.get("scope-1", "key-1")
        assert record is not None
        assert record.status is IdempotencyStatus.RESERVED
        assert record.scope == "scope-1"
        assert record.key == "key-1"
        assert record.request_hash == "hash-1"
        assert record.result is None
        assert record.error is None
        assert record.completed_at is None
        assert record.created_at.tzinfo is not None
        # Complete the reservation.
        await store.complete("scope-1", "key-1", {"output": "ok", "n": 42})
        completed = await store.get("scope-1", "key-1")
        assert completed is not None
        assert completed.status is IdempotencyStatus.COMPLETED
        assert completed.result == {"output": "ok", "n": 42}
        assert completed.error is None
        assert completed.completed_at is not None
        assert completed.completed_at.tzinfo is not None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. reserve same (scope, key) with a different hash -> IdempotencyConflictError.
#    Same hash returns the existing record (so the caller can branch on status).
# ---------------------------------------------------------------------------


def test_reserve_same_scope_key_with_different_hash_raises_conflict(store_factory):
    store = store_factory()

    async def _run():
        await store.reserve("scope-1", "key-1", "hash-a")
        # Different hash on the same (scope, key) -> conflict.
        with pytest.raises(IdempotencyConflictError):
            await store.reserve("scope-1", "key-1", "hash-b")
        # Same hash on the same (scope, key) -> returns the existing RESERVED
        # record (caller branches on status, not a conflict).
        existing = await store.reserve("scope-1", "key-1", "hash-a")
        assert existing is not None
        assert existing.status is IdempotencyStatus.RESERVED

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Two fresh reservations with different (scope, key) coexist independently.
#    Same key under different scopes does NOT collide.
# ---------------------------------------------------------------------------


def test_different_scope_or_key_coexist(store_factory):
    store = store_factory()

    async def _run():
        await store.reserve("scope-1", "key-1", "hash-a")
        await store.reserve("scope-2", "key-1", "hash-b")  # different scope
        await store.reserve("scope-1", "key-2", "hash-c")  # different key
        # All three are present and addressable by their (scope, key).
        r1 = await store.get("scope-1", "key-1")
        r2 = await store.get("scope-2", "key-1")
        r3 = await store.get("scope-1", "key-2")
        assert r1 is not None and r2 is not None and r3 is not None
        assert {r.request_hash for r in (r1, r2, r3)} == {"hash-a", "hash-b", "hash-c"}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. fail marks the record FAILED with the serialized error string. A FAILED
#    record allows retry: a later reserve with the SAME hash returns the
#    FAILED record (caller proceeds), and complete() then transitions it to
#    COMPLETED.
# ---------------------------------------------------------------------------


def test_fail_marks_failed_and_retry_can_complete(store_factory):
    store = store_factory()

    async def _run():
        await store.reserve("scope-1", "key-1", "hash-a")
        await store.fail("scope-1", "key-1", "boom: transient")
        failed = await store.get("scope-1", "key-1")
        assert failed is not None
        assert failed.status is IdempotencyStatus.FAILED
        assert failed.error == "boom: transient"
        assert failed.result is None
        assert failed.completed_at is not None
        # Retry path: reserve with the SAME hash returns the FAILED record
        # (caller branches on status -> FAILED means "retry allowed").
        existing = await store.reserve("scope-1", "key-1", "hash-a")
        assert existing is not None
        assert existing.status is IdempotencyStatus.FAILED
        # Caller re-executes and completes -- the FAILED record is
        # overwritten with COMPLETED.
        await store.complete("scope-1", "key-1", {"ok": True})
        final = await store.get("scope-1", "key-1")
        assert final is not None
        assert final.status is IdempotencyStatus.COMPLETED
        assert final.result == {"ok": True}
        assert final.error is None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. complete/fail are no-ops on missing (scope, key) -- defensive against the
#    race where the record was evicted between reserve and complete/fail.
# ---------------------------------------------------------------------------


def test_complete_and_fail_on_missing_record_are_noops(store_factory):
    store = store_factory()

    async def _run():
        # Should not raise.
        await store.complete("missing-scope", "missing-key", {"x": 1})
        await store.fail("missing-scope", "missing-key", "error")
        assert await store.get("missing-scope", "missing-key") is None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. File-only: path-traversal in scope or key -> ValueError. (SQL ids are
#    opaque column values, not path segments, so this guard is
#    FileIdempotencyStore-specific -- mirrors the file-only path-traversal
#    test in test_approval_store_contract.py.)
# ---------------------------------------------------------------------------


def test_path_traversal_in_scope_or_key_is_rejected(tmp_path):
    store = FileIdempotencyStore(root=tmp_path)

    async def _run():
        with pytest.raises(ValueError):
            await store.reserve("../evil", "key", "h")
        with pytest.raises(ValueError):
            await store.reserve("scope", "../evil", "h")
        with pytest.raises(ValueError):
            await store.get("../evil", "key")
        # ``..`` alone is rejected as a path segment -- would resolve to the
        # parent directory if written verbatim.
        with pytest.raises(ValueError):
            await store.complete("scope", "..", "result")

    asyncio.run(_run())
