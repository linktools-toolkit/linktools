#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_idempotency_store_contract.py — runs the
same fenced-claim IdempotencyStore contract against both FileIdempotencyStore
and SqlAlchemyIdempotencyStore (backend parity).

The contract covers (per the WP-10 claim/owner/generation/lease model):
1. claim -> complete(claim) -> get returns COMPLETED with cached result.
2. claim same (scope, key) with a different request_hash -> CONFLICT.
3. Two fresh claims with different (scope, key) coexist independently.
4. fail(claim) marks FAILED; a later claim with the SAME hash ACQUIRES a new
   generation (retry), and complete(claim) then transitions it to COMPLETED.
5. complete/fail are no-ops when the (scope, key) is missing or the claim's
   owner/generation no longer matches (stale-worker fencing).

Uses ``def test_x(store_factory): asyncio.run(_run())`` style."""

import asyncio

import pytest

from linktools.ai.storage.file.idempotency import FileIdempotencyStore
from linktools.ai.tool.idempotency import ClaimDisposition, IdempotencyStatus


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
        import threading

        outcome = {}

        def _runner():
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
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
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyIdempotencyStore(session_factory=session_factory)

    def _dispose_engines():
        for engine in engines:
            _run_in_new_loop(engine.dispose())

    request.addfinalizer(_dispose_engines)
    return sqlalchemy_factory


def test_claim_complete_get_roundtrip(store_factory):
    store = store_factory()

    async def _run():
        result = await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-1", owner_id="own-1"
        )
        assert result.disposition is ClaimDisposition.ACQUIRED
        assert result.claim is not None
        record = await store.get("scope-1", "key-1")
        assert record is not None
        assert record.status is IdempotencyStatus.RESERVED
        assert record.owner_id == "own-1"
        assert record.generation == 1
        assert record.lease_expires_at is not None
        await store.complete(result.claim, {"output": "ok", "n": 42})
        completed = await store.get("scope-1", "key-1")
        assert completed is not None
        assert completed.status is IdempotencyStatus.COMPLETED
        assert completed.result == {"output": "ok", "n": 42}

    asyncio.run(_run())


def test_claim_same_scope_key_with_different_hash_is_conflict(store_factory):
    store = store_factory()

    async def _run():
        await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-a", owner_id="own-1"
        )
        second = await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-b", owner_id="own-2"
        )
        assert second.disposition is ClaimDisposition.CONFLICT
        # Same hash, different owner, lease still valid -> IN_PROGRESS.
        third = await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-a", owner_id="own-2"
        )
        assert third.disposition is ClaimDisposition.IN_PROGRESS

    asyncio.run(_run())


def test_different_scope_or_key_coexist(store_factory):
    store = store_factory()

    async def _run():
        await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-a", owner_id="o"
        )
        await store.claim(
            scope="scope-2", key="key-1", request_hash="hash-b", owner_id="o"
        )
        await store.claim(
            scope="scope-1", key="key-2", request_hash="hash-c", owner_id="o"
        )
        r1 = await store.get("scope-1", "key-1")
        r2 = await store.get("scope-2", "key-1")
        r3 = await store.get("scope-1", "key-2")
        assert r1 is not None and r2 is not None and r3 is not None
        assert {r.request_hash for r in (r1, r2, r3)} == {"hash-a", "hash-b", "hash-c"}

    asyncio.run(_run())


def test_fail_then_retry_claim_can_complete(store_factory):
    store = store_factory()

    async def _run():
        first = await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-a", owner_id="own-1"
        )
        await store.fail(first.claim, "boom: transient")
        failed = await store.get("scope-1", "key-1")
        assert failed.status is IdempotencyStatus.FAILED
        assert failed.error == "boom: transient"
        # Retry: a new owner claims a fresh generation (FAILED -> retry).
        retry = await store.claim(
            scope="scope-1", key="key-1", request_hash="hash-a", owner_id="own-2"
        )
        assert retry.disposition is ClaimDisposition.ACQUIRED
        assert retry.claim.generation == 2
        await store.complete(retry.claim, {"ok": True})
        final = await store.get("scope-1", "key-1")
        assert final.status is IdempotencyStatus.COMPLETED
        assert final.result == {"ok": True}

    asyncio.run(_run())


def test_complete_and_fail_on_missing_or_stale_claim_are_rejected(store_factory):
    """complete/fail against a missing record, or with a stale owner/generation,
    raise LostIdempotencyClaimError (C-01 §6.7) -- never silently succeed."""
    from linktools.ai.errors import LostIdempotencyClaimError
    from linktools.ai.tool.idempotency import IdempotencyClaim

    store = store_factory()

    async def _run():
        from datetime import datetime, timezone

        ghost = IdempotencyClaim(
            scope="missing-scope",
            key="missing-key",
            request_hash="h",
            owner_id="ghost",
            generation=1,
            claimed_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc),
        )
        with pytest.raises(LostIdempotencyClaimError):
            await store.complete(ghost, {"x": 1})
        with pytest.raises(LostIdempotencyClaimError):
            await store.fail(ghost, "error")
        assert await store.get("missing-scope", "missing-key") is None
        # Stale-owner fencing: a second owner steals the lease, then the first
        # owner's complete()/fail() must be rejected (not overwrite owner-2).
        first = await store.claim(
            scope="scope-s",
            key="key-s",
            request_hash="h",
            owner_id="own-1",
            lease_seconds=0.01,
        )
        await asyncio.sleep(0.05)
        second = await store.claim(
            scope="scope-s", key="key-s", request_hash="h", owner_id="own-2"
        )
        assert second.disposition is ClaimDisposition.ACQUIRED
        with pytest.raises(LostIdempotencyClaimError):
            await store.complete(first.claim, {"stale": True})
        with pytest.raises(LostIdempotencyClaimError):
            await store.fail(first.claim, "stale")
        record = await store.get("scope-s", "key-s")
        assert record.status is IdempotencyStatus.RESERVED  # owner-2 still holds
        assert record.result is None

    asyncio.run(_run())


def test_path_traversal_in_scope_or_key_is_rejected(tmp_path):
    store = FileIdempotencyStore(root=tmp_path)

    async def _run():
        with pytest.raises(ValueError):
            await store.claim(
                scope="../evil", key="key", request_hash="h", owner_id="o"
            )
        with pytest.raises(ValueError):
            await store.claim(
                scope="scope", key="../evil", request_hash="h", owner_id="o"
            )
        with pytest.raises(ValueError):
            await store.get("../evil", "key")

    asyncio.run(_run())


def test_completed_record_cannot_be_reclaimed(store_factory):
    """C-01 §6.8: a COMPLETED record must never be flipped back to CLAIMED/
    RESERVED. A later claim returns REPLAY (the cached result), not ACQUIRED --
    even though the CAS only pins generation, the status clause in the WHERE
    blocks the re-claim."""
    store = store_factory()

    async def _run():
        first = await store.claim(
            scope="scope-c", key="key-c", request_hash="h", owner_id="own-1"
        )
        await store.complete(first.claim, {"done": True})
        # A second claim on the COMPLETED record must REPLAY, not re-acquire.
        again = await store.claim(
            scope="scope-c", key="key-c", request_hash="h", owner_id="own-2"
        )
        assert again.disposition is ClaimDisposition.REPLAY, again.disposition
        assert again.record.result == {"done": True}
        # And the persisted record is still COMPLETED.
        record = await store.get("scope-c", "key-c")
        assert record.status is IdempotencyStatus.COMPLETED

    asyncio.run(_run())
