#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression: two concurrent first-time idempotency claims on the same
``(scope, key)`` must never leak a raw ``IntegrityError``.

The unique ``(scope, key)`` constraint serializes the two concurrent fresh
INSERTs: the winner ACQUIRES; the loser's INSERT raises ``IntegrityError``.
``claim()`` must translate that exception into a stable disposition
(IN_PROGRESS / CONFLICT / REPLAY) by re-reading the winner's record in a fresh
session -- never propagate the database exception.

A bare ``asyncio.gather`` does not prove this: if the winner commits before
the loser's SELECT runs, the loser simply observes the committed row and
returns IN_PROGRESS without ever hitting the constraint -- so the test would
pass even with the recovery removed. Each test therefore forces both
fresh-INSERT flushes to rendezvous before either transaction commits, making
the loser deterministically collide. With the recovery removed the loser's
``IntegrityError`` propagates out of the gather and the test fails.
"""

import asyncio

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from linktools.ai.storage.sqlalchemy.idempotency import SqlAlchemyIdempotencyStore
from linktools.ai.storage.sqlalchemy.models import Base, ToolIdempotencyRow
from linktools.ai.tool.idempotency import ClaimDisposition


async def _make_store(tmp_path, db_name: str = "concurrent.db"):
    """Engine + schema + store. ``connect_args`` timeout -> sqlite busy-timeout
    so a blocked writer waits for the lock holder to commit instead of raising
    "database is locked"."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / db_name}",
        connect_args={"timeout": 30.0},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyIdempotencyStore(session_factory=session_factory)
    return engine, session_factory, store


def _force_fresh_insert_collision(monkeypatch) -> None:
    """Patch ``AsyncSession.flush`` so the first two flushes (the two claims'
    fresh-row INSERTs) rendezvous before either commits.

    By the time a claim reaches its flush it has already SELECTed, so both
    SELECTs observe no row; both then INSERT and exactly one hits the unique
    constraint. ``claim()`` opens one session per call and flushes exactly
    once on the fresh-INSERT path, so the first two flushes are precisely the
    two contending INSERTs. Any later flush (none on the recovery path here,
    which only SELECTs) passes through unmodified."""
    let_first_proceed = asyncio.Event()
    flushes = {"n": 0}
    real_flush = AsyncSession.flush

    async def coordinated(self, *args, **kwargs):
        flushes["n"] += 1
        if flushes["n"] == 1:
            # Block the first fresh INSERT until the second claim is also at
            # its flush, so neither transaction has committed yet.
            await asyncio.wait_for(let_first_proceed.wait(), timeout=5.0)
        elif flushes["n"] == 2:
            let_first_proceed.set()
        return await real_flush(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "flush", coordinated)


async def _row_count(session_factory) -> int:
    async with session_factory() as session:
        result = await session.execute(
            select(func.count()).select_from(ToolIdempotencyRow)
        )
        return int(result.scalar_one())


@pytest.mark.asyncio
async def test_concurrent_same_hash_claims_one_acquired_one_in_progress(
    tmp_path, monkeypatch
):
    _force_fresh_insert_collision(monkeypatch)
    engine, session_factory, store = await _make_store(tmp_path)

    async def claim(owner_id: str):
        return await store.claim(
            scope="tool:test",
            key="same-key",
            request_hash="same-hash",
            owner_id=owner_id,
        )

    first, second = await asyncio.gather(claim("worker-a"), claim("worker-b"))
    dispositions = {first.disposition, second.disposition}

    assert ClaimDisposition.ACQUIRED in dispositions, dispositions
    assert ClaimDisposition.IN_PROGRESS in dispositions, dispositions

    # The loser never persisted a duplicate -- exactly one row survives.
    assert await _row_count(session_factory) == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_different_hash_claims_one_acquired_one_conflict(
    tmp_path, monkeypatch
):
    _force_fresh_insert_collision(monkeypatch)
    engine, session_factory, store = await _make_store(tmp_path)

    async def claim(owner_id: str, request_hash: str):
        return await store.claim(
            scope="tool:test",
            key="same-key",
            request_hash=request_hash,
            owner_id=owner_id,
        )

    first, second = await asyncio.gather(
        claim("worker-a", "hash-a"),
        claim("worker-b", "hash-b"),
    )
    dispositions = {first.disposition, second.disposition}

    assert ClaimDisposition.ACQUIRED in dispositions, dispositions
    assert ClaimDisposition.CONFLICT in dispositions, dispositions

    assert await _row_count(session_factory) == 1

    await engine.dispose()
