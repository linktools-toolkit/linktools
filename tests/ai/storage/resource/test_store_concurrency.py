#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TOCTOU regression test for ResourceStore.put on SqlAlchemyResourceBackend.

The atomic ``raw_put_checked`` (spec section 16) folds precondition-check +
idempotency-reservation + mutate into ONE transaction. Two concurrent puts with
the same ``if_none_match`` precondition therefore cannot both succeed: the
unique ``path`` constraint backstops the atomic check, and a concurrent insert
that loses the race surfaces as ResourcePreconditionFailedError (translated from
the IntegrityError) regardless of how the two transactions interleaved. This
file is SqlAlchemy-only -- the File backend's atomicity is best-effort under an
in-process lock and the Memory backend has no real concurrency story, so neither
is exercised here."""
import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import ResourcePreconditionFailedError
from linktools.ai.storage.resource.models import WriteOptions
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.resource.store import ResourceStore
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.resource import SqlAlchemyResourceBackend


@pytest.mark.asyncio
async def test_concurrent_put_if_none_match_exactly_one_succeeds(tmp_path):
    """Two concurrent puts with if_none_match=True on a fresh path: exactly one
    succeeds, the other raises ResourcePreconditionFailedError. This is the
    TOCTOU guarantee -- without the atomic check+put the second could also pass
    the precondition (both read empty) and then one would hit a raw
    IntegrityError (or, pre-fix, both believe they created the resource)."""
    db_path = tmp_path / "concurrency.db"
    # connect_args timeout -> sqlite busy-timeout so a blocked writer waits for
    # the lock holder to commit instead of raising "database is locked".
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"timeout": 30.0},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    backend = SqlAlchemyResourceBackend(session_factory=session_factory)
    store = ResourceStore(primary=backend)
    path = ResourcePath("/concurrent.txt")

    results = {"success": 0, "conflict": 0}

    async def attempt(payload: bytes) -> None:
        try:
            await store.put(path, payload, options=WriteOptions(if_none_match=True))
            results["success"] += 1
        except ResourcePreconditionFailedError:
            results["conflict"] += 1

    await asyncio.gather(attempt(b"payload-a"), attempt(b"payload-b"))

    assert results["success"] == 1, "exactly one concurrent put must succeed"
    assert results["conflict"] == 1, "the loser must surface as a precondition conflict"

    # Final state: the resource exists with exactly one of the two payloads.
    resource = await store.get(path)
    assert resource is not None
    assert resource.content in (b"payload-a", b"payload-b")

    await engine.dispose()
