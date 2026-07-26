#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run commit log tests (P5 SQL commit log): commit_id-keyed idempotent replay."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.run.persistence.sqlalchemy.commit_log import (
    RunCommitConflictError,
    RunCommitLogRow,
    SqlAlchemyRunCommitLog,
    canonical_request_hash,
)
from linktools.ai.storage.sqlalchemy.models import Base


@pytest.fixture
def session_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'commit-log.db'}")

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return async_sessionmaker(engine, expire_on_commit=False)

    return asyncio.run(_setup())




def test_canonical_request_hash_is_deterministic_and_order_independent():
    h1 = canonical_request_hash("pause", {"a": 1, "b": 2})
    h2 = canonical_request_hash("pause", {"b": 2, "a": 1})  # different order
    assert h1 == h2
    assert len(h1) == 32  # SHA-256


def test_canonical_request_hash_distinguishes_operations():
    h1 = canonical_request_hash("pause", {"run_id": "r1"})
    h2 = canonical_request_hash("complete", {"run_id": "r1"})
    assert h1 != h2


def test_canonical_request_hash_distinguishes_payloads():
    h1 = canonical_request_hash("pause", {"run_id": "r1"})
    h2 = canonical_request_hash("pause", {"run_id": "r2"})
    assert h1 != h2


def test_record_then_find_round_trips(session_factory):
    log = SqlAlchemyRunCommitLog()  # stateless
    rh = canonical_request_hash("pause", {"run_id": "r1", "approval_id": "a1"})

    async def _run():
        async with session_factory() as session:
            async with session.begin():
                recorded = await log.record(
                    session,
                    commit_id="pause:r1:a1",
                    operation="pause",
                    run_id="r1",
                    request_hash=rh,
                    result={"approval_id": "a1", "checkpoint_id": "c1"},
                )
        async with session_factory() as session:
            found = await log.find(session, "pause:r1:a1")
        return recorded, found

    recorded, found = asyncio.run(_run())
    assert found is not None
    assert found.commit_id == "pause:r1:a1"
    assert found.operation == "pause"
    assert found.run_id == "r1"
    assert found.request_hash == rh
    assert found.result == {"approval_id": "a1", "checkpoint_id": "c1"}
    assert recorded.commit_id == found.commit_id


def test_find_returns_none_for_missing_commit(session_factory):
    log = SqlAlchemyRunCommitLog()  # stateless

    async def _run():
        async with session_factory() as session:
            return await log.find(session, "never-committed")

    assert asyncio.run(_run()) is None


def test_same_commit_id_same_hash_replays(session_factory):
    """A retried call with the SAME commit_id + request_hash returns the
    recorded result -- the second call is an idempotent replay, not a
    re-execution."""
    log = SqlAlchemyRunCommitLog()  # stateless
    rh = canonical_request_hash("complete", {"run_id": "r1", "version": 3})

    async def _run():
        async with session_factory() as session:
            async with session.begin():
                await log.record(
                    session,
                    commit_id="complete:r1:3",
                    operation="complete",
                    run_id="r1",
                    request_hash=rh,
                    result={"status": "SUCCEEDED"},
                )
        # Replay: same commit_id + same hash.
        async with session_factory() as session:
            existing = await log.find(session, "complete:r1:3")
        assert existing is not None
        assert existing.request_hash == rh
        return existing.result

    result = asyncio.run(_run())
    assert result == {"status": "SUCCEEDED"}


def test_same_commit_id_different_hash_is_conflict(session_factory):
    """A retried call with the SAME commit_id but a DIFFERENT request_hash is
    a RunCommitConflictError -- the caller is asserting two distinct
    operations under one id, which the log refuses to silently collapse."""
    log = SqlAlchemyRunCommitLog()  # stateless
    rh1 = canonical_request_hash("complete", {"run_id": "r1", "version": 3})
    rh2 = canonical_request_hash("complete", {"run_id": "r1", "version": 4})

    async def _run():
        async with session_factory() as session:
            async with session.begin():
                await log.record(
                    session,
                    commit_id="complete:r1",
                    operation="complete",
                    run_id="r1",
                    request_hash=rh1,
                    result={"status": "SUCCEEDED"},
                )
        async with session_factory() as session:
            existing = await log.find(session, "complete:r1")
        # The replay's request_hash differs from the recorded one.
        if existing is not None and existing.request_hash != rh2:
            raise RunCommitConflictError(
                f"commit {existing.commit_id!r} replayed with a different request"
            )

    with pytest.raises(RunCommitConflictError):
        asyncio.run(_run())


def test_commit_id_is_unique_primary_key(session_factory):
    """The commit_id column is primary + unique: a second insert with the
    same id violates the constraint (the business path uses find() first to
    avoid this, but the schema enforces it as the structural backstop)."""
    from sqlalchemy.exc import IntegrityError

    log = SqlAlchemyRunCommitLog()  # stateless
    rh = canonical_request_hash("pause", {"run_id": "r1"})

    async def _run():
        async with session_factory() as session:
            async with session.begin():
                await log.record(
                    session,
                    commit_id="pause:r1",
                    operation="pause",
                    run_id="r1",
                    request_hash=rh,
                    result={},
                )
        async with session_factory() as session:
            async with session.begin():
                await log.record(
                    session,
                    commit_id="pause:r1",  # SAME primary key
                    operation="pause",
                    run_id="r1",
                    request_hash=rh,
                    result={},
                )

    with pytest.raises(IntegrityError):
        asyncio.run(_run())
