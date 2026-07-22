#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset is part of the SQLAlchemy Unit of Work.
The session-bound SqlAlchemyAssetBackend reuses the UoW's AsyncSession, so an
asset mutation commits or rolls back with every other store in the unit. These
tests prove the shared-transaction semantics: atomic commit, atomic rollback,
CAS-conflict rollback, revision/idempotency rollback, UoW write-then-read
visibility, pre-commit invisibility, and single-revision move."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.models import WriteOptions
from linktools.ai.storage import SqlAlchemyStorage
from linktools.ai.storage.sqlalchemy.models import Base


def _storage(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/asset.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/asset.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )


def _run(coro):
    return asyncio.run(coro)


def test_asset_write_commits_with_the_uow(tmp_path):
    storage = _storage(tmp_path)
    path = AssetPath("/agents/a.md")

    async def _scenario():
        async with storage.transaction() as tx:
            await tx.assets.put(path, b"hello", options=WriteOptions())
            # Visible within the UoW before commit.
            assert (await tx.assets.get(path)) is not None

    _run(_scenario())
    # After the UoW committed, a fresh read observes the asset.
    assert _run(storage.assets.get(path)) is not None


def test_asset_write_rolls_back_when_the_uow_aborts(tmp_path):
    storage = _storage(tmp_path)
    path = AssetPath("/agents/rollback.md")

    async def _scenario():
        with pytest.raises(RuntimeError):
            async with storage.transaction() as tx:
                await tx.assets.put(path, b"transient", options=WriteOptions())
                # A later step in the same UoW fails -> everything rolls back.
                raise RuntimeError("downstream step failed")

    _run(_scenario())
    # The asset write was rolled back with the UoW.
    assert _run(storage.assets.get(path)) is None


def test_asset_cas_conflict_rolls_back_the_uow(tmp_path):
    storage = _storage(tmp_path)
    path = AssetPath("/agents/cas.md")

    async def _seed():
        await storage.assets.put(path, b"v1", options=WriteOptions())

    _run(_seed())

    async def _scenario():
        with pytest.raises(Exception):
            async with storage.transaction() as tx:
                # if_match on a stale etag -> precondition failure aborts the UoW.
                await tx.assets.put(
                    path,
                    b"v2",
                    options=WriteOptions(if_match="deadbeef"),
                )

    _run(_scenario())
    # Original content survives the aborted CAS attempt.
    asset = _run(storage.assets.get(path))
    assert asset is not None and asset.content == b"v1"


def test_asset_revision_rolls_back_with_the_uow(tmp_path):
    storage = _storage(tmp_path)
    path = AssetPath("/agents/rev.md")

    async def _seed():
        await storage.assets.put(path, b"first", options=WriteOptions())

    _run(_seed())
    before = _run(storage.assets._primary.revision())

    async def _scenario():
        with pytest.raises(RuntimeError):
            async with storage.transaction() as tx:
                await tx.assets.put(path, b"second", options=WriteOptions())
                raise RuntimeError("fail after asset write")

    _run(_scenario())
    after = _run(storage.assets._primary.revision())
    # The revision bump from the rolled-back write did not persist.
    assert after == before


def test_asset_idempotency_rolls_back_with_the_uow(tmp_path):
    storage = _storage(tmp_path)
    path = AssetPath("/agents/idem.md")

    async def _scenario():
        with pytest.raises(RuntimeError):
            async with storage.transaction() as tx:
                await tx.assets.put(
                    path,
                    b"x",
                    options=WriteOptions(idempotency_key="k1"),
                )
                raise RuntimeError("fail after idempotent put")

    _run(_scenario())
    # The idempotency record was rolled back with the UoW: a later replay with
    # the same key is NOT a cached hit (the record is gone).
    assert _run(storage.assets._primary.get_idempotency("put:k1")) is None


def test_uow_write_is_invisible_outside_before_commit(tmp_path):
    storage = _storage(tmp_path)
    path = AssetPath("/agents/hidden.md")

    async def _txn():
        async with storage.transaction() as tx:
            await tx.assets.put(path, b"inside", options=WriteOptions())
            # While the UoW is still open, the standalone (non-UoW) read path on
            # a separate session must NOT see the uncommitted write.
            return await storage.assets.get(path)

    in_txn = _run(_txn())
    assert in_txn is None, "uncommitted UoW write must be invisible outside it"
    # After commit, it is visible.
    assert _run(storage.assets.get(path)) is not None


def test_move_in_one_session_bumps_revision_once(tmp_path):
    storage = _storage(tmp_path)
    src = AssetPath("/agents/src.md")
    dst = AssetPath("/agents/dst.md")

    async def _seed():
        await storage.assets.put(src, b"payload", options=WriteOptions())

    _run(_seed())
    before = _run(storage.assets._primary.revision())

    async def _move():
        async with storage.transaction() as tx:
            await tx.assets.move(src, dst, options=WriteOptions())

    _run(_move())
    after = _run(storage.assets._primary.revision())
    # A single move is one state change -> exactly one revision bump.
    assert after - before == 1


def test_move_with_idempotency_key_replays_without_double_execution(tmp_path):
    # raw_move_checked folds idempotency check + move + save into one atomic
    # section: a replay with the same key returns the cached result and does
    # NOT execute the move twice (revision bumps once for the pair).
    storage = _storage(tmp_path)
    src = AssetPath("/agents/src.md")
    dst = AssetPath("/agents/dst.md")

    async def _seed():
        await storage.assets.put(src, b"payload", options=WriteOptions())

    _run(_seed())
    before = _run(storage.assets._primary.revision())

    async def _move_pair():
        first = await storage.assets.move(
            src, dst, options=WriteOptions(idempotency_key="mv1")
        )
        # Replay: source already moved, key cached -> returns cached result.
        second = await storage.assets.move(
            src, dst, options=WriteOptions(idempotency_key="mv1")
        )
        return first, second

    first, second = _run(_move_pair())
    after = _run(storage.assets._primary.revision())
    assert first.info.path == dst
    assert second.info.path == dst
    # One move executed (one revision bump) despite two calls.
    assert after - before == 1


def test_asset_run_session_commit_together(tmp_path):
    # : an asset write + a run write + a session write in one UoW commit
    # together -- all three are visible after the transaction.
    from datetime import datetime, timezone

    from linktools.ai.run.models import (
        RunInput,
        RunnableType,
        RunRecord,
        RunStatus,
    )
    from linktools.ai.session.models import SessionRecord, SessionStatus

    storage = _storage(tmp_path)
    path = AssetPath("/agents/x.md")
    now = datetime.now(timezone.utc)

    async def _scenario():
        async with storage.transaction() as tx:
            await tx.sessions.create(
                SessionRecord(
                    id="s1", parent_id=None, status=SessionStatus.ACTIVE,
                    version=1, created_at=now, updated_at=now,
                )
            )
            await tx.runs.create(
                RunRecord(
                    id="r1", root_run_id="r1", parent_run_id=None, session_id="s1",
                    runnable_id="a", runnable_type=RunnableType.AGENT,
                    status=RunStatus.RUNNING, input=RunInput(prompt="hi"),
                    result=None, error=None, version=1, created_at=now,
                    started_at=now, finished_at=None,
                )
            )
            await tx.assets.put(path, b"asset+run+session", options=WriteOptions())

    _run(_scenario())
    assert _run(storage.sessions.get("s1")) is not None
    assert _run(storage.runs.get("r1")) is not None
    assert _run(storage.assets.get(path)) is not None


def test_asset_run_session_rollback_together(tmp_path):
    # : when a later step in the UoW raises, the asset + run + session
    # writes all roll back together.
    from datetime import datetime, timezone

    from linktools.ai.run.models import (
        RunInput,
        RunnableType,
        RunRecord,
        RunStatus,
    )
    from linktools.ai.session.models import SessionRecord, SessionStatus

    storage = _storage(tmp_path)
    path = AssetPath("/agents/rollback.md")
    now = datetime.now(timezone.utc)

    async def _scenario():
        with pytest.raises(RuntimeError):
            async with storage.transaction() as tx:
                await tx.sessions.create(
                    SessionRecord(
                        id="s2", parent_id=None, status=SessionStatus.ACTIVE,
                        version=1, created_at=now, updated_at=now,
                    )
                )
                await tx.runs.create(
                    RunRecord(
                        id="r2", root_run_id="r2", parent_run_id=None,
                        session_id="s2", runnable_id="a",
                        runnable_type=RunnableType.AGENT, status=RunStatus.RUNNING,
                        input=RunInput(prompt="hi"), result=None, error=None,
                        version=1, created_at=now, started_at=now, finished_at=None,
                    )
                )
                await tx.assets.put(path, b"transient", options=WriteOptions())
                raise RuntimeError("downstream step failed")

    _run(_scenario())
    # All three writes rolled back with the UoW.
    assert _run(storage.sessions.get("s2")) is None
    assert _run(storage.runs.get("r2")) is None
    assert _run(storage.assets.get(path)) is None
