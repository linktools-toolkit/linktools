#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_checkpoint_store_contract.py"""
from datetime import datetime, timezone

import pytest

from linktools.ai.run.models import RunCheckpoint
from linktools.ai.storage.file.checkpoint import FileCheckpointStore


def _checkpoint(run_id="run-1", sequence=1) -> RunCheckpoint:
    return RunCheckpoint(
        id=f"{run_id}-{sequence}", run_id=run_id, sequence=sequence, format="msgpack",
        schema_version=1, payload=b"snapshot-bytes", created_at=datetime.now(timezone.utc),
    )


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileCheckpointStore(root=tmp_path / f"checkpoints-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.checkpoint import SqlAlchemyCheckpointStore

    counter = {"n": 0}
    engines = []

    def _run_in_new_loop(coro):
        # This factory is called synchronously from inside an already-running
        # pytest-asyncio event loop (the async test function), so we cannot use
        # asyncio.get_event_loop().run_until_complete() here -- that raises
        # "This event loop is already running". Run the setup coroutine to
        # completion on a separate thread with its own fresh event loop instead.
        import asyncio
        import threading

        outcome = {}

        def _runner():
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
                outcome["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def sqlalchemy_factory():
        counter["n"] += 1
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/checkpoints-db-{counter['n']}.db")
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            # The connection pool otherwise holds a connection bound to this
            # thread's event loop; dispose it so later operations (running on
            # pytest-asyncio's loop) open fresh connections instead of reusing
            # one tied to a loop that is about to be closed.
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyCheckpointStore(session_factory=session_factory)

    def _dispose_engines():
        # The store itself opens fresh connections on pytest-asyncio's loop
        # during the test. Those connections (and aiosqlite's background
        # worker threads) must be disposed before that loop closes at test
        # teardown, otherwise the worker thread tries to call back into an
        # already-closed loop and pytest reports an unraisable exception.
        for engine in engines:
            _run_in_new_loop(engine.dispose())

    request.addfinalizer(_dispose_engines)

    return sqlalchemy_factory


@pytest.mark.asyncio
async def test_save_then_get_roundtrip(store_factory):
    store = store_factory()
    await store.save(_checkpoint())
    fetched = await store.get("run-1-1")
    assert fetched is not None
    assert fetched.payload == b"snapshot-bytes"
    assert fetched.sequence == 1


@pytest.mark.asyncio
async def test_get_missing_returns_none(store_factory):
    store = store_factory()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_latest_returns_highest_sequence(store_factory):
    store = store_factory()
    await store.save(_checkpoint(sequence=1))
    await store.save(_checkpoint(sequence=3))
    await store.save(_checkpoint(sequence=2))
    latest = await store.latest("run-1")
    assert latest.sequence == 3


@pytest.mark.asyncio
async def test_latest_for_unknown_run_returns_none(store_factory):
    store = store_factory()
    assert await store.latest("nope") is None


@pytest.mark.asyncio
async def test_checkpoints_for_different_runs_are_isolated(store_factory):
    store = store_factory()
    await store.save(_checkpoint(run_id="run-a", sequence=1))
    await store.save(_checkpoint(run_id="run-b", sequence=1))
    latest_a = await store.latest("run-a")
    latest_b = await store.latest("run-b")
    assert latest_a.run_id == "run-a"
    assert latest_b.run_id == "run-b"


@pytest.mark.asyncio
async def test_created_at_roundtrips_as_timezone_aware(store_factory):
    store = store_factory()
    original = _checkpoint(sequence=1)
    await store.save(original)
    fetched = await store.get(original.id)
    assert fetched.created_at.tzinfo is not None
    assert fetched.created_at == original.created_at


@pytest.mark.asyncio
async def test_path_traversal_in_run_id_is_rejected(tmp_path):
    from linktools.ai.storage.file.checkpoint import FileCheckpointStore

    store = FileCheckpointStore(root=tmp_path)
    with pytest.raises(ValueError):
        await store.latest("../../etc/passwd")
