#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CheckpointStore contract: every backend (File, SQLAlchemy) must own
sequence assignment. Callers submit a NewRunCheckpoint (no id/sequence/
created_at); the Store returns the persisted RunCheckpoint with those fields
filled in. Append is serialized per-run so concurrent calls produce unique,
monotonic sequences."""

import asyncio

import pytest

from linktools.ai.run.models import NewRunCheckpoint
from linktools.ai.storage.file.checkpoint import FileCheckpointStore


def _new(run_id="run-1") -> NewRunCheckpoint:
    return NewRunCheckpoint(
        run_id=run_id,
        format="msgpack",
        schema_version=1,
        payload=b"snapshot-bytes",
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
        # asyncio.get_event_loop().run_until_complete() -- that raises
        # "This event loop is already running". Run the setup coroutine to
        # completion on a separate thread with its own fresh event loop instead.
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
            f"sqlite+aiosqlite:///{tmp_path}/checkpoints-db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyCheckpointStore(session_factory=session_factory)

    def _dispose_engines():
        for engine in engines:
            _run_in_new_loop(engine.dispose())

    request.addfinalizer(_dispose_engines)

    return sqlalchemy_factory


@pytest.mark.asyncio
async def test_append_assigns_monotonic_sequence(store_factory):
    store = store_factory()
    first = await store.append(_new())
    second = await store.append(_new())
    third = await store.append(_new())
    assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
    # id and created_at are Store-assigned, not caller-supplied.
    assert first.id and second.id and third.id
    assert first.created_at is not None


@pytest.mark.asyncio
async def test_latest_returns_highest_sequence(store_factory):
    store = store_factory()
    await store.append(_new())
    await store.append(_new())
    await store.append(_new())
    latest = await store.latest("run-1")
    assert latest is not None
    assert latest.sequence == 3


@pytest.mark.asyncio
async def test_append_then_get_roundtrip(store_factory):
    store = store_factory()
    persisted = await store.append(_new())
    fetched = await store.get(persisted.id)
    assert fetched is not None
    assert fetched.payload == b"snapshot-bytes"
    assert fetched.sequence == 1
    assert fetched.run_id == "run-1"


@pytest.mark.asyncio
async def test_get_missing_returns_none(store_factory):
    store = store_factory()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_latest_for_unknown_run_returns_none(store_factory):
    store = store_factory()
    assert await store.latest("nope") is None


@pytest.mark.asyncio
async def test_checkpoints_for_different_runs_are_isolated(store_factory):
    store = store_factory()
    a = await store.append(_new(run_id="run-a"))
    b = await store.append(_new(run_id="run-b"))
    assert a.sequence == 1 and b.sequence == 1
    assert (await store.latest("run-a")).run_id == "run-a"
    assert (await store.latest("run-b")).run_id == "run-b"


@pytest.mark.asyncio
async def test_concurrent_append_produces_unique_sequences(store_factory):
    """20 concurrent appends for the same run must yield sequences 1..20 with
    no duplicates (the per-run lock + counter / unique constraint prevent
    collisions)."""
    store = store_factory()
    results = await asyncio.gather(*(store.append(_new()) for _ in range(20)))
    sequences = sorted(r.sequence for r in results)
    assert sequences == list(range(1, 21)), (
        f"concurrent append sequences must be 1..20, got {sequences}"
    )
    assert len({r.id for r in results}) == 20, "each append gets a unique id"


@pytest.mark.asyncio
async def test_created_at_roundtrips_as_timezone_aware(store_factory):
    store = store_factory()
    persisted = await store.append(_new())
    fetched = await store.get(persisted.id)
    assert fetched.created_at.tzinfo is not None


@pytest.mark.asyncio
async def test_multiple_pause_resume_does_not_overwrite(store_factory):
    """A run paused/resumed several times accumulates checkpoints rather than
    overwriting sequence=1 -- the original File/SQL bug."""
    store = store_factory()
    for _ in range(3):
        await store.append(_new())
    latest = await store.latest("run-1")
    assert latest.sequence == 3


@pytest.mark.asyncio
async def test_path_traversal_in_run_id_is_rejected(tmp_path):
    store = FileCheckpointStore(root=tmp_path)
    with pytest.raises(ValueError):
        await store.latest("../../etc/passwd")
