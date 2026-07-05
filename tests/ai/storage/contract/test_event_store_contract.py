#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_event_store_contract.py"""
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import EventSequenceConflictError
from linktools.ai.events.envelope import EventEnvelope
from linktools.ai.events.payloads import RunStarted
from linktools.ai.storage.file.event import FileEventStore


def _event(run_id="run-1", sequence=1) -> EventEnvelope:
    return EventEnvelope(
        event_id=f"evt-{run_id}-{sequence}", sequence=sequence, occurred_at=datetime.now(timezone.utc),
        run_id=run_id, root_run_id=run_id, parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", payload=RunStarted(run_id=run_id, runnable_id="agent-1"),
    )


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileEventStore(root=tmp_path / f"events-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.event import SqlAlchemyEventStore

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
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/events-db-{counter['n']}.db")
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
        return SqlAlchemyEventStore(session_factory=session_factory)

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
async def test_append_then_list_roundtrip(store_factory):
    store = store_factory()
    await store.append(_event(sequence=1))
    await store.append(_event(sequence=2))
    page = await store.list("run-1")
    assert [e.sequence for e in page.items] == [1, 2]
    assert isinstance(page.items[0].payload, RunStarted)


@pytest.mark.asyncio
async def test_list_after_sequence_filters(store_factory):
    store = store_factory()
    await store.append(_event(sequence=1))
    await store.append(_event(sequence=2))
    await store.append(_event(sequence=3))
    page = await store.list("run-1", after_sequence=1)
    assert [e.sequence for e in page.items] == [2, 3]


@pytest.mark.asyncio
async def test_list_respects_limit(store_factory):
    store = store_factory()
    for seq in range(1, 6):
        await store.append(_event(sequence=seq))
    page = await store.list("run-1", limit=2)
    assert [e.sequence for e in page.items] == [1, 2]


@pytest.mark.asyncio
async def test_append_with_expected_sequence_conflict_raises(store_factory):
    store = store_factory()
    await store.append(_event(sequence=1))
    with pytest.raises(EventSequenceConflictError):
        await store.append(_event(sequence=1), expected_sequence=5)


@pytest.mark.asyncio
async def test_append_without_expected_sequence_still_rejects_duplicate_sequence(store_factory):
    store = store_factory()
    await store.append(_event(sequence=1))
    with pytest.raises(EventSequenceConflictError):
        await store.append(_event(sequence=1))


@pytest.mark.asyncio
async def test_events_for_different_runs_are_isolated(store_factory):
    store = store_factory()
    await store.append(_event(run_id="run-a", sequence=1))
    await store.append(_event(run_id="run-b", sequence=1))
    page_a = await store.list("run-a")
    page_b = await store.list("run-b")
    assert len(page_a.items) == 1
    assert len(page_b.items) == 1
    assert page_a.items[0].run_id == "run-a"


@pytest.mark.asyncio
async def test_occurred_at_roundtrips_as_timezone_aware(store_factory):
    store = store_factory()
    original = _event(sequence=1)
    await store.append(original)
    page = await store.list("run-1")
    fetched = page.items[0]
    assert fetched.occurred_at.tzinfo is not None
    assert fetched.occurred_at == original.occurred_at
