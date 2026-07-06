#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_event_store_contract.py

EventStore contract: append() takes the payload + run/stream context kwargs
and assigns the sequence itself (review doc §8.1) -- callers never construct
an EventEnvelope with a sequence. These tests verify the store-side sequence
assignment, isolation, and round-tripping for both the file and sqlalchemy
backends."""
import asyncio

import pytest

from linktools.ai.events.payloads import RunStarted
from linktools.ai.storage.file.event import FileEventStore


async def _append(
    store,
    *,
    run_id="run-1",
    root_run_id=None,
    parent_run_id=None,
    session_id="session-1",
    runnable_id="agent-1",
):
    """Append one RunStarted event with sensible default routing fields."""
    return await store.append(
        stream_id=run_id,
        run_id=run_id,
        root_run_id=root_run_id or run_id,
        parent_run_id=parent_run_id,
        session_id=session_id,
        runnable_id=runnable_id,
        payload=RunStarted(run_id=run_id, runnable_id=runnable_id),
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
async def test_append_assigns_sequence_starting_at_one(store_factory):
    store = store_factory()
    envelope = await _append(store)
    assert envelope.sequence == 1
    # The store mints the identity/timing fields the caller used to supply.
    assert envelope.event_id  # non-empty uuid string
    assert envelope.occurred_at.tzinfo is not None
    assert isinstance(envelope.payload, RunStarted)


@pytest.mark.asyncio
async def test_append_assigns_increasing_sequences(store_factory):
    store = store_factory()
    seqs = [(await _append(store)).sequence for _ in range(4)]
    assert seqs == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_append_then_list_roundtrip(store_factory):
    store = store_factory()
    await _append(store)
    await _append(store)
    page = await store.list("run-1")
    assert [e.sequence for e in page.items] == [1, 2]
    assert isinstance(page.items[0].payload, RunStarted)


@pytest.mark.asyncio
async def test_list_after_sequence_filters(store_factory):
    store = store_factory()
    for _ in range(3):
        await _append(store)
    page = await store.list("run-1", after_sequence=1)
    assert [e.sequence for e in page.items] == [2, 3]


@pytest.mark.asyncio
async def test_list_respects_limit(store_factory):
    store = store_factory()
    for _ in range(5):
        await _append(store)
    page = await store.list("run-1", limit=2)
    assert [e.sequence for e in page.items] == [1, 2]


@pytest.mark.asyncio
async def test_concurrent_appends_get_distinct_sequences(store_factory):
    # Fire several appends concurrently; the store's per-stream sequence
    # assignment must hand each one a unique sequence (review doc §8.1/§8.4).
    store = store_factory()
    envelopes = await asyncio.gather(*(_append(store) for _ in range(6)))
    seqs = sorted(e.sequence for e in envelopes)
    assert seqs == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_events_for_different_runs_are_isolated(store_factory):
    store = store_factory()
    await _append(store, run_id="run-a")
    await _append(store, run_id="run-b")
    page_a = await store.list("run-a")
    page_b = await store.list("run-b")
    assert len(page_a.items) == 1
    assert len(page_b.items) == 1
    assert page_a.items[0].run_id == "run-a"
    # Each stream's sequence counter is independent -- both start at 1.
    assert page_a.items[0].sequence == 1
    assert page_b.items[0].sequence == 1


@pytest.mark.asyncio
async def test_occurred_at_roundtrips_as_timezone_aware(store_factory):
    store = store_factory()
    appended = await _append(store)
    page = await store.list("run-1")
    fetched = page.items[0]
    assert fetched.occurred_at.tzinfo is not None
    assert fetched.occurred_at == appended.occurred_at


@pytest.mark.asyncio
async def test_append_routing_fields_roundtrip(store_factory):
    store = store_factory()
    await store.append(
        stream_id="run-1", run_id="run-1", root_run_id="root-1",
        parent_run_id="parent-1", session_id="sess-1", runnable_id="agent-x",
        payload=RunStarted(run_id="run-1", runnable_id="agent-x"),
    )
    envelope = (await store.list("run-1")).items[0]
    assert envelope.root_run_id == "root-1"
    assert envelope.parent_run_id == "parent-1"
    assert envelope.session_id == "sess-1"
    assert envelope.runnable_id == "agent-x"


@pytest.mark.asyncio
async def test_path_traversal_in_run_id_is_rejected(tmp_path):
    store = FileEventStore(root=tmp_path)
    with pytest.raises(ValueError):
        await store.list("../../etc/passwd")
