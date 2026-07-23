#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_event_store_contract.py

EventStore contract: append() takes the payload + run/stream context kwargs
and assigns the sequence itself (design note contract) -- callers never construct
an EventEnvelope with a sequence. These tests verify the store-side sequence
assignment, isolation, and round-tripping for both the file and sqlalchemy
backends."""

import asyncio

import pytest

from linktools.ai.events.payloads import RunStarted
from linktools.ai.storage.filesystem.event import FilesystemEventStore


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
            return FilesystemEventStore(root=tmp_path / f"events-{counter['n']}")

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
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/events-db-{counter['n']}.db"
        )
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
    # assignment must hand each one a unique sequence (design note contract/contract).
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
        stream_id="run-1",
        run_id="run-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        session_id="sess-1",
        runnable_id="agent-x",
        payload=RunStarted(run_id="run-1", runnable_id="agent-x"),
    )
    envelope = (await store.list("run-1")).items[0]
    assert envelope.root_run_id == "root-1"
    assert envelope.parent_run_id == "parent-1"
    assert envelope.session_id == "sess-1"
    assert envelope.runnable_id == "agent-x"


@pytest.mark.asyncio
async def test_event_envelope_has_stream_id(store_factory):
    """stream_id is a first-class EventEnvelope field."""
    store = store_factory()
    envelope = await _append(store)
    assert envelope.stream_id == "run-1"


@pytest.mark.asyncio
async def test_different_streams_can_share_sequence_number(store_factory):
    """the uniqueness boundary is (stream_id, sequence), not
    (run_id, sequence) -- two DIFFERENT streams that happen to share a run_id
    prefix (or, once a future caller mints stream_id != run_id, genuinely
    different streams for the SAME run) must each independently start
    sequence at 1 without colliding."""
    store = store_factory()
    a = await store.append(
        stream_id="stream-a",
        run_id="run-shared",
        root_run_id="run-shared",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        payload=RunStarted(run_id="run-shared", runnable_id="agent-1"),
    )
    b = await store.append(
        stream_id="stream-b",
        run_id="run-shared",
        root_run_id="run-shared",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        payload=RunStarted(run_id="run-shared", runnable_id="agent-1"),
    )
    assert a.sequence == 1
    assert b.sequence == 1
    assert a.stream_id == "stream-a"
    assert b.stream_id == "stream-b"
    page_a = await store.list("stream-a")
    page_b = await store.list("stream-b")
    assert len(page_a.items) == 1
    assert len(page_b.items) == 1


@pytest.mark.asyncio
async def test_event_sequence_unique_per_stream_not_per_run(store_factory):
    """appending twice to the SAME stream_id (even under different
    run_id values, an edge case only possible once a caller decouples the
    two) still assigns strictly increasing sequences within that stream."""
    store = store_factory()
    first = await store.append(
        stream_id="shared-stream",
        run_id="run-x",
        root_run_id="run-x",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        payload=RunStarted(run_id="run-x", runnable_id="agent-1"),
    )
    second = await store.append(
        stream_id="shared-stream",
        run_id="run-y",
        root_run_id="run-y",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        payload=RunStarted(run_id="run-y", runnable_id="agent-1"),
    )
    assert first.sequence == 1
    assert second.sequence == 2
    page = await store.list("shared-stream")
    assert [e.sequence for e in page.items] == [1, 2]


@pytest.mark.asyncio
async def test_path_traversal_in_run_id_is_rejected(tmp_path):
    store = FilesystemEventStore(root=tmp_path)
    with pytest.raises(ValueError):
        await store.list("../../etc/passwd")


@pytest.mark.asyncio
async def test_file_event_store_migrates_legacy_files_without_stream_id(tmp_path):
    """a FilesystemEventStore event file written before stream_id
    became a first-class field (no "stream_id" key in the JSON) must still
    load, with stream_id defaulting to run_id -- exact, not a guess, since
    every caller has always passed stream_id == run_id."""
    import json

    store = FilesystemEventStore(root=tmp_path)
    await _append(store, run_id="run-legacy")
    # Simulate a legacy file by rewriting it without the "stream_id" key.
    stream_dir = tmp_path / "run-legacy"
    event_path = next(stream_dir.glob("*.json"))
    raw = json.loads(event_path.read_text())
    del raw["stream_id"]
    event_path.write_text(json.dumps(raw))

    page = await store.list("run-legacy")
    assert len(page.items) == 1
    assert page.items[0].stream_id == "run-legacy"
