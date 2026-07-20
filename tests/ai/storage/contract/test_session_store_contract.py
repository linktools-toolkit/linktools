#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_session_store_contract.py"""

from datetime import datetime, timezone

import pytest

from linktools.ai.session.models import (
    MessageRole,
    NewSessionMessage,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage.filesystem.session import FilesystemSessionStore


def _record(session_id="session-1") -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        id=session_id,
        parent_id=None,
        status=SessionStatus.ACTIVE,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _message(role=MessageRole.USER, content="hi") -> NewSessionMessage:
    # G6/review3 contract: the input shape carries no id/sequence/created_at --
    # the SessionStore is the sole authority for assigning those.
    return NewSessionMessage(role=role, content=content, run_id=None)


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FilesystemSessionStore(root=tmp_path / f"sessions-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.session import SqlAlchemySessionStore

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
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/sessions-db-{counter['n']}.db"
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
        return SqlAlchemySessionStore(session_factory=session_factory)

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
async def test_create_then_get_roundtrip(store_factory):
    store = store_factory()
    created = await store.create(_record())
    fetched = await store.get("session-1")
    assert fetched is not None
    assert fetched.id == "session-1"
    assert created == fetched


@pytest.mark.asyncio
async def test_get_missing_returns_none(store_factory):
    store = store_factory()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_append_then_list_messages_in_order(store_factory):
    store = store_factory()
    await store.create(_record())
    persisted = await store.append_messages(
        "session-1",
        (_message(content="hi"), _message(role=MessageRole.ASSISTANT, content="hello")),
    )
    # append_messages returns the persisted messages with store-assigned
    # id/sequence/created_at (G6).
    assert [m.sequence for m in persisted] == [1, 2]
    messages = await store.list_messages("session-1")
    assert [m.content for m in messages] == ["hi", "hello"]
    assert messages[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_list_messages_after_sequence_filters(store_factory):
    store = store_factory()
    await store.create(_record())
    await store.append_messages("session-1", (_message(), _message(), _message()))
    messages = await store.list_messages("session-1", after_sequence=1)
    assert [m.sequence for m in messages] == [2, 3]


@pytest.mark.asyncio
async def test_append_messages_assigns_sequence_the_store_never_reuses(store_factory):
    """G6: the store -- not the caller -- assigns sequence numbers, and a
    second append call continues from the session's current max rather than
    restarting at 1."""
    store = store_factory()
    await store.create(_record())
    first_batch = await store.append_messages(
        "session-1", (_message(content="a"), _message(content="b"))
    )
    assert [m.sequence for m in first_batch] == [1, 2]
    second_batch = await store.append_messages("session-1", (_message(content="c"),))
    assert [m.sequence for m in second_batch] == [3]


@pytest.mark.asyncio
async def test_concurrent_append_messages_never_assigns_duplicate_sequence(
    store_factory,
):
    """G6: two coroutines racing to append to the SAME session must never be
    assigned the same sequence number -- the store is the sole sequence
    authority, so unlike the old caller-computed-sequence design, there is no
    read-then-compute-then-write gap for concurrent appenders to race in."""
    import asyncio

    store = store_factory()
    await store.create(_record())

    async def _append(content: str):
        return await store.append_messages("session-1", (_message(content=content),))

    results = await asyncio.gather(*(_append(f"msg-{i}") for i in range(10)))
    all_sequences = [batch[0].sequence for batch in results]
    assert sorted(all_sequences) == list(range(1, 11)), (
        f"expected sequences 1..10 with no duplicates, got {sorted(all_sequences)}"
    )
    messages = await store.list_messages("session-1")
    assert len(messages) == 10
    assert [m.sequence for m in messages] == list(range(1, 11))


@pytest.mark.asyncio
async def test_update_status_and_metadata(store_factory):
    store = store_factory()
    await store.create(_record())
    updated = await store.update(
        "session-1", status=SessionStatus.ARCHIVED, metadata={"k": "v"}
    )
    assert updated.status == SessionStatus.ARCHIVED
    assert dict(updated.metadata) == {"k": "v"}
    assert updated.version == 2


@pytest.mark.asyncio
async def test_sessions_are_isolated(store_factory):
    store = store_factory()
    await store.create(_record(session_id="session-a"))
    await store.create(_record(session_id="session-b"))
    await store.append_messages("session-a", (_message(),))
    messages_a = await store.list_messages("session-a")
    messages_b = await store.list_messages("session-b")
    assert len(messages_a) == 1
    assert len(messages_b) == 0


@pytest.mark.asyncio
async def test_path_traversal_in_session_id_is_rejected(tmp_path):
    store = FilesystemSessionStore(root=tmp_path)
    with pytest.raises(ValueError):
        await store.get("../../etc/passwd")
