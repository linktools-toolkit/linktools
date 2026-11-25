#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable acceptance, publication and fencing for regenerable turns."""

import asyncio

import pytest
import pytest_asyncio
from linktools.ai.persistence._session_turn_sql import (
    SESSION_TURN_TABLE,
    SessionTurnKey,
    SqlSessionTurns,
    build_session_turn_sql_metadata,
)
from linktools.ai.storage import create_sql_storage_context
from sqlalchemy import MetaData, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine


@pytest_asyncio.fixture
async def store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'turns.db'}")
    metadata = MetaData()
    build_session_turn_sql_metadata(metadata)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        turns = SqlSessionTurns(create_sql_storage_context(engine), metadata, namespace="test", tenant_id="a")
        await turns.create("chat", owner_principal_id="actor", agent_id="qa", config={})
        yield engine, metadata, turns
    finally:
        await engine.dispose()


async def accept(turns: SqlSessionTurns, turn: str, *, revision: int = 0, through: int = 0, request: str = "send") -> bool:
    return await turns.accept(
        SessionTurnKey("chat", turn), expected_revision=revision, context_through=through,
        request_key=request, request={"input": turn}, lease_seconds=60,
    )


@pytest.mark.asyncio
async def test_regenerable_turn_has_two_write_boundaries_and_one_visible_record(store):
    engine, _, turns = store
    statements = []
    commits = []

    def on_sql(_conn, _cursor, statement, *_args):
        verb = statement.split()[0]
        if verb in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            statements.append(verb)

    def on_commit(_conn):
        commits.append(len(statements))

    event.listen(engine.sync_engine, "before_cursor_execute", on_sql)
    event.listen(engine.sync_engine, "commit", on_commit)
    assert await turns.get_session("chat") is not None
    assert await turns.history("chat", through=0, limit=1) == ()
    key = SessionTurnKey("chat", "turn")
    assert await accept(turns, key.turn_id)
    assert await turns.finish(key, generation=1, turn_no=1, state="succeeded", result={"answer": "two"})
    assert statements == ["SELECT", "SELECT", "UPDATE", "INSERT", "UPDATE", "UPDATE"]
    assert commits == [4, 6]

    session = await turns.get_session("chat")
    assert session["active_turn_id"] is None and session["last_turn_no"] == 1
    turn, = await turns.history("chat", through=1, limit=2)
    assert turn["request_json"] == {"input": "turn"}
    assert turn["result_json"] == {"answer": "two"}
    assert (await turns.find_request("chat", "send"))["turn_id"] == b"turn"


@pytest.mark.asyncio
async def test_concurrent_admissions_and_duplicate_request_cannot_leave_a_second_occupant(store):
    _, _, turns = store
    go = asyncio.Event()

    async def contender(turn: str) -> bool:
        await go.wait()
        return await accept(turns, turn, request=turn)

    a = asyncio.create_task(contender("a"))
    b = asyncio.create_task(contender("b"))
    go.set()
    outcomes = await asyncio.gather(a, b)
    assert sum(outcomes) == 1
    winner = ("a", "b")[outcomes.index(True)]
    key = SessionTurnKey("chat", winner)
    assert await turns.finish(key, generation=1, turn_no=1, state="succeeded", result={})

    with pytest.raises(IntegrityError):
        await accept(turns, "duplicate", revision=2, through=1, request=winner)
    session = await turns.get_session("chat")
    assert session["revision"] == 2 and session["active_turn_id"] is None
    assert await turns.get_turn(SessionTurnKey("chat", "duplicate")) is None
    assert await accept(turns, "next", revision=2, through=1, request="new")


@pytest.mark.asyncio
async def test_takeover_fences_old_writer_and_cancel_precedes_success(store):
    engine, metadata, turns = store
    key = SessionTurnKey("chat", "turn")
    assert await accept(turns, key.turn_id)
    assert not await turns.take_over(key, generation=1, lease_seconds=60)
    table = metadata.tables[SESSION_TURN_TABLE]
    async with engine.begin() as connection:
        await connection.execute(table.update().values(lease_until=0))

    outcomes = await asyncio.gather(*(turns.take_over(key, generation=1, lease_seconds=60) for _ in range(2)))
    assert sum(outcomes) == 1
    assert not await turns.finish(key, generation=1, turn_no=1, state="succeeded", result={"stale": True})
    assert (await turns.get_session("chat"))["active_turn_id"] == b"turn"
    assert await turns.request_cancel(key)
    assert not await turns.finish(key, generation=2, turn_no=1, state="succeeded", result={"late": True})
    assert (await turns.get_session("chat"))["active_turn_id"] == b"turn"
    assert await turns.finish(key, generation=2, turn_no=1, state="cancelled", result=None)
    assert (await turns.get_turn(key))["state"] == "cancelled"
    assert await turns.history("chat", through=1, limit=2, only_success=True) == ()


@pytest.mark.asyncio
async def test_reopened_store_recovers_expired_cancellation_once(store):
    engine, metadata, turns = store
    key = SessionTurnKey("chat", "turn")
    assert await accept(turns, key.turn_id)
    assert await turns.request_cancel(key)
    async with engine.begin() as connection:
        await connection.execute(metadata.tables[SESSION_TURN_TABLE].update().values(lease_until=0))
    await engine.dispose()

    restarted_engine = create_async_engine(engine.url)
    try:
        restarted = SqlSessionTurns(
            create_sql_storage_context(restarted_engine), metadata, namespace="test", tenant_id="a",
        )
        pending = await restarted.get_turn(key)
        assert (pending["state"], pending["generation"]) == ("cancelling", 1)
        assert not await restarted.take_over(key, generation=1, lease_seconds=60)
        assert not await restarted.renew(key, generation=1, lease_seconds=60)

        outcomes = await asyncio.gather(*(
            restarted.take_over(key, generation=1, lease_seconds=60, cancelling=True) for _ in range(2)
        ))
        assert outcomes.count(True) == 1
        assert not await restarted.finish(key, generation=1, turn_no=1, state="cancelled", result=None)
        assert (await restarted.get_session("chat"))["active_turn_id"] == b"turn"
        assert await restarted.finish(key, generation=2, turn_no=1, state="cancelled", result=None)
        assert not await restarted.finish(key, generation=2, turn_no=1, state="cancelled", result=None)

        visible, = await restarted.history("chat", through=1, limit=2)
        assert (visible["turn_id"], visible["state"], visible["result_json"]) == (b"turn", "cancelled", None)
        assert await restarted.history("chat", through=1, limit=2, only_success=True) == ()
        assert await accept(restarted, "next", revision=2, through=1, request="new")
    finally:
        await restarted_engine.dispose()


@pytest.mark.asyncio
async def test_partition_and_session_must_match_exact_turn(store):
    engine, metadata, turns = store
    assert await accept(turns, "turn")
    other_tenant = SqlSessionTurns(create_sql_storage_context(engine), metadata, namespace="test", tenant_id="b")
    assert await other_tenant.get_turn(SessionTurnKey("chat", "turn")) is None
    assert await turns.get_turn(SessionTurnKey("other-chat", "turn")) is None
    assert not await other_tenant.request_cancel(SessionTurnKey("chat", "turn"))
    assert (await turns.get_turn(SessionTurnKey("chat", "turn")))["state"] == "accepted"
