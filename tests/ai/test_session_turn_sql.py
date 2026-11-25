#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable acceptance, publication and fencing for regenerable turns."""

import asyncio

import pytest
import pytest_asyncio
from linktools.ai.persistence._session_turn_sql import (
    SESSION_BINDING_TABLE,
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
        await turns.create("chat", config={})
        yield engine, metadata, turns
    finally:
        await engine.dispose()


_BINDING = {"agent_spec": {"id": "test-agent", "instructions": ["frozen"]}}


async def accept(turns: SqlSessionTurns, turn: str, *, revision: int = 0, through: int = 0, request: str = "send") -> bool:
    return await turns.accept(
        SessionTurnKey("chat", turn), expected_revision=revision, context_from=1, context_through=through,
        request_key=request, request={"input": turn}, binding=_BINDING, lease_seconds=60,
    )


@pytest.mark.asyncio
async def test_regenerable_turn_has_two_write_boundaries_and_one_visible_record(store):
    engine, metadata, turns = store
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
    assert statements == ["SELECT", "SELECT", "UPDATE", "INSERT", "INSERT", "UPDATE", "UPDATE"]
    assert commits == [5, 7]
    assert await accept(turns, "second", revision=2, through=1, request="second")
    assert statements[7:] == ["UPDATE", "INSERT"]
    assert commits == [5, 7, 9]

    session = await turns.get_session("chat")
    assert session["active_turn_id"] == b"second" and session["last_turn_no"] == 1
    turn, = await turns.history("chat", through=1, limit=2)
    assert turn["request_json"] == {"input": "turn"}
    assert turn["result_json"] == {"answer": "two"}
    replay = await turns.find_request("chat", "send")
    assert replay["turn_id"] == b"turn" and replay["agent_id"] == "test-agent"
    second = await turns.get_turn_for_recovery(SessionTurnKey("chat", "second"))
    assert second["binding_key"] == turn["binding_key"] and second["binding_json"] == _BINDING
    async with engine.connect() as connection:
        table = metadata.tables[SESSION_BINDING_TABLE]
        assert len((await connection.execute(table.select())).all()) == 1


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


@pytest.mark.asyncio
async def test_admission_pins_selected_history_range_even_when_empty(store):
    _, _, turns = store
    assert await accept(turns, "first")
    assert await turns.finish(
        SessionTurnKey("chat", "first"), generation=1, turn_no=1, state="succeeded", result={},
    )
    key = SessionTurnKey("chat", "second")
    assert await turns.accept(
        key, expected_revision=2, context_from=2, context_through=1,
        request_key="second", request={}, binding=_BINDING, lease_seconds=60,
    )
    admitted = await turns.get_turn(key)
    assert (admitted["context_from"], admitted["context_through"], admitted["turn_no"]) == (2, 1, 2)
    assert await turns.history("chat", after=admitted["context_from"] - 1,
                               through=admitted["context_through"], limit=128) == ()


@pytest.mark.asyncio
async def test_recent_context_reads_latest_positions_and_reuses_covered_history(store):
    engine, metadata, turns = store
    for number in range(1, 131):
        turn = f"turn-{number}"
        assert await accept(turns, turn, revision=2 * (number - 1), through=number - 1, request=turn)
        key = SessionTurnKey("chat", turn)
        if number == 130:
            assert await turns.request_cancel(key)
        assert await turns.finish(
            key, generation=1, turn_no=number, state="cancelled" if number == 130 else "succeeded", result={},
        )

    reads = []

    def capture(_connection, _cursor, statement, parameters, *_rest):
        if statement.startswith("SELECT"):
            reads.append(parameters)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    at_129 = await turns.recent_context("chat", through=129)
    assert [turn.turn_no for turn in at_129] == list(range(2, 130))
    warm = await turns.recent_context("chat", through=130)
    before_hit = len(reads)
    assert await turns.recent_context("chat", through=130) == warm
    assert before_hit == len(reads) == 2
    fresh = SqlSessionTurns(create_sql_storage_context(engine), metadata, namespace="test", tenant_id="a")
    assert await fresh.recent_context("chat", through=130) == warm
    assert [turn.turn_no for turn in warm] == list(range(3, 130))
    assert len(reads) == 3
    assert reads[1][2:4] == (129, 130)


@pytest.mark.asyncio
async def test_runtime_uses_its_conversation_route_for_new_turn_storage(tmp_path):
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    async with Runtime.open("test", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db"),
                            features=RuntimeFeatures("test"), metrics=None) as runtime:
        runner = runtime._session_service._regenerable_turns
        assert runner is not None
        turns = runtime._state.session_turns
        assert turns is not None
        await turns.create("chat", config={})
        assert (await turns.get_session("chat"))["session_key"] == b"chat"
