#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable acceptance, publication and fencing for regenerable turns."""

import asyncio

import pytest
import pytest_asyncio
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._message_root import NativeMessageRoot
from linktools.ai.persistence._session_turn_sql import (
    SESSION_BINDING_TABLE,
    SESSION_TURN_TABLE,
    SessionTurnKey,
    SqlSessionTurns,
)
from linktools.ai.storage import create_sql_storage_context
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine


@pytest_asyncio.fixture
async def store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'turns.db'}")
    try:
        await provision_database(engine)
        async with Persistence.open("test", storage=PersistenceStorage.sql(engine), tenant_id="a") as state:
            turns = state.session_turns
            assert turns is not None
            await turns.create("chat", config={})
            _, archive_provider, _ = turns._transcript_write_binding
            await archive_provider(create=True)
            yield engine, turns._turns.metadata, turns
    finally:
        await engine.dispose()


_TRANSCRIPT = (ModelResponse(parts=[TextPart("two")]),)
_BINDING = {"agent_spec": {"id": "test-agent", "instructions": ["frozen"]}}


async def accept(turns: SqlSessionTurns, turn: str, *, revision: int = 0, through: int = 0, request: str = "send") -> bool:
    return await turns.accept(
        SessionTurnKey("chat", turn), expected_revision=revision, context_from=1, context_through=through,
        request_key=request, request={"prompt": turn}, binding=_BINDING, lease_seconds=60,
    )


@pytest.mark.asyncio
async def test_regenerable_turn_has_two_write_boundaries_and_one_visible_record(store):
    engine, metadata, turns = store
    statements = []
    native_writes = []
    commits = []

    def on_sql(_conn, _cursor, statement, *_args):
        verb = statement.split()[0]
        if verb in ("SELECT", "INSERT", "UPDATE", "DELETE") and any(
            table in statement for table in ("ai_sessions", "ai_session_turns", "ai_session_bindings")
        ):
            statements.append(verb)
        if verb == "INSERT" and any(table in statement for table in ("ai_messages", "ai_message_sequences")):
            native_writes.append(statement)

    def on_commit(_conn):
        commits.append((len(statements), len(native_writes)))

    event.listen(engine.sync_engine, "before_cursor_execute", on_sql)
    event.listen(engine.sync_engine, "commit", on_commit)
    try:
        assert await turns.get_session("chat") is not None
        key = SessionTurnKey("chat", "turn")
        assert await turns.get_history_turn(key, lineage=()) is None
        assert await turns.read_context_range("chat", after=0, through=0) == ()
        # History captures metadata and (when present) body ownership in one
        # read snapshot; separate its zero-write commit from the two mutations.
        assert statements == ["SELECT", "SELECT"] and commits == [(2, 0)]
        commits.clear()
        assert await accept(turns, key.turn_id)
        assert await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        assert statements == ["SELECT", "SELECT", "UPDATE", "INSERT", "INSERT", "UPDATE", "UPDATE"]
        assert len(native_writes) == 2
        assert commits == [(5, 0), (7, 2)]
        assert await accept(turns, "second", revision=2, through=1, request="second")
        assert statements[7:] == ["UPDATE", "INSERT"]
        assert commits == [(5, 0), (7, 2), (9, 2)]
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", on_sql)
        event.remove(engine.sync_engine, "commit", on_commit)

    session = await turns.get_session("chat")
    assert session["active_turn_id"] == b"second" and session["last_turn_no"] == 1
    turn = await turns.get_turn(key)
    assert turn["request_json"] == {"prompt": "turn"}
    assert set(turn["result_json"]) == {"transcript", "binding_refs"}
    root = NativeMessageRoot.from_json(turn["result_json"]["transcript"])
    assert root.count == 1 and turn["result_json"]["binding_refs"] == []
    published = await turns.get_history_turn(key, lineage=())
    assert published["messages"][0].parts[0].content == "two"
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
    assert await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)

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
    assert not await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
    assert (await turns.get_session("chat"))["active_turn_id"] == b"turn"
    assert await turns.request_cancel(key)
    assert not await turns.finish(key, generation=2, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
    assert (await turns.get_session("chat"))["active_turn_id"] == b"turn"
    assert await turns.finish(key, generation=2, turn_no=1, state="cancelled", result=None)
    assert (await turns.get_turn(key))["state"] == "cancelled"
    async with engine.connect() as connection:
        assert (await connection.execute(select(func.count()).select_from(metadata.tables["ai_message_sequences"]))).scalar_one() == 0
    assert await turns.read_context_range("chat", after=0, through=1) == ()


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
        async with Persistence.open(
            "test", storage=PersistenceStorage.sql(restarted_engine), tenant_id="a",
        ) as restarted_state:
            restarted = restarted_state.session_turns
            assert restarted is not None
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

            visible = await restarted.get_history_turn(key, lineage=())
            assert (visible["turn_id"], visible["state"], visible["result_json"]) == (b"turn", "cancelled", None)
            assert await restarted.read_context_range("chat", after=0, through=1) == ()
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
        SessionTurnKey("chat", "first"), generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT,
    )
    key = SessionTurnKey("chat", "second")
    assert await turns.accept(
        key, expected_revision=2, context_from=2, context_through=1,
        request_key="second", request={}, binding=_BINDING, lease_seconds=60,
    )
    admitted = await turns.get_turn(key)
    assert (admitted["context_from"], admitted["context_through"], admitted["turn_no"]) == (2, 1, 2)
    assert await turns.read_context_range(
        "chat", after=admitted["context_from"] - 1, through=admitted["context_through"],
    ) == ()


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
            key, generation=1, turn_no=number, state="cancelled" if number == 130 else "succeeded",
            transcript=None if number == 130 else _TRANSCRIPT,
        )

    reads = []
    body_reads = []

    def capture(_connection, _cursor, statement, parameters, *_rest):
        if statement.lstrip().startswith(("SELECT", "WITH")):
            if "ai_session_turns" in statement:
                reads.append(parameters)
            if "ai_message_sequences" in statement and "ai_messages" in statement:
                body_reads.append(parameters)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        at_129 = await turns.recent_context("chat", through=129)
        assert [turn.turn_no for turn in at_129] == list(range(2, 130))
        assert (len(reads), len(body_reads)) == (1, 1)
        warm = await turns.recent_context("chat", through=130)
        assert (len(reads), len(body_reads)) == (2, 1)
        assert await turns.recent_context("chat", through=130) == warm
        assert (len(reads), len(body_reads)) == (2, 1)
        fresh = SqlSessionTurns(create_sql_storage_context(engine), metadata, namespace="test", tenant_id="a")
        fresh.bind_transcript_reader(*turns._transcript_source)
        assert await fresh.recent_context("chat", through=130) == warm
        assert [turn.turn_no for turn in warm] == list(range(3, 130))
        assert (len(reads), len(body_reads)) == (3, 2)
        # Cancelled turn 130 advances the range without needing a body read.
        assert {value for value in reads[1] if type(value) is int and value > 128} == {129, 130}
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
async def test_runtime_uses_its_conversation_route_for_new_turn_storage(tmp_path):
    from linktools.ai.core import AuthorizationAction, Principal, ResourceKind, TenantAuthorizationPolicy
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    class ChatAuthorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if (principal.principal_id, action, resource.kind, resource.id) != (
                "tester", AuthorizationAction.SESSION_READ, ResourceKind.SESSION, "chat",
            ):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async with Runtime.open("test", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db"),
                            features=RuntimeFeatures("test"), metrics=None,
                            tenant_id="default", authorization=ChatAuthorization()) as runtime:
        runner = runtime._session_service._regenerable_turns
        assert runner is not None
        turns = runtime._state.session_turns
        assert turns is not None
        await turns.create("chat", config={})
        state = await runtime.history.turn_session_state(
            "chat", principal=Principal(principal_id="tester", tenant_id="default"),
        )
        assert state["session_id"] == "chat"


@pytest.mark.asyncio
async def test_ordinary_snapshot_remaps_native_root_without_needing_step_checkpoints(tmp_path):
    from linktools.ai.snapshot import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore

    source = PersistenceStorage.from_root(tmp_path / "source")
    archive = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)
    key = SessionTurnKey("chat", "turn")
    target = tmp_path / "restored"
    async with Persistence.open("test", storage=source, tenant_id="a") as state:
        turns = state.session_turns
        assert turns is not None
        await turns.create("chat", config={})
        assert await turns.accept(
            key, expected_revision=0, context_from=1, context_through=0, request_key="send",
            request={"prompt": "question", "principal": {"principal_id": "owner", "tenant_id": "a"}},
            binding=_BINDING, lease_seconds=60,
        )
        assert await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        source_row = await turns.get_turn(key)
        source_root = NativeMessageRoot.from_json(source_row["result_json"]["transcript"])
    async with Persistence.open("test", storage=source, tenant_id="a", read_only=True) as state:
        reference = await state.export_snapshot(object_store=archive, limits=limits)

    await Persistence.restore_snapshot(reference, object_store=archive, root=target, limits=limits)
    async with Persistence.open(
        "test", storage=PersistenceStorage.from_root(target), tenant_id="a", read_only=True,
    ) as state:
        turns = state.session_turn_reader
        restored = await turns.get_history_turn(key, lineage=())
        restored_root = NativeMessageRoot.from_json(restored["result_json"]["transcript"])
        assert restored_root.group_id != source_root.group_id
        assert restored_root.store_key == source_root.store_key and restored_root.slices == source_root.slices
        assert [part.content for message in restored["messages"] for part in message.parts] == ["two"]
