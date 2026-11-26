#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical SQL timeline windows project a bounded committed prefix."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from sqlalchemy import event
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.capability import SessionCapability
from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HostAuthorization(TenantAuthorizationPolicy):
    pass


@asynccontextmanager
async def _window_host(tmp_path):
    waiting, proceed = asyncio.Event(), asyncio.Event()

    async def respond(messages, _info):
        prompt = [part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)]
        if prompt == ["pending"]:
            waiting.set()
            await proceed.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("window")
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(_UsageFunctionModel(respond)), features)
    async with (
        Persistence.open("window", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        principal = Principal("owner", "tenant")
        await runtime.sessions.create_turn_session("session", principal=principal)
        try:
            yield state, runtime, principal, waiting, proceed
        finally:
            proceed.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (0, 1, 4))
@pytest.mark.parametrize("pending", (False, True))
async def test_window_matches_counter_and_page(tmp_path, count, pending):
    async with _window_host(tmp_path) as (state, runtime, principal, waiting, proceed):
        for index in range(count):
            turn = await runtime.sessions.start(
                "session", f"turn-{index}", agent="default", principal=principal, idempotency_key=f"key-{index}",
            )
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        active = None
        if pending:
            active = await runtime.sessions.start(
                "session", "pending", agent="default", principal=principal, idempotency_key="key-pending",
            )
            await asyncio.wait_for(waiting.wait(), 5)
        engine = state.session_turns._context.engine.sync_engine
        statements = []

        def observe(_connection, clause, _multiparams, _params, _execution_options):
            if clause.is_select and state.session_turns._turns in clause.get_final_froms():
                statements.append(clause)

        cursor = None
        end = count
        while True:
            event.listen(engine, "before_execute", observe)
            try:
                page = await runtime.history.turn_session("session", principal=principal, cursor=cursor, limit=2)
            finally:
                event.remove(engine, "before_execute", observe)
            assert page["session"]["last_turn_no"] == count
            assert page["session"]["active_turn_id"] == (None if active is None else active.key.turn_id)
            assert [item["turn_no"] for item in page["timeline"]["items"]] == list(range(max(1, end - 1), end + 1))
            assert len(statements) == int(end > 0)
            if statements:
                clause, = statements
                assert clause._limit == 3 and clause._offset is None
                for dialect in (sqlite.dialect(), mysql.dialect(), postgresql.dialect()):
                    sql = str(clause.compile(dialect=dialect))
                    assert "ORDER BY ai_session_turns.turn_no DESC" in sql
                statements.clear()
            end -= 2
            cursor = page["timeline"]["next_cursor"]
            assert bool(cursor) is (end > 0)
            if cursor is None:
                break
        with pytest.raises(AIError) as invalid:
            await runtime.history.turn_session("session", principal=principal, cursor="wrong")
        assert invalid.value.code is ErrorCode.CURSOR_INVALID
        if active is not None:
            proceed.set()
            assert (await active.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("missing-transcript", "wrong-root", "bad-binding-ref", "bad-output-position"))
async def test_window_keeps_corruption_checks(tmp_path, damage):
    async with _window_host(tmp_path) as (state, runtime, principal, _waiting, _proceed):
        turn = await runtime.sessions.start(
            "session", "turn", agent="default", principal=principal, idempotency_key="key",
        )
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        store = state.session_turns
        result = dict((await store.get_turn(turn.key))["result_json"])
        if damage == "missing-transcript":
            del result["transcript"]
        elif damage == "wrong-root":
            result["transcript"] = "not a root"
        elif damage == "bad-binding-ref":
            result["binding_refs"] = [[-1, None]]
        else:
            result["output_ref"] = [1000, 0]
        async with store._context.engine.begin() as connection:
            changed = await connection.execute(store._turns.update().filter_by(
                **store._turn_values(turn.key),
            ).values(result_json=result))
            assert changed.rowcount == 1
        with pytest.raises(AIError) as cursor_error:
            await runtime.history.turn_session("session", principal=principal, cursor="wrong")
        assert cursor_error.value.code is ErrorCode.CURSOR_INVALID
        with pytest.raises(AIError) as caught:
            await runtime.history.turn_session("session", principal=principal)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_memory_window_uses_existing_fallback():
    async with Persistence.open("window", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        assert await state.conversation.sessions.capture_timeline_window(
            "session", tenant_id="tenant", before=None, limit=1, pending=False,
        ) is None


def test_joined_fact_adapter_does_not_require_sequence_windows():
    from linktools.ai.persistence.state import SqlStateFactReader, SqlStateSequenceReader

    class JoinedOnly:
        async def read_joined_facts(self, context, statement):
            return ()

    adapter = JoinedOnly()
    assert isinstance(adapter, SqlStateFactReader)
    assert not isinstance(adapter, SqlStateSequenceReader)
