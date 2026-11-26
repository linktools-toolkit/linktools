#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Only the current canonical SQL generation can publish Step or terminal state."""

import asyncio

import pytest
from sqlalchemy import event
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext
from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.turns import SessionTurnKey

from .test_grouped_session_admission import _native_runtime, _native_start
from .test_session_turn_sql import _TRANSCRIPT, accept, store

pytestmark = pytest.mark.asyncio

_KEY = SessionTurnKey("chat", "turn")


async def _guard(turns, *, generation=1, mutation=None):
    owner = turns.transcript_state_store

    async def apply(scope):
        await turns.guard_step_in_transaction(scope.transaction(owner), _KEY, generation=generation)
        if mutation is not None:
            await mutation()

    await owner.storage_group.mutate((owner,), apply)


async def _expire(engine, turns):
    async with engine.begin() as connection:
        changed = await connection.execute(
            turns._turns.update().filter_by(**turns._turn_values(_KEY), generation=1).values(lease_until=0),
        )
        assert changed.rowcount == 1


@pytest.mark.parametrize("takeover", (False, True))
async def test_expiry_or_takeover_blocks_stale_step_and_terminal(store, takeover):
    engine, _metadata, turns = store
    assert await accept(turns, _KEY.turn_id)
    await _guard(turns)
    await _expire(engine, turns)
    if takeover:
        assert await turns.take_over(_KEY, generation=1, lease_seconds=60)
    before = await turns.get_turn(_KEY)
    session = await turns.get_session(_KEY.session_id)

    with pytest.raises(AIError) as old:
        await _guard(turns)
    assert old.value.code is ErrorCode.STORAGE_CONFLICT
    assert not await turns.finish(_KEY, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
    assert await turns.get_turn(_KEY) == before
    assert await turns.get_session(_KEY.session_id) == session
    if takeover:
        assert before["generation"] == 2
        await _guard(turns, generation=2)
        assert await turns.finish(_KEY, generation=2, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        assert (await turns.get_session(_KEY.session_id))["active_turn_id"] is None


async def test_database_writer_wait_cannot_use_a_stale_live_lease_sample(store):
    engine, _metadata, turns = store
    assert await accept(turns, _KEY.turn_id)
    waiting = asyncio.Event()
    entered = []

    def observe_guard(_connection, _cursor, statement, *_):
        if statement.lstrip().upper().startswith("UPDATE AI_SESSION_TURNS SET GENERATION="):
            waiting.set()

    async def guarded_write():
        entered.append(True)

    async with engine.begin() as other:
        # Hold the actual SQLite writer lock and make expiry visible only
        # when it is released, after the old actor has submitted its UPDATE.
        await other.execute(turns._turns.update().filter_by(
            **turns._turn_values(_KEY), generation=1,
        ).values(lease_until=0))
        event.listen(engine.sync_engine, "before_cursor_execute", observe_guard)
        attempted = asyncio.create_task(_guard(turns, mutation=guarded_write))
        try:
            await asyncio.wait_for(waiting.wait(), 5)
            assert not attempted.done() and not entered
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", observe_guard)
    with pytest.raises(AIError) as stale:
        await asyncio.wait_for(attempted, 5)
    assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    assert not entered


async def test_retry_after_rollback_must_present_original_generation(store):
    engine, _metadata, turns = store
    assert await accept(turns, _KEY.turn_id)
    calls = []

    async def retryable_write():
        calls.append(True)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    with pytest.raises(AIError) as rolled_back:
        await _guard(turns, mutation=retryable_write)
    assert rolled_back.value.code is ErrorCode.STORAGE_CONFLICT and calls == [True]
    assert (await turns.get_turn(_KEY))["generation"] == 1

    await _expire(engine, turns)
    assert await turns.take_over(_KEY, generation=1, lease_seconds=60)
    with pytest.raises(AIError) as stale:
        await _guard(turns, mutation=retryable_write)
    assert stale.value.code is ErrorCode.STORAGE_CONFLICT and calls == [True]
    await _guard(turns, generation=2)

async def test_revoked_turn_preserves_the_already_admitted_tool_receipt(tmp_path, monkeypatch):
    from linktools.ai.runtime._tool import RuntimeToolOperationBridge

    entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    effects = []
    original_complete = RuntimeToolOperationBridge.complete

    async def work(context: AgentContext[None]) -> str:
        effects.append(context.execution_id)
        entered.set()
        await release.wait()
        return "known"

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("old producer cannot publish this")])
        return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="original-call")])

    async def complete(bridge, decision, result):
        confirmed = await original_complete(bridge, decision, result)
        settled.set()
        return confirmed

    monkeypatch.setattr(RuntimeToolOperationBridge, "complete", complete)
    async with _native_runtime(tmp_path, model, tool=work) as (state, runtime):
        await runtime.agents.get().create_session("session")
        turn = await _native_start(runtime)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            turns, tools = state.session_turns, state.checkpoint.tools
            [claimed] = await tools.list_by_execution(turn.key.turn_id, tenant_id=runtime.tenant_id)
            assert claimed.status is ToolOperationStatus.CLAIMED
            async with turns._context.engine.begin() as connection:
                expired = await connection.execute(
                    turns._turns.update().filter_by(**turns._turn_values(turn.key), generation=1).values(lease_until=0),
                )
                assert expired.rowcount == 1
            assert await turns.take_over(turn.key, generation=1, lease_seconds=60)
            release.set()
            await asyncio.wait_for(settled.wait(), 5)
            [receipt] = await tools.list_by_execution(turn.key.turn_id, tenant_id=runtime.tenant_id)
            assert receipt.status is ToolOperationStatus.COMPLETED and receipt.result_payload is not None
            assert (receipt.tool_operation_id, receipt.owner, receipt.fence, receipt.producer_generation) == (
                claimed.tool_operation_id, claimed.owner, claimed.fence, claimed.producer_generation,
            )
            assert effects == [turn.key.turn_id]
            row = await turns.get_turn(turn.key)
            assert row["generation"] == 2 and row["state"] == "accepted"
        finally:
            release.set()
