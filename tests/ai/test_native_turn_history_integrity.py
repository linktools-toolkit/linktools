#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Committed history pages reject missing canonical turn rows."""

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode

from .test_session_timeline_window import _window_host


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("newest", "lookahead", "inherited"))
async def test_committed_history_rejects_missing_turn(tmp_path, damage):
    async with _window_host(tmp_path) as (state, runtime, principal, _, _):
        turns = []
        for number in range(1, 5):
            turn = await runtime.sessions.start_turn(
                "session", f"turn-{number}", agent="default",
                principal=principal, idempotency_key=f"key-{number}",
            )
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
            turns.append(turn)

        history_session = "session"
        if damage == "inherited":
            source = await runtime.history.turn_session_state("session", principal=principal)
            assert await runtime.sessions.fork_turn_session(
                "session", "fork", principal=principal,
                expected_revision=source["revision"], idempotency_key="fork",
            )
            turn = await runtime.sessions.start_turn(
                "fork", "new branch turn", agent="default",
                principal=principal, idempotency_key="fork-turn",
            )
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
            history_session = "fork"

        page = await runtime.history.turn_session(history_session, principal=principal, limit=2)
        newest = 5 if history_session == "fork" else 4
        assert [item["turn_no"] for item in page["timeline"]["items"]] == [newest - 1, newest]
        assert page["timeline"]["next_cursor"] is not None

        missing = turns[1] if damage == "lookahead" else turns[-1]
        store = state.session_turns
        async with store._context.engine.begin() as connection:
            removed = await connection.execute(store._turns.delete().filter_by(**store._turn_values(missing.key)))
            assert removed.rowcount == 1

        with pytest.raises(AIError) as caught:
            await runtime.history.turn_session(history_session, principal=principal, limit=2)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
