#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact native SQL Session identity, receipt and generation survive store reload."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey

from .test_session_turn_sql import _TRANSCRIPT, accept

pytestmark = pytest.mark.asyncio


async def test_native_sql_layout_replay_and_fence_expired_takeover(tmp_path):
    storage = PersistenceStorage.sqlite(tmp_path / "turns.db")
    async with (
        Persistence.open("turn-layout", storage=storage) as source,
        Persistence.open("turn-layout", storage=storage) as peer,
    ):
        first, other = source.session_turns, peer.session_turns
        assert first._context.engine is not other._context.engine
        await first.create("chat", config={})
        key = SessionTurnKey("chat", "first")
        assert await accept(first, key.turn_id, request="original")
        assert (await other.find_request("chat", "original"))["turn_id"] == b"first"
        assert (await other.get_session("chat"))["active_turn_id"] == b"first"
        assert not await accept(other, "competing", revision=1, request="competing")
        original = await other.get_turn(key)
        assert original["generation"] == 1
        assert not await other.take_over(key, generation=1, lease_seconds=60)
        assert await first.request_cancel(key)
        assert not await other.take_over(key, generation=1, lease_seconds=60, cancelling=True)
        assert (await other.get_turn(key))["state"] == "cancelling"

        async with first._context.engine.begin() as connection:
            expired = await connection.execute(first._turns.update().filter_by(
                **first._turn_values(key), generation=1, state="cancelling",
            ).values(lease_until=0))
            assert expired.rowcount == 1
        assert await other.take_over(key, generation=1, lease_seconds=60, cancelling=True)
        assert not await first.take_over(key, generation=1, lease_seconds=60, cancelling=True)
        assert not await first.finish(key, generation=1, turn_no=1, state="cancelled")
        assert await other.finish(key, generation=2, turn_no=1, state="cancelled")
        settled = await first.get_turn(key)
        assert settled["generation"] == 2 and settled["state"] == "cancelled"
        assert (await first.find_request("chat", "original"))["turn_id"] == b"first"
        assert (await first.get_session("chat"))["active_turn_id"] is None
        revision = (await first.get_session("chat"))["revision"]
        assert await accept(first, "next", revision=revision, through=1, request="new")
        assert await other.get_turn(key) == settled


async def test_native_sql_broken_current_owner_is_not_read_as_idle(tmp_path):
    storage = PersistenceStorage.sqlite(tmp_path / "broken.db")
    async with Persistence.open("turn-layout", storage=storage) as state:
        turns = state.session_turns
        await turns.create("chat", config={})
        key = SessionTurnKey("chat", "current")
        assert await accept(turns, key.turn_id, request="original")
        row = await turns.get_turn(key)
        async with turns._context.engine.begin() as connection:
            damaged = await connection.execute(turns._sessions.update().filter_by(
                **turns._session_values("chat"), active_turn_id=b"current",
            ).values(active_turn_id=b"missing-turn"))
            assert damaged.rowcount == 1

        occupied = await turns.get_session("chat")
        assert occupied["active_turn_id"] == b"missing-turn"
        assert await turns.get_turn(SessionTurnKey("chat", "missing-turn")) is None
        assert not await accept(turns, "competing", revision=occupied["revision"], request="new")
        assert not await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        assert await turns.find_request("chat", "new") is None
        assert await turns.get_session("chat") == occupied
        assert await turns.get_turn(key) == row


async def test_exact_published_identity_cannot_rewrite_its_native_body(tmp_path):
    storage = PersistenceStorage.sqlite(tmp_path / "published.db")
    key = SessionTurnKey("chat", "current")
    async with Persistence.open("turn-layout", storage=storage) as state:
        turns = state.session_turns
        await turns.create("chat", config={})
        assert await accept(turns, key.turn_id, request="original")
        assert await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        expected = await turns.get_turn(key)

    async with Persistence.open("turn-layout", storage=storage) as reopened:
        turns = reopened.session_turns
        owner = await turns.get_session("chat")
        receipt = await turns.find_request("chat", "original")
        assert receipt["turn_id"] == b"current"
        assert not await turns.take_over(key, generation=1, lease_seconds=60)
        assert not await turns.finish(
            key, generation=1, turn_no=1, state="succeeded",
            transcript=(ModelResponse(parts=[TextPart("unrelated replacement")]),),
        )
        # Same prior caller key must not manufacture another row or leave
        # occupancy set if the constraint rejects the new turn write.
        with pytest.raises(AIError) as duplicate:
            await accept(turns, "wrong", revision=owner["revision"], through=1, request="original")
        assert duplicate.value.code is ErrorCode.STORAGE_CONFLICT
        assert await turns.get_session("chat") == owner
        assert await turns.get_turn(key) == expected
        assert await turns.get_turn(SessionTurnKey("chat", "wrong")) is None
        result = await turns.get_history_turn(key, lineage=())
        assert result["messages"][0].text == "two"
