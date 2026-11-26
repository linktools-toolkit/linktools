#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SQL session display always checks current membership before reading bodies."""

import asyncio

import pytest

from linktools.ai.core import AuthorizationAction, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryCacheBackend

from .test_pending_session_projection import _runtimes

pytestmark = pytest.mark.asyncio


async def _start(runtime, session_id, key, prompt):
    return await runtime.sessions.start(
        session_id, prompt, agent="default", principal=runtime.default_principal, idempotency_key=key,
    )


async def test_native_display_rechecks_policy_before_loading_transcript(tmp_path, monkeypatch):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    async with _runtimes(tmp_path, cache=cache, shared=False, native=True) as (runtimes, *_):
        runtime, storage = runtimes[0]
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session", principal=principal)
        turn = await _start(runtime, "session", "one", "question")
        result = await turn.wait_final(timeout_seconds=5)
        assert (await session.timeline())["items"] == [result.view]
        authorization = runtime._authorization
        authorize = authorization.authorize
        requests = []

        async def revoked(actor, action, resource):
            await authorize(actor, action, resource)
            requests.append((action, resource.id))
            if action is AuthorizationAction.SESSION_READ:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        async def no_bodies(*args, **kwargs):
            pytest.fail("revoked membership cannot reach transcript reads")

        with monkeypatch.context() as patch:
            patch.setattr(authorization, "authorize", revoked)
            patch.setattr(storage.session_turns, "_read_transcripts", no_bodies)
            with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
                await session.history()
            with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
                await runtime.history.turn_result("session", turn.key.turn_id, principal=principal)
        assert requests == [(AuthorizationAction.SESSION_READ, "session")] * 2


async def test_native_display_sees_external_revision_without_age_window(tmp_path):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    async with _runtimes(tmp_path, cache=cache, native=True) as (runtimes, *_):
        owner, _ = runtimes[0]
        reader, _ = runtimes[1]
        session = await owner.agents.get().create_session("session", principal=owner.default_principal)
        principal = reader.default_principal
        before = await reader.sessions.refresh("session", principal=principal)
        assert await session.update(expected_revision=before["session"]["revision"], metadata={"value": "changed"})
        fresh = await reader.sessions.refresh("session", principal=principal)
        assert fresh["session"]["revision"] > before["session"]["revision"]
        assert fresh["session"]["metadata"] == {"value": "changed"}
        assert await reader.sessions.refresh("session", principal=principal) == fresh
        with pytest.raises(ValueError, match="max_age=0"):
            await reader.sessions.refresh("session", principal=principal, max_age=3)


@pytest.mark.parametrize("damage", ["session", "result"])
async def test_native_display_does_not_serve_previous_membership_or_result(tmp_path, damage):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    async with _runtimes(tmp_path, cache=cache, native=True) as (runtimes, *_):
        owner, state = runtimes[0]
        reader, _ = runtimes[1]
        await owner.agents.get().create_session("session", principal=owner.default_principal)
        turn = await _start(owner, "session", "one", "question")
        terminal = await turn.wait_final(timeout_seconds=5)
        principal = reader.default_principal
        assert (await reader.sessions.refresh("session", principal=principal))["timeline"]["items"] == [terminal.view]
        assert await reader.history.turn_result("session", turn.key.turn_id, principal=principal) == terminal.view
        store = state.session_turns
        async with store._context.engine.begin() as connection:
            if damage == "session":
                table = store._sessions
                # Keep the turn durable while removing the owning Session row.
                await connection.execute(table.delete().where(table.c.session_key == b"session"))
            else:
                table = store._turns
                await connection.execute(table.update().where(table.c.turn_id == turn.key.turn_id.encode()).values(result_json={}))
        code = ErrorCode.AUTHORIZATION_DENIED if damage == "session" else ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as page_error:
            await reader.sessions.refresh("session", principal=principal)
        assert page_error.value.code is code
        with pytest.raises(AIError) as exact_error:
            await reader.history.turn_result("session", turn.key.turn_id, principal=principal)
        assert exact_error.value.code is code


@pytest.mark.parametrize("limit", (1, 2, 10))
@pytest.mark.parametrize("with_history", (False, True))
async def test_native_page_keeps_occupancy_separate_from_committed_history(tmp_path, limit, with_history):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    async with _runtimes(tmp_path, cache=cache, native=True) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, _ = runtimes[1]
        await owner.agents.get().create_session("session", principal=owner.default_principal)
        principal = reader.default_principal
        prior = []
        if with_history:
            previous = await _start(owner, "session", "prior", "prior question")
            prior = [(await previous.wait_final(timeout_seconds=5)).view]
        baseline = await reader.sessions.refresh("session", principal=principal, limit=limit)
        assert baseline["timeline"]["items"] == prior
        block()
        active = await _start(owner, "session", "active", "live question")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            for _ in range(2):
                current = await reader.sessions.refresh("session", principal=principal, limit=limit)
                assert current["session"]["active_turn_id"] == active.key.turn_id
                assert current["timeline"] == baseline["timeline"]
                exact = await reader.history.turn_result("session", active.key.turn_id, principal=principal)
                assert not exact["conversation_committed"]
                assert exact["user_input"]["prompt"] == "live question"
        finally:
            finish.set()
        result = await active.wait_final(timeout_seconds=5)
        assert result.status is ExecutionStatus.SUCCEEDED
        settled = await reader.sessions.refresh("session", principal=principal, limit=limit)
        assert settled["session"]["active_turn_id"] is None
        assert settled["timeline"]["items"] == [*prior, result.view][-limit:]
        if with_history and limit == 1:
            older = await reader.sessions.refresh(
                "session", principal=principal, cursor=settled["timeline"]["next_cursor"], limit=limit,
            )
            assert older["timeline"]["items"] == prior
        else:
            assert settled["timeline"]["next_cursor"] is None
        with pytest.raises(AIError):
            await reader.sessions.refresh("session", principal=principal, limit=0)
