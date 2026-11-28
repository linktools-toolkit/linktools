#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session model context is selected before old message bodies are loaded."""

import asyncio
from contextlib import AsyncExitStack

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._session_turn_sql import SessionTurnKey, SqlSessionTurnReader
from linktools.ai.runtime import Runtime

from ._runtime_test_helpers import configure_execution
from .test_runtime_session_start_preparation import _features as _base_features


def _features(model):
    features = _base_features(model)
    configure_execution(features, retain_history=True)
    return features

pytestmark = pytest.mark.asyncio


class _SessionAuthorization(TenantAuthorizationPolicy):
    """Authorize sessions in the fixture's isolated namespace."""


@pytest.mark.parametrize("state", ("cancelled", "failed"))
@pytest.mark.parametrize("reopen", (False, True))
async def test_interrupted_turn_retains_the_request_for_continue(tmp_path, state, reopen):
    entered = asyncio.Event()
    attempts = []

    async def model(messages, info):
        prompts = [part for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        attempts.append([part.content for part in prompts])
        if len(attempts) == 1:
            if state == "failed":
                raise RuntimeError("synthetic provider failure")
            entered.set()
            await asyncio.Future()
        assert attempts[-1] == ["Inspect my files", "Continue"]
        assert "The previous turn was interrupted." in (info.instructions or "")
        return ModelResponse(parts=[TextPart("Continuing file inspection")])

    storage = PersistenceStorage.sqlite(tmp_path / "state.db")

    async def open_runtime(stack):
        persistence = await stack.enter_async_context(Persistence.open("interrupted-context", storage=storage))
        return await stack.enter_async_context(Runtime.open(
            persistence=persistence, features=_features(model), authorization=_SessionAuthorization(),
        ))

    async with AsyncExitStack() as stack:
        runtime = await open_runtime(stack)
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        first = await runtime.sessions.start_turn(
            "session", "Inspect my files", agent="default", principal=principal, idempotency_key="inspect",
        )
        if state == "cancelled":
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert await first.cancel()
        assert (await first.wait_final(timeout_seconds=5)).status is (
            ExecutionStatus.CANCELLED if state == "cancelled" else ExecutionStatus.FAILED
        )
        if reopen:
            await stack.aclose()
            runtime = await open_runtime(stack)
            principal = runtime.default_principal
        second = await runtime.sessions.start_turn(
            "session", "Continue", agent="default", principal=principal, idempotency_key="continue",
        )
        result = await second.wait_final(timeout_seconds=5)
        assert result.view["output"] == {"text": "Continuing file inspection"}


@pytest.mark.parametrize("selection,expected_messages", (("all", 3), ("current_turn", 1)))
async def test_session_selects_context_before_prior_body_read_and_preserves_timeline(
    tmp_path, monkeypatch, selection, expected_messages,
):
    seen = []
    ranges = []

    async def model(messages, info):
        seen.append(tuple(messages))
        return ModelResponse(parts=[TextPart("answer")])

    original = SqlSessionTurnReader.read_context_range

    async def read_context_range(self, session_id, **kwargs):
        result = await original(self, session_id, **kwargs)
        ranges.append((kwargs["after"], kwargs["through"], sum(len(turn.messages) for turn in result)))
        return result

    monkeypatch.setattr(SqlSessionTurnReader, "read_context_range", read_context_range)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")

    def open_runtime(state):
        return Runtime.open(
            persistence=state, features=_features(model), metrics=None, authorization=_SessionAuthorization(),
        )

    async with Persistence.open("session-start-preparation", storage=storage) as state, open_runtime(state) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        first = await runtime.agents.get().session("session").start("old question", idempotency_key="old")
        assert (await first.wait_final(timeout_seconds=5)).view["output"] == {"text": "answer"}

    ranges.clear()
    async with Persistence.open("session-start-preparation", storage=storage) as state, open_runtime(state) as runtime:
        agent = runtime.agents.get()
        principal = runtime.default_principal
        second = await agent.session("session").start(
            "new question", session_context=selection, idempotency_key="new",
        )
        assert (await second.wait_final(timeout_seconds=5)).view["output"] == {"text": "answer"}
        assert len(seen[-1]) == expected_messages
        assert ranges == ([] if selection == "current_turn" else [(0, 1, 2)])
        page = await runtime.history.turn_session("session", principal=principal)
        turns = page["timeline"]["items"]
        assert {turn["turn_id"] for turn in turns} == {first.key.turn_id, second.key.turn_id}
        assert {turn["user_input"]["prompt"] for turn in turns} == {"old question", "new question"}
        again = await agent.session("session").start(
            "new question", session_context=selection, idempotency_key="new",
        )
        assert again.key == second.key
        with pytest.raises(AIError) as error:
            await agent.session("session").start(
                "new question", idempotency_key="new",
                session_context="current_turn" if selection == "all" else "all",
            )
        assert error.value.code == ErrorCode.IDEMPOTENCY_CONFLICT

    async with Persistence.open("session-start-preparation", storage=storage) as state, open_runtime(state) as runtime:
        record = await state.session_turn_reader.get_turn(SessionTurnKey("session", second.key.turn_id))
        assert record["request_json"].get("session_context", "all") == selection
        page = await runtime.history.turn_session("session", principal=runtime.default_principal)
        assert len(page["timeline"]["items"]) == 2


async def test_current_turn_requires_a_session_and_rejects_unknown_selection(tmp_path):
    async def model(messages, info):
        raise AssertionError("invalid context selection must fail before the model")

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.open(
            persistence=state, features=_features(model), metrics=None, authorization=_SessionAuthorization(),
        ) as runtime:
            for kwargs in ({"session_context": "current_turn"}, {"session_id": "session", "session_context": "unknown"}):
                with pytest.raises(AIError) as error:
                    await runtime.agents.get().start("question", **kwargs)
                assert error.value.code == ErrorCode.REQUEST_FIELD_INVALID
