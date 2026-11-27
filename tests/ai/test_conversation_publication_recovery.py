#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication and recovery boundaries of the native SQLite Session SDK."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import AgentContext, SessionCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

_NAMESPACE = "native-publication-recovery"
_SESSION = "session"


class _Authorization(TenantAuthorizationPolicy):
    """Explicit host policy for native turns."""


def _features(calls, effects):
    async def effect(context: AgentContext[None]) -> str:
        del context
        effects.append("once")
        return "confirmed"

    async def model(messages, info):
        calls.append((tuple(
            part.content
            for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)
        ), info.instructions or "", tuple(
            part.content
            for message in messages if isinstance(message, ModelResponse)
            for part in message.parts if isinstance(part, TextPart)
        )))
        if isinstance(messages[-1], ModelRequest):
            if any(isinstance(part, UserPromptPart) and part.content == "effect input" for part in messages[-1].parts):
                return ModelResponse(parts=[ToolCallPart("effect", {}, "original-effect")])
            for part in messages[-1].parts:
                if isinstance(part, ToolReturnPart):
                    assert part.content == "confirmed"
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures(_NAMESPACE)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.tool(effect, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("effect",))
    return model_features(_Models(_UsageFunctionModel(model)), features)


@asynccontextmanager
async def _host(storage, calls, effects):
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=_features(calls, effects), authorization=_Authorization()) as runtime,
    ):
        yield state, runtime


async def _send(runtime, prompt, request_id):
    return await runtime.sessions.start(
        _SESSION, prompt, principal=runtime.default_principal,
        agent=runtime.agents.get(), idempotency_key=request_id,
    )


async def _expire(turns, key, *, generation=1):
    async with turns._context.engine.begin() as connection:
        expired = await connection.execute(turns._turns.update().filter_by(
            **turns._turn_values(key), generation=generation, state="accepted",
        ).values(lease_until=0))
        assert expired.rowcount == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_turn", [False, True])
async def test_unknown_release_must_not_report_success_without_envelope(tmp_path, monkeypatch, prior_turn):
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    calls, effects = [], []
    rejected = asyncio.Event()
    async with _host(storage, calls, effects) as (state, runtime):
        principal = runtime.default_principal
        await runtime.agents.get().create_session(_SESSION, principal=principal)
        if prior_turn:
            prior = await _send(runtime, "prior committed turn", "prior")
            assert (await prior.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED

        assert state.session_turns is not None

        async def lose_release(_key, *, state, **_options):
            assert state == "succeeded"
            rejected.set()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(state.session_turns, "finish", lose_release)
            turn = await _send(runtime, "effect input", "request")
            await asyncio.wait_for(rejected.wait(), 5)
            with pytest.raises(AIError) as raised:
                await turn.wait_final(timeout_seconds=5)
            assert raised.value.code in {ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_RECOVERY_REQUIRED}
            view = await runtime.history.turn_result(_SESSION, turn.key.turn_id, principal=principal)
            assert not view["conversation_committed"] and view["output"] is None
            session = await runtime.history.turn_session_state(_SESSION, principal=principal)
            assert session["active_turn_id"] == turn.key.turn_id
            page = await runtime.history.turn_session(_SESSION, principal=principal)
            assert len(page["timeline"]["items"]) == int(prior_turn)
            async with (
                Persistence.open(_NAMESPACE, storage=storage, read_only=True) as reader,
                History.open(persistence=reader, authorization=_Authorization()) as history,
            ):
                cold_view = await history.turn_result(_SESSION, turn.key.turn_id, principal=principal)
                assert cold_view["conversation_committed"] is False and cold_view["output"] is None
        receipts = await state.checkpoint.tools.list_by_execution(turn.key.turn_id, tenant_id=principal.tenant_id)
        assert len(receipts) == 1
        await _expire(state.session_turns, turn.key)

    async with _host(storage, calls, effects) as (state, runtime):
        joined = await _send(runtime, "effect input", "request")
        assert joined.key == turn.key
        assert await runtime.sessions.recover_turn(_SESSION, turn.key.turn_id, principal=principal)
        result = await joined.wait_final(timeout_seconds=5)
        page = await runtime.history.turn_session(_SESSION, principal=principal)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert len(page["timeline"]["items"]) == 1 + int(prior_turn)
        assert page["timeline"]["items"][-1] == result.view
        assert await state.checkpoint.tools.list_by_execution(turn.key.turn_id, tenant_id=principal.tenant_id) == receipts
    assert len(calls) == 2 + int(prior_turn) and effects == ["once"]


@pytest.mark.asyncio
async def test_admitted_failure_retains_input_without_publishing_an_answer(tmp_path, monkeypatch):
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    calls, effects = [], []
    async with _host(storage, calls, effects) as (_, runtime):
        principal = runtime.default_principal
        await runtime.agents.get().create_session(_SESSION, principal=principal)

        async def reject(*_args, **_kwargs):
            raise AIError(ErrorCode.SESSION_CONFLICT)

        runner = runtime.sessions._require_turn_sessions()[0]
        with monkeypatch.context() as patch:
            patch.setattr(runner._executor, "execute_regenerable", reject)
            turn = await _send(runtime, "original rejected input", "rejected")
            failed = await turn.wait_final(timeout_seconds=5)
        assert failed.status is ExecutionStatus.FAILED
        assert failed.view["user_input"]["prompt"] == "original rejected input"
        assert failed.view["output"] is None and failed.view["error_code"] == ErrorCode.SESSION_CONFLICT.value
        assert not calls
        next_turn = await _send(runtime, "next real input", "next")
        assert (await next_turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        page = await runtime.history.turn_session(_SESSION, principal=principal)
        first, second = page["timeline"]["items"]
        assert first == failed.view
        assert second["user_input"]["prompt"] == "next real input"
        assert second["output"] == {"text": "answer"}
        [(prompts, instruction, previous_answers)] = calls
        assert prompts == ("original rejected input", "next real input")
        assert "The previous turn was interrupted. Its results were not confirmed." in instruction
        assert previous_answers == ("[Runtime: This turn was interrupted before a final answer.]",)
        assert effects == []


@pytest.mark.asyncio
async def test_corrupted_native_turn_receipt_cannot_masquerade_as_valid_history(tmp_path):
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    calls, effects = [], []
    async with _host(storage, calls, effects) as (state, runtime):
        principal = runtime.default_principal
        await runtime.agents.get().create_session(_SESSION, principal=principal)
        intact = await _send(runtime, "retained input", "intact")
        assert (await intact.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        damaged = await _send(runtime, "damaged input", "damaged")
        assert (await damaged.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        turns = state.session_turns
        assert turns is not None
        row = await turns.get_turn(damaged.key)
        assert row is not None
        async with turns._context.engine.begin() as connection:
            updated = await connection.execute(turns._turns.update().filter_by(
                **turns._turn_values(damaged.key),
            ).values(result_json={**row["result_json"], "transcript": {}}))
            assert updated.rowcount == 1
        assert (await runtime.history.turn_result(_SESSION, intact.key.turn_id, principal=principal))["output"] == {"text": "answer"}
        with pytest.raises(AIError) as timeline_error:
            await runtime.history.turn_session(_SESSION, principal=principal)
        assert timeline_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    async with (
        Persistence.open(_NAMESPACE, storage=storage, read_only=True) as reader,
        History.open(persistence=reader, authorization=_Authorization()) as history,
    ):
        with pytest.raises(AIError) as retained_error:
            await history.turn_result(_SESSION, damaged.key.turn_id, principal=principal)
        assert retained_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_completed_receipt_old_handoff_after_later_turn(tmp_path, monkeypatch):
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    calls, effects = [], []
    async with _host(storage, calls, effects) as (state, runtime):
        principal = runtime.default_principal
        await runtime.agents.get().create_session(_SESSION, principal=principal)
        turns = state.session_turns
        assert turns is not None
        finish = turns.finish

        async def lose_acknowledgement(key, **options):
            assert await finish(key, **options)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(turns, "finish", lose_acknowledgement)
            first = await _send(runtime, "effect input", "first")
            first_result = await first.wait_final(timeout_seconds=5)
            assert first_result.status is ExecutionStatus.SUCCEEDED
        second = await _send(runtime, "second", "second")
        assert (await second.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        page = await runtime.history.turn_session(_SESSION, principal=principal)

    async with _host(storage, calls, effects) as (_, runtime):
        assert not await runtime.sessions.recover_turn(_SESSION, first.key.turn_id, principal=principal)
        replay = await _send(runtime, "effect input", "first")
        assert replay.key == first.key
        assert (await replay.wait_final(timeout_seconds=5)).view == first_result.view
        assert await runtime.history.turn_session(_SESSION, principal=principal) == page
        state = await runtime.history.turn_session_state(_SESSION, principal=principal)
        assert state["active_turn_id"] is None
    assert len(calls) == 3 and effects == ["once"]
