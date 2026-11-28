#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, SessionCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime.turn_clients import TurnObserver
from linktools.ai.runtime._tool import RuntimeToolOperationBridge

from ._runtime_test_helpers import configure_execution
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
async def test_native_terminal_unknown_keeps_slot_until_exact_late_commit(tmp_path, monkeypatch):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Provide the explicit authority for canonical session reads and writes."""

    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("native-unknown-terminal", defaults={"execution": {"retain_history": True}})
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.from_root(tmp_path / "native")
    async with (
        Persistence.open("native-unknown-terminal", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=HostAuthorization()) as runtime,
    ):
        await runtime.agents.get().create_session("session")
        turns = state.session_turns
        original = turns.finish
        release = asyncio.Event()
        pending = []
        principal = runtime.default_principal
        agent = runtime.agents.get()

        async def start(prompt="question", request_id="request"):
            return await runtime.sessions.start("session", prompt, agent=agent,
                                                principal=principal, idempotency_key=request_id)

        async def lose_ack(key, **kwargs):
            assert kwargs["state"] == "succeeded" and kwargs["generation"] == 1
            assert not pending

            async def commit_later():
                await release.wait()
                return await original(key, **kwargs)

            pending.append(asyncio.create_task(commit_later()))
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(turns, "finish", lose_ack)
        try:
            async with asyncio.timeout(8):
                with TurnObserver(runtime, "session", principal, preview=False, local_workspace=False) as observer:
                    sent = await start()
                    with pytest.raises(AIError) as stopped:
                        async for frame in observer.events(sent.key.turn_id):
                            assert frame["type"] != "terminal"
                    assert stopped.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

                assert (await turns.get_session("session"))["active_turn_id"] == sent.key.turn_id.encode()
                accepted = await turns.get_turn(sent.key)
                assert accepted["state"] == "accepted" and accepted["generation"] == 1
                for _ in range(2):
                    with pytest.raises(AIError) as unresolved:
                        await runtime.history.turn_result(
                            "session", sent.key.turn_id, principal=principal, producer_stopped_generation=1,
                        )
                    assert unresolved.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                duplicate = await start()
                assert duplicate.key == sent.key and calls == 1
                with pytest.raises(AIError) as busy:
                    await start("next", "next")
                assert busy.value.code is ErrorCode.SESSION_BUSY
                assert await turns.find_request("session", "next") is None
                assert not await runtime.sessions.recover_turn("session", sent.key.turn_id, mode="resume", principal=principal)

                release.set()
                [server_commit] = pending
                assert await server_commit
                finished = await duplicate.wait_final(timeout_seconds=5)
                assert finished.key == sent.key and finished.status is ExecutionStatus.SUCCEEDED
                assert finished.view["output"] == {"text": "answer"}
                assert (await start()).key == sent.key and calls == 1
                assert (await turns.get_session("session"))["active_turn_id"] is None
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_tool_unknown_ack_does_not_dispatch_before_or_after_late_claim(tmp_path, monkeypatch):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Provide the explicit authority for canonical session reads and writes."""

    calls = 0
    effects = []

    async def effect(context: AgentContext[None]) -> str:
        effects.append(context.execution_id)
        return "done"

    async def model(messages, _info):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert isinstance(messages[-1], ModelRequest)
            [result] = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
            assert (result.tool_name, result.tool_call_id, result.outcome) == ("effect", "effect", "failed")
            assert "Task interrupted; tool outcome unknown." in str(result.content)
            return ModelResponse(parts=[TextPart("could not confirm the effect")])
        assert calls == 1
        return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="effect")])

    features = RuntimeFeatures("native-unknown-tool")
    configure_execution(features, retain_history=True)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.tool(effect, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("effect",))
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.from_root(tmp_path / "native")
    async with (
        Persistence.open("native-unknown-tool", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=HostAuthorization()) as runtime,
    ):
        await runtime.agents.get().create_session("session")
        turns = state.session_turns
        repository = state.checkpoint.tools
        release = asyncio.Event()
        pending = []
        principal = runtime.default_principal
        original = RuntimeToolOperationBridge.prepare_boundaries

        def expose_unknown(bridge, *, admission, terminal, unknown=None):
            async def lose_ack(request, **kwargs):
                assert not pending

                async def commit_later():
                    await release.wait()
                    return await admission(request, **kwargs)

                pending.append(asyncio.create_task(commit_later()))
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            return original(bridge, admission=lose_ack, terminal=terminal, unknown=unknown)

        async def start(prompt, request_id):
            return await runtime.sessions.start("session", prompt, agent=runtime.agents.get(),
                                                principal=principal, idempotency_key=request_id)

        monkeypatch.setattr(RuntimeToolOperationBridge, "prepare_boundaries", expose_unknown)
        try:
            async with asyncio.timeout(8):
                with TurnObserver(runtime, "session", principal, preview=False, local_workspace=False) as observer:
                    sent = await start("question", "request")
                    with pytest.raises(AIError) as stopped:
                        async for frame in observer.events(sent.key.turn_id):
                            assert frame["type"] != "terminal"
                    assert stopped.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                key = sent.key
                assert await repository.list_by_execution(key.turn_id, tenant_id=runtime.tenant_id) == ()
                assert effects == [] and calls == 1
                assert (await start("question", "request")).key == key
                with pytest.raises(AIError) as busy:
                    await start("next", "next")
                assert busy.value.code is ErrorCode.SESSION_BUSY
                assert await turns.find_request("session", "next") is None

                release.set()
                [server_commit] = pending
                confirmed = (await server_commit).value
                [stored] = await repository.list_by_execution(key.turn_id, tenant_id=runtime.tenant_id)
                assert stored.tool_operation_id == confirmed.tool_operation_id
                assert stored.tool_call_id == "effect" and stored.status is ToolOperationStatus.CLAIMED
                assert stored.producer_generation == 1 and effects == []
                assert not await runtime.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)

                # Even after genuine lease expiry, a committed claim whose
                # caller lost the ACK cannot authorize a fresh effect.
                async with turns._context.engine.begin() as connection:
                    expired = await connection.execute(turns._turns.update().filter_by(
                        **turns._turn_values(key), generation=1, state="accepted",
                    ).values(lease_until=0))
                    assert expired.rowcount == 1
                assert await runtime.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)
                finished = await sent.wait_final(timeout_seconds=5)
                assert finished.key == key and finished.status is ExecutionStatus.SUCCEEDED
                assert finished.view["output"] == {"text": "could not confirm the effect"}
                assert len(pending) == 1 and effects == [] and calls == 2
                assert await repository.list_by_execution(key.turn_id, tenant_id=runtime.tenant_id) == (stored,)
                assert (await turns.get_session("session"))["active_turn_id"] is None
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)
