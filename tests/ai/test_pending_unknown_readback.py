#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime.turn_clients import TurnObserver
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_terminal_payload_runtime import _PayloadBuffer


@pytest.mark.asyncio
async def test_native_terminal_unknown_keeps_slot_until_exact_late_commit(tmp_path, monkeypatch):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Provide the explicit authority for canonical session reads and writes."""

    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("native-unknown-terminal")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
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
                assert not await runtime.sessions.recover_turn("session", sent.key.turn_id, principal=principal)

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
async def test_pending_unknown_absent_readback_keeps_admission_until_late_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    effects = []

    async def effect(context: AgentContext[None]) -> str:
        effects.append(context.execution_id)
        return "done"

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[ToolCallPart('effect', {}, tool_call_id='effect')])

    buffer = _PayloadBuffer()
    features = RuntimeFeatures('pending-unknown')
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal(buffer=buffer)), scope='runtime')
    features.tool(effect, effect_policy='non_replay_safe')
    features.agent('default', model='default', allow_tools=('effect',))
    storage = PersistenceStorage.sqlite(
        tmp_path / 'state.db',
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Persistence.open('pending-unknown', storage=storage) as persistence:
        context = Runtime.open(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features))
        runtime = await context.__aenter__()
        release = asyncio.Event()
        dispatched = asyncio.Event()
        pending = []
        try:
            session = await runtime.agents.get().create_session('session')
            group = runtime._state.execution.executions.state_store.storage_group
            original = type(group).mutate

            async def delayed(owner, stores, callback):
                if owner is group and not dispatched.is_set():
                    async def server_commit():
                        await release.wait()
                        return await original(owner, stores, callback)
                    pending.append(asyncio.create_task(server_commit()))
                    dispatched.set()
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return await original(owner, stores, callback)

            monkeypatch.setattr(type(group), 'mutate', delayed)
            execution = await session.start('question', idempotency_key='request')
            await asyncio.wait_for(dispatched.wait(), 5)
            backend = runtime._execution_service.runtime_backend()
            turn = backend._pending_turns.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert turn is not None
            with pytest.raises(AIError) as unknown:
                await execution.wait(timeout_seconds=5)
            assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert turn.uncertain
            original_failure = turn.failure
            for _ in range(2):
                with pytest.raises(AIError) as repeated:
                    await execution.wait(timeout_seconds=5)
                assert repeated.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                assert turn.uncertain and turn.failure is original_failure
            assert len(pending) == 1
            assert buffer.accepted and buffer.acknowledged == []
            captured, ids = await backend._execution_buffer.read(execution.execution_id)
            assert ids and captured.execution.execution_id == execution.execution_id
            assert backend._pending_turns.peek_request(
                tenant_id=runtime.tenant_id, scope=turn.reservation.idempotency.scope,
                idempotency_key=turn.reservation.idempotency.idempotency_key,
            ) is turn
            with pytest.raises(AIError) as busy:
                await session.start('next', idempotency_key='next')
            assert busy.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert len(calls) == 1
            release.set()
            await asyncio.gather(*pending)
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.FAILED
            assert effects == []
            assert (await session.start('question', idempotency_key='request')).execution_id == execution.execution_id
            assert len(calls) == 1
        finally:
            release.set()
            await asyncio.gather(*pending)
            await context.__aexit__(None, None, None)
        assert set(buffer.acknowledged) == set(buffer.accepted)
