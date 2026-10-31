#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_terminal_payload_runtime import _PayloadBuffer


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ('terminal', 'promotion'))
async def test_pending_unknown_absent_readback_keeps_admission_until_late_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    calls = []
    effects = []

    async def effect(context: AgentContext[None]) -> str:
        effects.append(context.execution_id)
        return "done"

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[ToolCallPart('effect', {}, tool_call_id='effect')] if boundary == 'promotion' else [TextPart('answer')])

    buffer = _PayloadBuffer()
    features = RuntimeFeatures('pending-unknown')
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal(buffer=buffer)), scope='runtime')
    features.tool(effect, effect_policy='replay_safe')
    features.agent('default', model='default', allow_tools=('effect',))
    storage = PersistenceStorage.sqlite(
        tmp_path / 'state.db',
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Persistence.open('pending-unknown', storage=storage) as persistence:
        context = Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features))
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
                idempotency_key_digest=turn.reservation.idempotency.idempotency_key_digest,
            ) is turn
            with pytest.raises(AIError) as busy:
                await session.start('next', idempotency_key='next')
            assert busy.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert len(calls) == 1
            release.set()
            await asyncio.gather(*pending)
            expected = ExecutionStatus.SUCCEEDED if boundary == 'terminal' else ExecutionStatus.FAILED
            assert (await execution.wait(timeout_seconds=5)).result.status is expected
            assert effects == []
            assert (await session.start('question', idempotency_key='request')).execution_id == execution.execution_id
            assert len(calls) == 1
        finally:
            release.set()
            await asyncio.gather(*pending)
            await context.__aexit__(None, None, None)
        assert set(buffer.acknowledged) == set(buffer.accepted)
