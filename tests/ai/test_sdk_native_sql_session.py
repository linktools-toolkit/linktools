#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The writable SQL SDK admits, observes, and reads one native turn."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import AuthorizationAction, ExecutionStatus, TenantAuthorizationPolicy, UsageMetrics
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import CreateSessionRequest

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
async def test_public_sql_sdk_session_creates_and_replays_native_turn(tmp_path):
    class Authorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if action is AuthorizationAction.EXECUTION_READ:
                reads.append(resource)

    calls = []
    reads = []

    async def model(_messages, _info):
        calls.append(True)
        return ModelResponse(parts=[TextPart("native")], usage=RequestUsage(input_tokens=7, output_tokens=11))

    features = RuntimeFeatures("sdk-native-session")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", allow_tools=(), allow_capabilities=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open("sdk-native-session", storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=Authorization()) as runtime,
    ):
        principal = runtime.default_principal
        created = await runtime.sessions.create(CreateSessionRequest(principal, "chat", "create-chat"))
        opened = await runtime.sessions.get(created.session_id, principal=principal)
        snapshot = await opened.inspect()
        assert snapshot["session_id"] == "chat" and snapshot["active_turn_id"] is None
        assert "agent_id" not in snapshot and "active_execution_id" not in snapshot

        agent = runtime.agents.get()
        async with asyncio.timeout(10), opened.start("Hello", agent=agent, idempotency_key="send") as (turn, frames):
            assert turn.key.session_id == "chat"
            terminal = [frame async for frame in frames][-1]
            assert terminal["type"] == "terminal" and terminal["state"] == "succeeded"
            result = await turn.result()
        assert result.key == turn.key and result.status is ExecutionStatus.SUCCEEDED
        assert result.usage == UsageMetrics(model_requests=1, input_tokens=7, output_tokens=11)
        assert result.view["output"] == {"text": "native"}
        assert reads[-1].id == turn.key.turn_id and reads[-1].owner_principal_id == principal.principal_id

        async with asyncio.timeout(10), opened.start("Hello", agent=agent, idempotency_key="send") as (same, frames):
            assert same.key == turn.key
            [terminal] = [frame async for frame in frames]
            assert terminal["type"] == "terminal" and terminal["state"] == "succeeded"
        assert calls == [True]
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=principal.tenant_id) is None
