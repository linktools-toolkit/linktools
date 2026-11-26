#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The writable SQL SDK admits, observes, and reads one native turn."""

import asyncio

import pytest
from contextlib import aclosing

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import AuthorizationAction, ExecutionStatus, TenantAuthorizationPolicy, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
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
        exactly_64 = "案" * 21 + "a"
        assert CreateSessionRequest(principal, exactly_64, "create").session_id == exactly_64
        with pytest.raises(AIError) as too_long:
            CreateSessionRequest(principal, exactly_64 + "x", "create")
        assert too_long.value.code is ErrorCode.REQUEST_FIELD_INVALID
        with pytest.raises(AIError) as idempotency:
            await runtime.sessions.create_turn_session("missing", principal=principal, idempotency_key=exactly_64 + "x")
        assert idempotency.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID
        created = await runtime.sessions.create(CreateSessionRequest(principal, "chat", "create-chat"))
        opened = await runtime.sessions.get(created.session_id, principal=principal)
        snapshot = await opened.inspect()
        assert snapshot["session_id"] == "chat" and snapshot["active_turn_id"] is None
        assert "agent_id" not in snapshot and "active_execution_id" not in snapshot

        agent = runtime.agents.get()
        with pytest.raises(AIError) as missing:
            await agent.start("must not create a legacy session", principal=principal,
                              session_id="absent", idempotency_key="missing")
        assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await state.conversation.sessions.get("absent", tenant_id=principal.tenant_id) is None
        with pytest.raises(AIError) as missing:
            await runtime.sessions.get("absent", principal=principal)
        assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        native_created = await agent.create_session("agent-created", principal=principal)
        assert (await native_created.inspect())["session_id"] == "agent-created"
        assert await state.conversation.sessions.get("agent-created", tenant_id=principal.tenant_id) is None
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
        direct = await runtime.sessions.start("chat", "Hello", principal=principal, agent=agent, idempotency_key="send")
        assert direct.key == turn.key
        async with asyncio.timeout(10):
            waited = await direct.wait()
        assert waited.key == turn.key and waited.usage == result.usage
        assert calls == [True]
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=principal.tenant_id) is None


        async with asyncio.timeout(10):
            plan = await agent.start_plan("Plan next", principal=principal, session_id=created.session_id,
                                          idempotency_key="plan")
            planned = await plan.wait()
        assert planned.key.session_id == created.session_id and planned.key != turn.key
        assert planned.status is ExecutionStatus.SUCCEEDED
        assert calls == [True, True]
        assert await state.execution.executions.get(planned.key.turn_id, tenant_id=principal.tenant_id) is None

        async with asyncio.timeout(10):
            started = await agent.start("third", principal=principal, session_id=created.session_id,
                                        idempotency_key="start-native")
            assert (await started.wait_final()).status is ExecutionStatus.SUCCEEDED
            run = await agent.run("fourth", principal=principal, session_id=created.session_id,
                                  idempotency_key="run-native")
            plan_run = await agent.plan("fifth", principal=principal, session_id=created.session_id,
                                        idempotency_key="plan-native")
        assert run.cursor is plan_run.cursor is None
        assert run.result.key.session_id == plan_run.result.key.session_id == created.session_id
        assert run.result.status is plan_run.result.status is ExecutionStatus.SUCCEEDED
        assert calls == [True] * 5

        lazy = agent.session(created.session_id, principal=principal)
        assert (await lazy.inspect())["session_id"] == created.session_id
        assert (await lazy.history())["timeline"] == await lazy.timeline()
        async with aclosing(lazy.watch()) as occupancy:
            assert (await anext(occupancy)).active_turn_id is None


@pytest.mark.asyncio
async def test_sql_host_without_session_authorization_does_not_fall_back(tmp_path):
    runtime_name = "sdk-sql-no-session-auth"
    features = RuntimeFeatures(runtime_name)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", allow_tools=(), allow_capabilities=())

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("standalone")])

    model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open(runtime_name, storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features) as runtime,
    ):
        principal = runtime.default_principal
        agent = runtime.agents.get()
        for operation in (
            lambda: agent.create_session("chat"),
            lambda: runtime.sessions.create(CreateSessionRequest(principal, "chat", "create")),
            lambda: runtime.sessions.get("chat", principal=principal),
            lambda: runtime.sessions.inspect("chat", principal=principal),
            lambda: agent.start("must not create a legacy session", session_id="chat"),
        ):
            with pytest.raises(AIError) as missing:
                await operation()
            assert missing.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert await state.conversation.sessions.get("chat", tenant_id=principal.tenant_id) is None
        standalone = await agent.start("no session requested")
        assert (await standalone.wait(timeout_seconds=5)).result.output == {"text": "standalone"}
