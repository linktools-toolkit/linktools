#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The writable SQL SDK admits, observes, and reads one native turn."""

import asyncio

import pytest
from contextlib import aclosing

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

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

    features = RuntimeFeatures("sdk-native-session", defaults={"execution": {"retain_history": True}})
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
            await agent.session("absent").start("must not create a legacy session", principal=principal,
                              idempotency_key="missing")
        assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert state.conversation.sessions is None
        with pytest.raises(AIError) as missing:
            await runtime.sessions.get("absent", principal=principal)
        assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        native_created = await agent.create_session("agent-created", principal=principal)
        assert (await native_created.inspect())["session_id"] == "agent-created"
        assert state.conversation.sessions is None
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
        direct = await runtime.sessions.start_turn("chat", "Hello", principal=principal, agent=agent, idempotency_key="send")
        assert direct.key == turn.key
        async with asyncio.timeout(10):
            waited = await direct.wait()
        assert waited.key == turn.key and waited.usage == result.usage
        assert calls == [True]
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=principal.tenant_id) is None


        async with asyncio.timeout(10):
            plan = await runtime.sessions.start_turn(created.session_id, "Plan next", agent=agent, principal=principal, idempotency_key="plan", mode="plan", planning=True)
            planned = await plan.wait()
        assert planned.key.session_id == created.session_id and planned.key != turn.key
        assert planned.status is ExecutionStatus.SUCCEEDED
        assert calls == [True, True]
        assert await state.execution.executions.get(planned.key.turn_id, tenant_id=principal.tenant_id) is None

        async with asyncio.timeout(10):
            started = await agent.session(created.session_id).start("third", principal=principal,
                                        idempotency_key="start-native")
            assert (await started.wait_final()).status is ExecutionStatus.SUCCEEDED
            run = await agent.session(created.session_id).run("fourth", principal=principal,
                                  idempotency_key="run-native")
            plan_run = await agent.session(created.session_id).plan("fifth", principal=principal,
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
    features = RuntimeFeatures(runtime_name, defaults={"execution": {"retain_history": True}})
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
            lambda: agent.session("chat").start("must not create a legacy session"),
        ):
            with pytest.raises(AIError) as missing:
                await operation()
            assert missing.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert state.conversation.sessions is None
        standalone = await agent.start("no session requested")
        assert (await standalone.wait(timeout_seconds=5)).result.output == {"text": "standalone"}


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["bound-sdk-agent", "agent-id"])
async def test_native_sdk_freezes_effective_agent_planning_and_thinking(tmp_path, role):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Authorize native sessions explicitly."""

    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("done")])

    def features(*, revision, planning, thinking):
        result = RuntimeFeatures("sdk-native-defaults", defaults={"execution": {"retain_history": True}})
        result.agent("default", revision=revision, allow_tools=(), allow_capabilities=(), planning=planning, thinking=thinking)
        return model_features(_Models(_UsageFunctionModel(model)), result)

    storage = PersistenceStorage.from_root(tmp_path)
    async with (
        Persistence.open("sdk-native-defaults", storage=storage) as state,
        Runtime.open(persistence=state, authorization=HostAuthorization(),
                     features=features(revision=1, planning=True, thinking="high")) as runtime,
    ):
        principal = runtime.default_principal
        agent = runtime.agents.get()
        await agent.create_session("session")
        selected = agent if role == "bound-sdk-agent" else "default"

        async def send(key, **options):
            return await runtime.sessions.start_turn(
                "session", "question", agent=selected, principal=principal, idempotency_key=key, **options,
            )

        admitted = (await agent.session("session").start("question", idempotency_key="turn")
                    if role == "bound-sdk-agent" else await send("turn"))
        assert (await admitted.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        row = await state.session_turns.get_turn(admitted.key)
        assert row["request_json"]["planning"] is True and row["request_json"]["thinking"] == "high"
        assert (await send("turn", planning=True, thinking="high")).key == admitted.key
        assert (await send("turn")).key == admitted.key
        assert calls == 1

        for options in ({"planning": False}, {"thinking": False}, {"thinking": "low"}):
            with pytest.raises(AIError) as mismatch:
                await send("turn", **options)
            assert mismatch.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert calls == 1

        overriding = await send("override", planning=False, thinking=False)
        assert (await overriding.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        row = await state.session_turns.get_turn(overriding.key)
        assert row["request_json"].get("planning", False) is False and row["request_json"]["thinking"] is False
        with pytest.raises(AIError) as mismatch:
            await send("override")
        assert mismatch.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert calls == 2

    if role == "agent-id":
        async with (
            Persistence.open("sdk-native-defaults", storage=storage) as state,
            Runtime.open(persistence=state, authorization=HostAuthorization(),
                         features=features(revision=2, planning=False, thinking="low")) as runtime,
        ):
            same = await runtime.sessions.start_turn(
                "session", "question", agent="default", principal=runtime.default_principal, idempotency_key="turn",
            )
            assert same.key == admitted.key
            assert (await same.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert calls == 2


@pytest.mark.asyncio
async def test_agent_bound_native_session_mutations_use_canonical_revisions(tmp_path):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Authorize the canonical SQL session on the Runtime's behalf."""

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("sdk-native-mutations", defaults={"execution": {"retain_history": True}})
    features.agent("default", allow_tools=(), allow_capabilities=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open("sdk-native-mutations", storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=HostAuthorization()) as runtime,
    ):
        agent = runtime.agents.get()
        await agent.create_session("source")
        bound = agent.session("source")
        admitted = await bound.start("original", idempotency_key="send")
        assert (await admitted.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        revision = (await bound.inspect())["revision"]

        assert not await bound.update(expected_revision=revision - 1, metadata={"wrong": True})
        assert await bound.update(expected_revision=revision, metadata={"title": "Updated"})
        assert (await bound.inspect())["metadata"] == {"title": "Updated"}
        assert await bound.fork("stale", expected_revision=revision, idempotency_key="stale") is None
        source_revision = (await bound.inspect())["revision"]
        child = await bound.fork("child", expected_revision=source_revision, idempotency_key="fork")
        assert child is not None and child.session_id == "child" and child.agent_id == bound.agent_id
        repeated = await bound.fork("child", expected_revision=source_revision, idempotency_key="fork")
        assert repeated is not None and repeated.session_id == child.session_id
        timeline = await child.timeline()
        assert [turn["turn_id"] for turn in timeline["items"]] == [admitted.key.turn_id]
        assert (await child.inspect())["metadata"] == {"title": "Updated"}

        child_revision = (await child.inspect())["revision"]
        assert await child.update(expected_revision=child_revision, metadata={"title": "Child"})
        assert not await child.close(expected_revision=child_revision)
        assert await child.close(expected_revision=child_revision + 1)
        assert (await child.inspect())["status"] == "CLOSED"
        assert (await bound.inspect())["status"] == "OPEN"
        for session_id in ("source", "child", "stale"):
            assert state.conversation.sessions is None

        with pytest.raises(AIError) as invalid:
            await bound.close(expected_revision=source_revision + 1, force=True)
        assert invalid.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert invalid.value.safe_details["field"] == "force"
