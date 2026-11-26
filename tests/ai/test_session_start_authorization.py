#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session starts authorize execution resources, not session IDs as executions."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import AuthorizationAction, CommitPolicy, ExecutionStatus, HmacCursorSigner, Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import ResumeSessionRequest, SessionManager
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_session_actor_identity import SharedSessionPolicy


class ExecutionResourcePolicy(SharedSessionPolicy):
    """Allow a shared-session actor, but reject a session used as an execution."""

    def __init__(self) -> None:
        super().__init__()
        self.execution_resources: list[str] = []
        self.denied: set[str] = set()
        self.deny_new = False
        self.existing_owners: dict[str, str] = {}

    async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        if action is AuthorizationAction.EXECUTION_RUN:
            self.execution_resources.append(resource.id)
            if (resource.id in self.existing_owners
                    and resource.owner_principal_id != self.existing_owners[resource.id]):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if (resource.kind is not ResourceKind.EXECUTION or resource.id == "shared"
                    or self.deny_new or resource.id in self.denied):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await super().authorize(principal, action, resource)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", (False, True))
async def test_shared_session_start_authorizes_the_execution_identity(tmp_path: Path, terminal: bool) -> None:
    calls = 0
    entered, finish = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("session-start-authorization")
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()), scope="runtime")
    cache = InMemoryCacheBackend(max_bytes=2**24)
    policy = ExecutionResourcePolicy()
    owner, peer = Principal("owner", "default"), Principal("peer", "default")
    async with Persistence.open(
        "session-start-authorization", storage=PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache),
    ) as state, Runtime.open(
        persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
        authorization=policy,
    ) as runtime:
        agent = runtime.agents.get()
        session = await agent.create_session("shared", principal=owner)

        async def start(prompt: str, key: str):
            return await runtime.sessions.start(session.session_id, prompt, principal=peer, agent=agent,
                                                idempotency_key=key)
        policy.deny_new = True
        with pytest.raises(AIError) as denied:
            await start("denied fresh", "denied-fresh")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert calls == 0
        policy.deny_new = False
        policy.execution_resources.clear()
        execution = await start("peer turn", "peer-turn")
        policy.existing_owners[execution.key.turn_id] = "peer"
        try:
            await asyncio.wait_for(entered.wait(), 5)
            replay = await start("peer turn", "peer-turn")
            assert replay.key.turn_id == execution.key.turn_id
            policy.denied.add(execution.key.turn_id)
            with pytest.raises(AIError) as denied:
                await start("peer turn", "peer-turn")
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert calls == 1
        finally:
            finish.set()
        assert (await execution.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        with pytest.raises(AIError) as denied:
            await start("peer turn", "peer-turn")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        policy.denied.clear()
        replay = await start("peer turn", "peer-turn")
        assert replay.key.turn_id == execution.key.turn_id
        assert calls == 1
        assert policy.execution_resources
        assert policy.execution_resources[-1] == execution.key.turn_id
        assert "shared" not in policy.execution_resources
        assert await state.conversation.sessions.get("shared", tenant_id="default") is None
        assert await state.execution.executions.get(execution.key.turn_id, tenant_id="default") is None
        row = await state.session_turns.get_turn(execution.key)
        assert row["request_json"]["principal"]["principal_id"] == "peer"


@pytest.mark.asyncio
async def test_generic_session_execution_adapter_retains_its_admission_guard() -> None:
    from .test_session_timeline import _session

    policy = ExecutionResourcePolicy()
    policy.deny_new = True
    async with Persistence.open(
        "generic-session-authorization", storage=PersistenceStorage.in_memory(), tenant_id="tenant",
    ) as state:
        await state.conversation.sessions.create(_session())
        manager = SessionManager(
            state.conversation, state.execution.executions, policy, object(),
            HmacCursorSigner("session", b"session-authorization-key"),
        )
        with pytest.raises(AIError) as denied:
            await manager.resume("default", "binding", "session", ResumeSessionRequest(
                Principal("owner", "tenant"), "question", "request", None, "run", False, False,
            ))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert policy.execution_resources == ["session"]
