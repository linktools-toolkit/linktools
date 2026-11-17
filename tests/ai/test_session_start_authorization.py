#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session starts authorize execution resources, not session IDs as executions."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import AuthorizationAction, ExecutionStatus, Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_session_actor_identity import SharedSessionPolicy


class ExecutionResourcePolicy(SharedSessionPolicy):
    """Allow a shared-session actor, but reject a session used as an execution."""

    def __init__(self) -> None:
        super().__init__()
        self.execution_resources: list[str] = []

    async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        if action is AuthorizationAction.EXECUTION_RUN:
            self.execution_resources.append(resource.id)
            if resource.kind is not ResourceKind.EXECUTION or resource.id == "shared":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await super().authorize(principal, action, resource)


@pytest.mark.asyncio
async def test_shared_session_start_authorizes_the_execution_identity(tmp_path: Path) -> None:
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("session-start-authorization")
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    policy = ExecutionResourcePolicy()
    owner, peer = Principal("owner", "default"), Principal("peer", "default")
    async with Persistence.open(
        "session-start-authorization", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
    ) as state, Runtime.attach(
        persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
        authorization=policy,
    ) as runtime:
        session = await runtime.agents.get().create_session("shared", principal=owner)
        execution = await session.start("peer turn", principal=peer, idempotency_key="peer-turn")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert policy.execution_resources
        assert set(policy.execution_resources) == {execution.execution_id}
        assert (await state.conversation.sessions.get("shared", tenant_id="default")).owner_principal_id == "owner"
        assert (await state.execution.executions.get(execution.execution_id, tenant_id="default")).principal_id == "peer"
