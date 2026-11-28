#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native steps schedule ordinary tools using their frozen sequential declaration."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, SessionCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("sequential", (False, True))
async def test_native_step_tools_honor_frozen_sequential_declaration(tmp_path, sequential):
    class Policy(TenantAuthorizationPolicy):
        pass

    entered_first, entered_second, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    order = []

    async def write(_context: AgentContext[None], value: int) -> str:
        order.append(("start", value))
        (entered_first if value == 1 else entered_second).set()
        await release.wait()
        order.append(("finish", value))
        return str(value)

    async def model(messages, info):
        tool, = info.function_tools
        assert tool.name == "write" and tool.sequential is sequential
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("write", {"value": 1}, "first-call"),
            ToolCallPart("write", {"value": 2}, "second-call"),
        ])

    features = RuntimeFeatures("native-sequential")
    adapted = features.tool(write, **({"sequential": True} if sequential else {}))
    assert adapted.tool_def.sequential is sequential
    definition, = (entry for entry in (await features.resolve()).contributions if entry.kind == "tool")
    assert definition.contract.get("sequential", False) is sequential
    # The original mutable Tool cannot change the captured declaration.
    adapted.sequential = not sequential
    assert definition.materialize_tool().tool_def.sequential is sequential
    features.agent("default", allow_tools=("write",))
    model_features(_Models(_UsageFunctionModel(model)), features)

    async with Persistence.open(
        "native-sequential", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
    ) as state:
        features.capability(SessionCapability(checkpoints=True), scope="runtime")
        async with Runtime.open(
            persistence=state, features=features, authorization=Policy(),
        ) as runtime, asyncio.timeout(10):
            actor = runtime.default_principal
            await runtime.sessions.create_turn_session("session", principal=actor)
            turn = await runtime.sessions.start(
                "session", "write", agent="default", principal=actor, idempotency_key="one",
            )
            try:
                await entered_first.wait()
                if not sequential:
                    await entered_second.wait()
            finally:
                release.set()
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
    if sequential:
        assert order == [("start", 1), ("finish", 1), ("start", 2), ("finish", 2)]
    else:
        assert sorted(order[:2]) == [("start", 1), ("start", 2)]
        assert sorted(order[2:]) == [("finish", 1), ("finish", 2)]
