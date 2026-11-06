#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning reads and writes stay with the current run's resolved store."""

from copy import deepcopy

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.planning import InMemoryPlanStore, PlanItem

from linktools.ai.capability.planning import HarnessPlanning
from linktools.ai.capability import PydanticToolControlCapability

pytestmark = pytest.mark.asyncio


def _writing_model(content: str, captured: list) -> FunctionModel:
    called = False

    async def respond(messages, info):
        nonlocal called
        captured.append(deepcopy(messages))
        if called:
            return ModelResponse(parts=[TextPart("done")])
        called = True
        return ModelResponse(parts=[ToolCallPart(
            "write_plan",
            {"items": [{"id": "step", "content": content, "status": "pending"}]},
            tool_call_id="write-current-plan",
        )])

    return FunctionModel(respond)


async def test_planning_capability_reuse_reads_and_writes_current_run_store() -> None:
    stores = {name: InMemoryPlanStore() for name in ("first", "second")}
    await stores["first"].set_items([PlanItem(content="first seed marker")])
    await stores["second"].set_items([PlanItem(content="second seed marker")])
    capability = HarnessPlanning.bind(lambda ctx: stores[str(ctx.prompt)])
    agent = Agent(TestModel(), capabilities=[capability])
    for name in ("first", "second"):
        captured = []
        with agent.override(model=_writing_model(f"{name} result marker", captured)):
            await agent.run(name)
        assert f"{name} seed marker" in str(captured[0])
        other = "second" if name == "first" else "first"
        assert f"{other} seed marker" not in str(captured[0])
        assert f"{other} result marker" not in str(captured[0])
        assert [item.content for item in await stores[name].get_items()] == [f"{name} result marker"]

    assert [item.content for item in await stores["first"].get_items()] == ["first result marker"]


async def test_invalid_plan_retries_before_changing_store() -> None:
    store = InMemoryPlanStore()
    await store.set_items([PlanItem(content="original plan")])
    calls = 0

    async def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart(
                "write_plan", {"items": [
                    {"id": "same", "content": "first invalid item"},
                    {"id": "same", "content": "second invalid item"},
                ]}, tool_call_id="invalid-plan",
            )])
        if calls == 2:
            assert [item.content for item in await store.get_items()] == ["original plan"]
            assert any(
                isinstance(part, (RetryPromptPart, ToolReturnPart))
                and part.tool_call_id == "invalid-plan"
                and "ids must be unique" in str(part.content)
                for message in messages for part in message.parts
            )
            return ModelResponse(parts=[ToolCallPart(
                "write_plan", {"items": [{"id": "valid", "content": "corrected plan"}]},
                tool_call_id="valid-plan",
            )])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), capabilities=[
        PydanticToolControlCapability(), HarnessPlanning.bind(lambda _ctx: store),
    ])
    await agent.run("plan the work")
    assert [item.content for item in await store.get_items()] == ["corrected plan"]
