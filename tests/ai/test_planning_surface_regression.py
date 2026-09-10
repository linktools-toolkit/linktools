#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness Planning must preserve the intentionally narrow LinkTools surface."""

import pytest
from linktools.ai.runtime._plan import PlanItem as RuntimePlanItem, _validated_items
from linktools.ai.runtime._capabilities import (
    PLANNING_TOOL_NAMES,
    compose_platform_capabilities,
)
from linktools.ai.runtime.state._steps import (
    StagingStepStore,
)
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanItem,
    Planning,
    TaskStatus,
)

pytestmark = pytest.mark.asyncio


async def test_linktools_planning_registers_only_write_plan() -> None:
    capabilities = await compose_platform_capabilities(
        agent_name="agent",
        step_run_id="run",
        segment_sequence=1,
        history_id=None,
        memory_scope=None,
        step_store=StagingStepStore(),
        memory_store=None,
        runtime_tool_names=PLANNING_TOOL_NAMES,
        context_target_tokens=None,
        parent_step_run_id=None,
        plan_store_resolver=lambda _ctx: None,  # type: ignore[return-value]
    )
    planning = next(
        capability for capability in capabilities if isinstance(capability, Planning)
    )
    assert tuple(planning.get_toolset().tools) == PLANNING_TOOL_NAMES


async def test_harness_planning_prompt_is_request_scoped_and_cache_safe() -> None:
    store = InMemoryPlanStore()
    await store.set_items([PlanItem(content="ship it", status=TaskStatus.in_progress)])
    capability = Planning(
        store=store,
        tools=PLANNING_TOOL_NAMES,
        id="linktools-planning",
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )
    original = UserPromptPart("continue")
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[ModelRequest(parts=[original])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    captured: list[ModelRequest] = []

    async def handler(current: ModelRequestContext) -> ModelResponse:
        assert isinstance(current.messages[-1], ModelRequest)
        captured.append(current.messages[-1])
        return ModelResponse(parts=[])

    await capability.wrap_model_request(
        context,
        request_context=request_context,
        handler=handler,
    )

    assert len(captured) == 1
    reminder = captured[0].parts[-1]
    assert isinstance(reminder, UserPromptPart)
    assert not isinstance(reminder.content, str)
    assert any(isinstance(item, CachePoint) for item in reminder.content)
    assert request_context.messages[0].parts[0] is original


async def test_runtime_plan_persistence_adds_no_arbitrary_size_limit() -> None:
    items = [RuntimePlanItem("x" * 600) for _ in range(129)]

    assert len(_validated_items(items)) == 129
