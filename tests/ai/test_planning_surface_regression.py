#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning dependency upgrades must not widen the LinkTools tool surface."""

import pytest
from linktools.ai.runtime._capabilities import (
    PLANNING_TOOL_NAMES,
    _RuntimePlanningCapability,
    compose_platform_capabilities,
)
from linktools.ai.runtime.state import StagingStepStore
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


async def test_linktools_planning_registers_only_write_plan() -> None:
    capabilities = await compose_platform_capabilities(
        agent_name="agent",
        conversation_id=None,
        step_run_id="run",
        segment_sequence=1,
        history_id=None,
        memory_scope=None,
        step_store=StagingStepStore(),
        memory_store=None,
        runtime_tool_names=PLANNING_TOOL_NAMES,
        plan_mode=False,
        trusted_tool_classes=(("write_plan", "control"),),
        trusted_mcp_selectors=(),
        context_target_tokens=None,
        parent_step_run_id=None,
        tool_operations=None,
        background_tasks=set(),
        plan_store_resolver=lambda _ctx: None,  # type: ignore[return-value]
    )
    planning = next(
        capability
        for capability in capabilities
        if isinstance(capability, _RuntimePlanningCapability)
    )
    assert tuple(planning.get_toolset().tools) == PLANNING_TOOL_NAMES


class _PlanView:
    owner_kind = "session"
    owner_id = "session"

    def __init__(self) -> None:
        self.reads = 0

    async def get_plan(self) -> dict[str, object]:
        self.reads += 1
        return {
            "items": [{"content": "ship it", "status": "in_progress"}],
            "revision": 3,
        }


async def test_planning_prompt_is_request_scoped_and_not_transcript_content() -> None:
    store = _PlanView()
    capability = _RuntimePlanningCapability(
        lambda _ctx: store,  # type: ignore[arg-type]
        id="planning",
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )
    await capability.before_run(context)
    original = UserPromptPart("continue")
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[ModelRequest(parts=[original])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    await capability.before_model_request(context, request_context)

    assert store.reads == 1
    assert len(request_context.messages[-1].parts) == 2
    assert isinstance(request_context.messages[-1].parts[-1], UserPromptPart)
    assert request_context.messages[-1].parts[-1].content.startswith(
        "[linktools.plan.v1]"
    )

    await capability.after_model_request(
        context,
        request_context=request_context,
        response=ModelResponse(parts=[]),
    )

    assert request_context.messages == [ModelRequest(parts=[original])]
