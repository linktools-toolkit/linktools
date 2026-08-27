#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning dependency upgrades must not widen the LinkTools tool surface."""

import pytest
from linktools.ai.runtime._capabilities import (
    PLANNING_TOOL_NAMES,
    compose_platform_capabilities,
)
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import InMemoryStepStore

pytestmark = pytest.mark.asyncio


async def test_linktools_planning_registers_only_write_plan() -> None:
    capabilities = await compose_platform_capabilities(
        agent_name="agent",
        conversation_id=None,
        step_run_id="run",
        segment_sequence=1,
        history_id=None,
        memory_scope=None,
        step_store=InMemoryStepStore(),
        memory_store=None,
        runtime_tool_names=PLANNING_TOOL_NAMES,
        plan_mode=False,
        trusted_tool_classes=(("write_plan", "control"),),
        trusted_mcp_selectors=(),
        context_target_tokens=None,
        parent_step_run_id=None,
        subagent_delegate=None,
        tool_operations=None,
        background_tasks=set(),
        plan_store_resolver=lambda _ctx: None,  # type: ignore[return-value]
    )
    planning = next(
        capability
        for capability in capabilities
        if isinstance(capability, Planning)
    )
    assert tuple(planning.tools or ()) == PLANNING_TOOL_NAMES
