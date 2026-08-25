#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning dependency upgrades must not widen the Linktools tool surface."""

from pathlib import Path
from typing import Any

import pytest
from linktools.ai.agent._capabilities import (
    PLANNING_TOOL_NAMES,
    AgentRunScope,
    compose_platform_capabilities,
)
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import InMemoryStepStore

pytestmark = pytest.mark.asyncio


async def test_linktools_planning_registers_only_write_plan(tmp_path: Path) -> None:
    scope = AgentRunScope(
        root=tmp_path,
        agent_name="agent",
        conversation_id=None,
        step_run_id="run",
        segment_sequence=1,
        memory_scope=None,
        step_store=InMemoryStepStore(),
        memory_store=None,
        platform_tool_names=PLANNING_TOOL_NAMES,
        planning=True,
    )
    capabilities = await compose_platform_capabilities(
        scope,
        model_factory=lambda value: value or "test",
        parent_model="test",
    )
    planning = next(
        capability
        for capability in capabilities
        if isinstance(capability, Planning)
    )
    assert tuple(planning.tools or ()) == PLANNING_TOOL_NAMES
