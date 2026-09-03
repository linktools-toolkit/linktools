#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill tool regression for model-correctable selection misses."""

import pytest
from linktools.ai.capability import SkillCapability, SkillDefinition, SkillSourceRegistry
from linktools.ai.runtime._skill_adapter import _PydanticSkillCapability
from linktools.ai.spec import SkillSpec
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


async def test_missing_skill_id_is_model_retry() -> None:
    capability = _PydanticSkillCapability(
        SkillCapability(
            (SkillDefinition(SkillSpec("known", content="instructions")),),
            SkillSourceRegistry(),
        )
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ModelRetry, match="skill not found"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "missing"},
            context,
            tools["load_skill"],
        )


async def test_missing_skill_resource_is_tool_failure() -> None:
    capability = _PydanticSkillCapability(
        SkillCapability(
            (SkillDefinition(SkillSpec("known", content="instructions")),),
            SkillSourceRegistry(),
        )
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolFailed, match="skill resource not found"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "missing.txt"},
            context,
            tools["load_skill"],
        )
