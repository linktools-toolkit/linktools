#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill tool regression for retry and execution-failure semantics."""

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


def _skill_toolset():
    capability = _PydanticSkillCapability(
        SkillCapability(
            (SkillDefinition(SkillSpec("known", content="instructions")),),
            SkillSourceRegistry(),
        )
    )
    return capability.get_toolset()


async def test_missing_skill_id_is_model_retry() -> None:
    toolset = _skill_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ModelRetry, match="skill not found"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "missing"},
            context,
            tools["load_skill"],
        )


async def test_invalid_skill_resource_path_is_model_retry() -> None:
    toolset = _skill_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ModelRetry, match="skill resource path is invalid"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "../outside"},
            context,
            tools["load_skill"],
        )


async def test_missing_skill_resource_is_tool_failure() -> None:
    toolset = _skill_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolFailed, match="skill resource not found"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "missing.txt"},
            context,
            tools["load_skill"],
        )
