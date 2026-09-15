#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill tool behavior for model-correctable selection misses."""

import pytest
from linktools.ai.capability import (
    SkillCapability,
    SkillDefinition,
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
    ToolCallFailed,
    ToolCallRejected,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import SkillSpec
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


class _OutsideRootSource:
    @property
    def id(self) -> str:
        return "source"

    async def inspect(self, root: str) -> SkillResourceView:
        del root
        raise AssertionError("inspect should not be called")

    async def read(self, root: str, path: str) -> bytes:
        del root, path
        raise AIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT)


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


async def test_missing_skill_id_is_model_retry() -> None:
    capability = SkillCapability(
        (SkillDefinition(SkillSpec("known", content="instructions")),),
        SkillSourceRegistry(),
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallRejected, match="skill id or resource path is invalid"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "missing"},
            context,
            tools["load_skill"],
        )


async def test_missing_skill_resource_is_tool_failure() -> None:
    capability = SkillCapability(
        (SkillDefinition(SkillSpec("known", content="instructions")),),
        SkillSourceRegistry(),
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallFailed, match="skill resource is unavailable"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "missing.txt"},
            context,
            tools["load_skill"],
        )


async def test_outside_root_skill_resource_is_only_tool_failure_at_model_boundary() -> None:
    definition = SkillDefinition(
        SkillSpec("known", content="instructions"),
        SkillSourceRef("source", "known"),
    )
    capability = SkillCapability(
        (definition,),
        SkillSourceRegistry((_OutsideRootSource(),)),
    )

    with pytest.raises(AIError) as direct_error:
        await capability.load_skill("known", "resource.txt")
    assert direct_error.value.code is ErrorCode.ASSET_PATH_OUTSIDE_ROOT

    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)
    with pytest.raises(ToolCallFailed, match="skill resource is unavailable"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "resource.txt"},
            context,
            tools["load_skill"],
        )
