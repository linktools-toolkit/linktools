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
    ToolCallRetry,
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


class _MissingRootSource:
    @property
    def id(self) -> str:
        return "source"

    async def inspect(self, root: str) -> SkillResourceView:
        del root
        raise AIError(ErrorCode.ASSET_NOT_FOUND)

    async def read(self, root: str, path: str) -> bytes:
        del root, path
        raise AssertionError("read should not be called")


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

    with pytest.raises(ToolCallRetry, match="skill id is not available"):
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

    with pytest.raises(ToolCallFailed, match="resource does not exist"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "missing.txt"},
            context,
            tools["load_skill"],
        )


async def test_missing_skill_root_does_not_suggest_retrying_the_same_call() -> None:
    definition = SkillDefinition(
        SkillSpec("known", content="instructions"),
        SkillSourceRef("source", "known"),
    )
    capability = SkillCapability(
        (definition,),
        SkillSourceRegistry((_MissingRootSource(),)),
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallFailed) as raised:
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known"},
            context,
            tools["load_skill"],
        )

    assert "resource root is unavailable" in raised.value.message
    assert "same load_skill call will not resolve it" in raised.value.message


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
    with pytest.raises(ToolCallFailed, match="outside the skill root"):
        await toolset.call_tool(
            "load_skill",
            {"skill_id": "known", "path": "resource.txt"},
            context,
            tools["load_skill"],
        )
