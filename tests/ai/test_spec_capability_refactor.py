#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final declaration, capability, and runtime-leaf contracts."""

import json

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import (
    LinkToolsSkills,
    SkillDefinition,
    SkillSourceRegistry,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._capabilities import select_runtime_tool_names
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)


def test_runtime_tool_selection_keeps_framework_tools_outside_allow_tools() -> None:
    assert select_runtime_tool_names(
        ordinary_tool_policy=("write_plan",),
        memory_scope="memory",
    ) == ()
    assert select_runtime_tool_names(
        ordinary_tool_policy=(),
        memory_scope=None,
        planning=True,
    ) == ("write_plan",)
    assert select_runtime_tool_names(
        ordinary_tool_policy=(),
        memory_scope=None,
        subagent_available=True,
    ) == ("delegate_task", "list_subagents")


def test_runtime_tool_selection_honors_memory_wildcard() -> None:
    assert select_runtime_tool_names(
        ordinary_tool_policy=("*",),
        memory_scope="memory",
    ) == ("delete_memory", "read_memory", "search_memory", "write_memory")


def test_agent_spec_codec_rejects_invalid_v1_payload() -> None:
    with pytest.raises(AIError) as error:
        AgentSpecCodec().decode(
            json.dumps(
                {
                    "version": 1,
                    "id": "agent",
                    "model": "model",
                    "planning": "yes",
                }
            ).encode()
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_declaration_codecs_preserve_unknown_additive_fields() -> None:
    skill_payload = {
        "version": 1,
        "id": "skill",
        "content": "skill content",
        "future_metadata": {"$future_v2": ["ignored"]},
    }
    decoded_skill = SkillSpecCodec().decode(json.dumps(skill_payload).encode())
    assert decoded_skill == SkillSpec("skill", "skill content")
    assert decoded_skill._extensions["future_metadata"] == {
        "$future_v2": ["ignored"]
    }

    mcp_payload = {
        "version": 1,
        "id": "mcp",
        "command": "echo",
        "future_metadata": {"$future_v2": ["ignored"]},
    }
    decoded_mcp = MCPServerSpecCodec().decode(json.dumps(mcp_payload).encode())
    assert decoded_mcp == MCPServerSpec("mcp", "echo")
    assert decoded_mcp._extensions["future_metadata"] == {
        "$future_v2": ["ignored"]
    }


def test_agent_spec_codec_ignores_unknown_usage_limit_fields() -> None:
    payload = {
        "version": 1,
        "id": "agent",
        "usage_limits": {
            "model_requests": 1,
            "future_limit": {"unit": "request"},
        },
    }

    decoded = AgentSpecCodec().decode(json.dumps(payload).encode())

    assert decoded.usage_limits == AgentUsageLimits(model_requests=1)


def test_spec_constructors_reject_invalid_values() -> None:
    with pytest.raises(TypeError):
        AgentUsageLimits(model_requests=True)
    with pytest.raises(ValueError):
        AgentUsageLimits()
    with pytest.raises(ValueError):
        AgentSpec("")
    with pytest.raises(TypeError):
        AgentSpec("agent", instructions=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkillSpec("", "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", "")


@pytest.mark.asyncio
async def test_linktools_skills_is_the_direct_skill_capability() -> None:
    capability = LinkToolsSkills(
        (
            SkillDefinition(SkillSpec("z", "z skill")),
            SkillDefinition(SkillSpec("a", "a skill")),
        ),
        SkillSourceRegistry(),
    )
    assert [item["id"] for item in await capability.list_skills()] == ["a", "z"]
    assert capability.get_toolset().id == "linktools-skill"

    with pytest.raises(AIError) as error:
        LinkToolsSkills(
            (
                SkillDefinition(SkillSpec("same", "one")),
                SkillDefinition(SkillSpec("same", "two")),
            ),
            SkillSourceRegistry(),
        )
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


async def _business(value: str) -> str:
    return value


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_runtime_tool_boundary_requires_a_descriptor_for_every_leaf() -> None:
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([_business]),),
        {
            "_business": ManagedToolDescriptor(
                effect_owner="none",
                effect="none",
                tool_class="business",
            )
        },
        id="business",
    )
    context = _context()
    tools = await boundary.get_tools(context)
    assert (
        await boundary.call_tool(
            "_business",
            {"value": "ok"},
            context,
            tools["_business"],
        )
        == "ok"
    )

    unknown = RuntimeToolBoundaryToolset(
        (FunctionToolset([_business]),),
        {},
        id="invalid",
    )
    with pytest.raises(AIError) as error:
        await unknown.get_tools(context)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
