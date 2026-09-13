#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final declaration, capability, and runtime-leaf contracts."""

import json

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai import Tool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import (
    CapabilityGroup,
    LinkToolsSkills,
    SkillDefinition,
    SkillSourceRegistry,
    tool_semantic_metadata,
    validate_tool_semantic_metadata,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._harness_memory import select_harness_memory_tools
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.runtime.state import (
    RuntimeDomain,
    runtime_domain_uses_object_store,
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


def test_memory_owner_selects_only_its_declared_tools() -> None:
    assert select_harness_memory_tools(("write_plan",)) == ()
    assert select_harness_memory_tools(("read_memory",)) == ("read_memory",)
    assert select_harness_memory_tools(("*",)) == (
        "delete_memory",
        "read_memory",
        "search_memory",
        "write_memory",
    )


def test_tool_semantic_metadata_preserves_upstream_values() -> None:
    metadata = tool_semantic_metadata(
        base={"upstream": "retained"},
        effect="replay_safe",
        plan_safe=True,
        tool_class="business",
    )

    assert metadata["upstream"] == "retained"
    assert metadata["linktools.ai.effect"] == "replay_safe"
    assert metadata["linktools.ai.plan_safe"] is True
    assert metadata["linktools.ai.tool_class"] == "business"


@pytest.mark.parametrize(
    "metadata",
    (
        {"linktools.ai.effect": "unknown"},
        {"linktools.ai.plan_safe": 1},
        {"linktools.ai.tool_class": "filesystem"},
        {"linktools.ai.path_fields": ("path",)},
        {"linktools.ai.path_fields": ["path", "path"]},
        {"linktools.ai.context_dedupe": "legacy"},
    ),
)
def test_invalid_tool_semantics_fail_without_fallback(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(AIError) as error:
        validate_tool_semantic_metadata(metadata)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_business_tool_semantics_are_frozen_in_tool_metadata() -> None:
    async def business_tool(
        _ctx: RunContext[None],
        value: str,
    ) -> str:
        return value

    group = CapabilityGroup[None]("business")
    tool = group.tool(
        business_tool,
        effect="replay_safe",
        plan_safe=True,
    )

    candidate = (await group.freeze())[0]

    assert tool.tool_def.metadata == {
        "linktools.ai.effect": "replay_safe",
        "linktools.ai.plan_safe": True,
        "linktools.ai.tool_class": "business",
    }
    assert candidate.semantic_contract["metadata"] == tool.tool_def.metadata
    assert "config" not in candidate.semantic_contract


def test_runtime_domain_object_store_trait_has_one_owner() -> None:
    object_domains = {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.TASK,
        RuntimeDomain.RECOVERY,
    }

    assert {
        domain
        for domain in RuntimeDomain
        if runtime_domain_uses_object_store(domain)
    } == object_domains
    assert not runtime_domain_uses_object_store(RuntimeDomain.EVALUATION)


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


def test_declaration_codecs_ignore_unknown_additive_fields() -> None:
    agent_payload = {
        "version": 1,
        "id": "agent",
        "future_metadata": {"future": True},
    }
    assert AgentSpecCodec().decode(json.dumps(agent_payload).encode()) == AgentSpec(
        "agent"
    )

    skill_payload = {
        "version": 1,
        "id": "skill",
        "content": "skill content",
        "future_metadata": {"future": True},
    }
    assert SkillSpecCodec().decode(json.dumps(skill_payload).encode()) == SkillSpec(
        "skill",
        "skill content",
    )

    mcp_payload = {
        "version": 1,
        "id": "mcp",
        "command": "echo",
        "future_metadata": {"future": True},
    }
    assert MCPServerSpecCodec().decode(json.dumps(mcp_payload).encode()) == MCPServerSpec(
        "mcp",
        "echo",
    )


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
    assert capability.get_toolset().id == "linktools.ai.skills"

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
        (
            FunctionToolset(
                [
                    Tool(
                        _business,
                        metadata=tool_semantic_metadata(
                            effect="none",
                            tool_class="business",
                        ),
                    )
                ]
            ),
        ),
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


@pytest.mark.asyncio
async def test_runtime_tool_boundary_does_not_rewrite_explicit_descriptor() -> None:
    boundary = RuntimeToolBoundaryToolset(
        (
            FunctionToolset(
                [
                    Tool(
                        _business,
                        metadata={"linktools.ai.effect": "invalid"},
                    )
                ]
            ),
        ),
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
