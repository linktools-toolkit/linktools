#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability selection and immutable skill catalog contracts."""

import json

import pytest
from linktools.ai.agent import select_platform_tool_names, tool_name_allowed
from linktools.ai.agent._capabilities import (
    PLAN_SAFE_METADATA_KEY,
    tool_allowed_in_planning,
)
from linktools.ai.asset import AssetRef
from linktools.ai.capability._skill import (
    SkillCatalogSnapshot,
    SkillDescriptor,
    bind_skill_capability,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)
from pydantic_ai.tools import ToolDefinition


def test_platform_tool_selection_keeps_planning_outside_allow_tools() -> None:
    assert (
        select_platform_tool_names(
            allow_tools=("write_plan",),
            memory_scope="memory",
        )
        == ()
    )
    assert select_platform_tool_names(
        allow_tools=(),
        memory_scope=None,
        planning=True,
    ) == ("write_plan",)
    assert select_platform_tool_names(
        allow_tools=(),
        memory_scope=None,
        subagent_available=True,
    ) == ("delegate_task",)


def test_platform_tool_selection_honors_wildcard_allow_tools() -> None:
    assert tool_name_allowed("read_memory", ("*",))
    assert select_platform_tool_names(
        allow_tools=("*",),
        memory_scope="memory",
    ) == ("delete_memory", "read_memory", "search_memory", "write_memory")


def test_planning_gate_requires_framework_filesystem_provenance() -> None:
    classes = (("read_file", "filesystem.read"),)
    trusted = ToolDefinition(
        name="read_file",
        capability_id="workspace-filesystem",
    )
    fake = ToolDefinition(name="read_file", capability_id="custom-filesystem")
    explicit_custom = ToolDefinition(
        name="read_file",
        capability_id="custom-filesystem",
        metadata={PLAN_SAFE_METADATA_KEY: True},
    )

    assert tool_allowed_in_planning(
        trusted,
        trusted_tool_classes=classes,
        trusted_mcp_selectors=(),
    )
    assert not tool_allowed_in_planning(
        fake,
        trusted_tool_classes=classes,
        trusted_mcp_selectors=(),
    )
    assert tool_allowed_in_planning(
        explicit_custom,
        trusted_tool_classes=classes,
        trusted_mcp_selectors=(),
    )


def test_planning_gate_uses_trusted_mcp_provenance_not_name_prefix() -> None:
    trusted = ToolDefinition(
        name="mcp__trusted__read",
        capability_id="mcp__trusted",
        metadata={PLAN_SAFE_METADATA_KEY: True},
    )
    custom = ToolDefinition(
        name="mcp__custom__read",
        capability_id="custom-mcp",
        metadata={PLAN_SAFE_METADATA_KEY: True},
    )
    spoofed = ToolDefinition(
        name="mcp__trusted__read",
        capability_id="custom-mcp",
    )

    assert not tool_allowed_in_planning(
        trusted,
        trusted_tool_classes=(),
        trusted_mcp_selectors=("mcp__trusted",),
    )
    assert tool_allowed_in_planning(
        custom,
        trusted_tool_classes=(),
        trusted_mcp_selectors=("mcp__trusted",),
    )
    assert not tool_allowed_in_planning(
        spoofed,
        trusted_tool_classes=(),
        trusted_mcp_selectors=("mcp__trusted",),
    )


def test_planning_gate_rejects_non_boolean_plan_safe_metadata() -> None:
    tool = ToolDefinition(
        name="custom",
        capability_id="custom",
        metadata={PLAN_SAFE_METADATA_KEY: "yes"},
    )

    with pytest.raises(AIError) as error:
        tool_allowed_in_planning(
            tool,
            trusted_tool_classes=(),
            trusted_mcp_selectors=(),
        )

    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.parametrize(
    "payload",
    (
        {"id": 1, "revision": 1, "model": "default"},
        {"id": "agent", "revision": True, "model": "default"},
        {"id": "agent", "revision": "1", "model": "default"},
        {"id": "agent", "revision": 1, "model": 1},
    ),
)
def test_agent_spec_codec_rejects_type_coercion(payload: dict[str, object]) -> None:
    with pytest.raises(AIError) as error:
        AgentSpecCodec().decode(json.dumps(payload).encode("utf-8"))
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_asset_spec_codecs_ignore_unknown_additive_fields() -> None:
    skill_payload = {
        "id": "skill",
        "revision": 1,
        "content": "skill content",
        "future_metadata": {"$future_v2": ["ignored"]},
    }
    assert SkillSpecCodec().decode(
        json.dumps(skill_payload).encode("utf-8")
    ) == SkillSpec("skill", 1, "skill content")

    mcp_payload = {
        "id": "mcp",
        "revision": 1,
        "command": "echo",
        "future_metadata": {"$future_v2": ["ignored"]},
    }
    assert MCPServerSpecCodec().decode(
        json.dumps(mcp_payload).encode("utf-8")
    ) == MCPServerSpec("mcp", 1, "echo")


@pytest.mark.parametrize(
    ("codec", "payload"),
    (
        (SkillSpecCodec(), {"id": 1, "revision": 1, "content": "skill"}),
        (SkillSpecCodec(), {"id": "skill", "revision": True, "content": "skill"}),
        (MCPServerSpecCodec(), {"id": "mcp", "revision": "1", "command": "echo"}),
        (MCPServerSpecCodec(), {"id": "mcp", "revision": 1, "command": "echo", "args": [1]}),
    ),
)
def test_asset_spec_codecs_reject_type_coercion(codec: object, payload: dict[str, object]) -> None:
    with pytest.raises(AIError) as error:
        codec.decode(json.dumps(payload).encode("utf-8"))
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_spec_constructors_reject_mismatched_runtime_types() -> None:
    with pytest.raises(TypeError):
        AgentUsageLimits(model_requests=True)
    with pytest.raises(TypeError):
        AgentSpec(1, 1, "default")
    with pytest.raises(TypeError):
        AgentSpec("agent", True, "default")
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, 1)
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", instructions="not-an-array")
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", instructions=(1,))
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", metadata=[])
    with pytest.raises(TypeError):
        AgentSpec("agent", 1, "default", usage_limits=object())
    with pytest.raises(TypeError):
        SkillSpec(1, 1, "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", True, "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", 1, 1)
    with pytest.raises(TypeError):
        MCPServerSpec(1, 1, "echo")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", True, "echo")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", 1, 1)
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", 1, "echo", args="not-an-array")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", 1, "echo", args=(1,))


def test_spec_constructors_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        AgentUsageLimits()
    with pytest.raises(ValueError):
        AgentUsageLimits(model_requests=0)
    with pytest.raises(ValueError):
        AgentSpec("", 1, "default")
    with pytest.raises(ValueError):
        AgentSpec("agent", 0, "default")
    with pytest.raises(ValueError):
        AgentSpec("agent", 1, "")
    with pytest.raises(ValueError):
        SkillSpec("", 1, "content")
    with pytest.raises(ValueError):
        SkillSpec("skill", 0, "content")
    with pytest.raises(ValueError):
        MCPServerSpec("", 1, "echo")
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", 0, "echo")
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", 1, "")


def test_skill_catalog_snapshot_is_sorted_and_immutable() -> None:
    first = SkillSpec("z", 1, "z skill")
    second = SkillSpec("a", 1, "a skill")
    catalog = SkillCatalogSnapshot(
        (SkillDescriptor("z", 1, "z"), SkillDescriptor("a", 1, "a")),
        (first, second),
    )
    assert tuple(item.id for item in catalog.descriptors) == ("a", "z")
    assert tuple(item.id for item in catalog.specifications) == ("a", "z")


def test_skill_binding_requires_one_spec_per_discovered_asset() -> None:
    with pytest.raises(AIError) as error:
        bind_skill_capability((AssetRef("skill", "missing"),), ())
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
