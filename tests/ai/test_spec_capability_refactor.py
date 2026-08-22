#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability selection and immutable skill catalog contracts."""

import json

import pytest
from pydantic_ai.tools import ToolDefinition

from linktools.ai.agent import select_platform_tool_names
from linktools.ai.agent._capabilities import PLAN_SAFE_METADATA_KEY, tool_allowed_in_planning
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
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)


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
    with pytest.raises(ValueError):
        AgentSpec("agent", 1, "default", instructions="not-an-array")
    with pytest.raises(ValueError):
        AgentSpec("agent", 1, "default", usage_limits=object())
    with pytest.raises(ValueError):
        SkillSpec("skill", True, "content")
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", 1, "echo", args="not-an-array")


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
