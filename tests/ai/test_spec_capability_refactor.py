#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final declaration, runtime policy, and SkillCapability contracts."""

import json

import pytest
from linktools.ai.capability import SkillCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._capabilities import (
    PLAN_SAFE_METADATA_KEY,
    select_runtime_tool_names,
    tool_allowed_in_planning,
    tool_name_allowed,
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
from pydantic_ai.tools import ToolDefinition


def test_runtime_tool_selection_keeps_planning_outside_allow_tools() -> None:
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
    ) == ("delegate_task",)


def test_runtime_tool_selection_honors_wildcard_for_ordinary_memory_tools() -> None:
    assert tool_name_allowed("read_memory", ("*",))
    assert select_runtime_tool_names(
        ordinary_tool_policy=("*",),
        memory_scope="memory",
    ) == ("delete_memory", "read_memory", "search_memory", "write_memory")


def test_planning_gate_requires_framework_filesystem_provenance() -> None:
    classes = (("read_file", "filesystem.read"),)
    trusted = ToolDefinition(name="read_file", capability_id="workspace-filesystem")
    fake = ToolDefinition(name="read_file", capability_id="custom-filesystem")
    explicit_custom = ToolDefinition(
        name="custom_read",
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
    spoofed = ToolDefinition(name="mcp__trusted__read", capability_id="custom-mcp")

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
    ("payload", "expected_code"),
    (
        (
            {"version": 1, "id": 1, "model": "default"},
            ErrorCode.OUTPUT_CONTRACT_INVALID,
        ),
        (
            {"version": 1, "id": "agent", "model": 1},
            ErrorCode.OUTPUT_CONTRACT_INVALID,
        ),
        (
            {"version": True, "id": "agent", "model": "default"},
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            {"version": 2, "id": "agent", "model": "default"},
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
        ),
    ),
)
def test_agent_spec_codec_rejects_invalid_v1_payload(
    payload: dict[str, object],
    expected_code: ErrorCode,
) -> None:
    with pytest.raises(AIError) as error:
        AgentSpecCodec().decode(json.dumps(payload).encode("utf-8"))
    assert error.value.code is expected_code


def test_declaration_codecs_preserve_unknown_additive_fields_without_affecting_semantics() -> None:
    skill_payload = {
        "version": 1,
        "id": "skill",
        "content": "skill content",
        "future_metadata": {"$future_v2": ["ignored"]},
    }
    decoded_skill = SkillSpecCodec().decode(json.dumps(skill_payload).encode("utf-8"))
    assert decoded_skill == SkillSpec("skill", "skill content")
    assert decoded_skill._extensions["future_metadata"] == {"$future_v2": ["ignored"]}

    mcp_payload = {
        "version": 1,
        "id": "mcp",
        "command": "echo",
        "future_metadata": {"$future_v2": ["ignored"]},
    }
    decoded_mcp = MCPServerSpecCodec().decode(json.dumps(mcp_payload).encode("utf-8"))
    assert decoded_mcp == MCPServerSpec("mcp", "echo")
    assert decoded_mcp._extensions["future_metadata"] == {"$future_v2": ["ignored"]}


@pytest.mark.parametrize(
    ("codec", "payload"),
    (
        (SkillSpecCodec(), {"version": 1, "id": 1, "content": "skill"}),
        (MCPServerSpecCodec(), {"version": 1, "id": "mcp", "command": 1}),
        (MCPServerSpecCodec(), {"version": 1, "id": "mcp", "command": "echo", "args": [1]}),
    ),
)
def test_declaration_codecs_reject_type_coercion(codec: object, payload: dict[str, object]) -> None:
    with pytest.raises((AIError, TypeError, ValueError)):
        codec.decode(json.dumps(payload).encode("utf-8"))  # type: ignore[attr-defined]


def test_spec_constructors_reject_mismatched_runtime_types() -> None:
    with pytest.raises(TypeError):
        AgentUsageLimits(model_requests=True)
    with pytest.raises(TypeError):
        AgentSpec(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AgentSpec("agent", model="")
    with pytest.raises(TypeError):
        AgentSpec("agent", instructions="not-an-array")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AgentSpec("agent", instructions=(1,))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AgentSpec("agent", usage_limits=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkillSpec("", "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", "")
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", "echo", args="not-an-array")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MCPServerSpec("mcp", "echo", args=(1,))  # type: ignore[arg-type]


def test_spec_constructors_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        AgentUsageLimits()
    with pytest.raises(ValueError):
        AgentUsageLimits(model_requests=0)
    with pytest.raises(ValueError):
        AgentSpec("")
    with pytest.raises(ValueError):
        SkillSpec("", "content")
    with pytest.raises(ValueError):
        MCPServerSpec("", "echo")


def test_skill_capability_sorts_selected_skills_and_rejects_duplicates() -> None:
    capability = SkillCapability((SkillSpec("z", "z skill"), SkillSpec("a", "a skill")))
    assert tuple(item.id for item in capability.skills) == ("a", "z")

    with pytest.raises(AIError) as error:
        SkillCapability((SkillSpec("same", "one"), SkillSpec("same", "two")))
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT
