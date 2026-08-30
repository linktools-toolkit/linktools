#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentic extension Permission contract regressions."""

import pytest

from linktools.ai.workspace import (
    PermissionDecision,
    ToolPermissionRule,
    WorkspacePolicy,
    WorkspaceToolPermissionPolicy,
)


def test_permission_contract_is_public_and_defaults_allow() -> None:
    decision: PermissionDecision = "allow"
    assert decision == "allow"
    policy = WorkspacePolicy()
    policy.validate()
    assert policy.tool_permissions == WorkspaceToolPermissionPolicy()
    assert not policy.tool_permissions.requires_approval
    assert policy.tool_permissions.decide(tool_name="read_file", tool_class="filesystem.read") == "allow"


@pytest.mark.parametrize("value", [None, 1, True, (), []])
def test_permission_rule_rejects_non_string_decision_with_type_error(value: object) -> None:
    with pytest.raises(TypeError):
        ToolPermissionRule(value, tool_name="read_file")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, 1, True, (), []])
def test_permission_policy_rejects_non_string_default_with_type_error(value: object) -> None:
    with pytest.raises(TypeError):
        WorkspaceToolPermissionPolicy(default=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, True, (), []])
def test_permission_rule_rejects_non_string_tool_name_with_type_error(value: object) -> None:
    with pytest.raises(TypeError):
        ToolPermissionRule("allow", tool_name=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, True, (), []])
def test_permission_rule_rejects_non_string_tool_class_with_type_error(value: object) -> None:
    with pytest.raises(TypeError):
        ToolPermissionRule("allow", tool_class=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, True, (), []])
def test_permission_decide_rejects_non_string_inputs_with_type_error(value: object) -> None:
    policy = WorkspaceToolPermissionPolicy()
    with pytest.raises(TypeError):
        policy.decide(tool_name=value, tool_class=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        policy.decide(tool_name="read_file", tool_class=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("decision", ["", "ALLOW", "unknown"])
def test_permission_invalid_decision_string_is_value_error(decision: str) -> None:
    with pytest.raises(ValueError):
        ToolPermissionRule(decision, tool_name="read_file")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WorkspaceToolPermissionPolicy(default=decision)  # type: ignore[arg-type]


@pytest.mark.parametrize("tool_name", ["", " read_file", "read_file ", "read*", "*"])
def test_permission_invalid_tool_name_is_value_error(tool_name: str) -> None:
    with pytest.raises(ValueError):
        ToolPermissionRule("allow", tool_name=tool_name)
    with pytest.raises(ValueError):
        WorkspaceToolPermissionPolicy().decide(tool_name=tool_name, tool_class=None)


@pytest.mark.parametrize("tool_class", ["filesystem", "filesystem.*", "path:/tmp", "shell:rm"])
def test_permission_unknown_tool_class_is_value_error(tool_class: str) -> None:
    with pytest.raises(ValueError):
        ToolPermissionRule("allow", tool_class=tool_class)
    with pytest.raises(ValueError):
        WorkspaceToolPermissionPolicy().decide(tool_name="tool", tool_class=tool_class)


def test_permission_rule_requires_exactly_one_selector() -> None:
    with pytest.raises(ValueError):
        ToolPermissionRule("allow")
    with pytest.raises(ValueError):
        ToolPermissionRule("allow", tool_name="read_file", tool_class="filesystem.read")


def test_permission_exact_tool_and_class_rules_support_all_decisions() -> None:
    for decision in ("allow", "ask", "deny"):
        exact = WorkspaceToolPermissionPolicy(
            rules=(ToolPermissionRule(decision, tool_name="read_file"),),
            default="allow",
        )
        by_class = WorkspaceToolPermissionPolicy(
            rules=(ToolPermissionRule(decision, tool_class="filesystem.read"),),
            default="allow",
        )
        assert exact.decide(tool_name="read_file", tool_class="filesystem.read") == decision
        assert by_class.decide(tool_name="read_file", tool_class="filesystem.read") == decision


def test_permission_precedence_is_deny_then_ask_then_allow() -> None:
    policy = WorkspaceToolPermissionPolicy(
        rules=(
            ToolPermissionRule("allow", tool_name="read_file"),
            ToolPermissionRule("ask", tool_class="filesystem.read"),
            ToolPermissionRule("deny", tool_name="read_file"),
        ),
        default="allow",
    )
    assert policy.requires_approval
    assert policy.decide(tool_name="read_file", tool_class="filesystem.read") == "deny"

    ask_over_allow = WorkspaceToolPermissionPolicy(
        rules=(
            ToolPermissionRule("allow", tool_name="read_file"),
            ToolPermissionRule("ask", tool_class="filesystem.read"),
        )
    )
    assert ask_over_allow.decide(tool_name="read_file", tool_class="filesystem.read") == "ask"


def test_permission_exact_mcp_final_tool_name_is_supported() -> None:
    policy = WorkspaceToolPermissionPolicy(
        rules=(ToolPermissionRule("deny", tool_name="mcp__server__read"),)
    )
    assert policy.decide(tool_name="mcp__server__read", tool_class=None) == "deny"
    assert policy.decide(tool_name="mcp__server__write", tool_class=None) == "allow"
