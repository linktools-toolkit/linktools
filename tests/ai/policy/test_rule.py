#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_rule.py"""

import pytest

from linktools.ai.policy.rule import (
    ApprovalMode,
    Permission,
    RiskLevel,
    SideEffectKind,
    ToolPolicyMetadata,
)


def test_permission_values():
    assert Permission.READ.value == "read"
    assert Permission.WRITE.value == "write"
    assert Permission.EXECUTE.value == "execute"
    assert Permission.NETWORK.value == "network"
    assert Permission.ADMIN.value == "admin"
    assert {m.value for m in Permission} == {
        "read",
        "write",
        "execute",
        "network",
        "admin",
    }


def test_risk_level_ordinal_compare():
    assert RiskLevel.LOW.value == 1
    assert (RiskLevel.MEDIUM < RiskLevel.HIGH) is True
    assert (RiskLevel.CRITICAL > RiskLevel.HIGH) is True
    assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL


def test_side_effect_kind_members():
    assert {m.value for m in SideEffectKind} == {
        "none",
        "read_only",
        "namespace_mutating",
        "destructive",
    }


def test_approval_mode_members():
    assert {m.value for m in ApprovalMode} == {"never", "on_risk", "always"}


def test_tool_policy_metadata_constructs_and_frozen():
    meta = ToolPolicyMetadata(
        permissions=frozenset({Permission.READ, Permission.WRITE}),
        risk=RiskLevel.MEDIUM,
        side_effect=SideEffectKind.READ_ONLY,
        approval=ApprovalMode.NEVER,
    )
    assert meta.permissions == frozenset({Permission.READ, Permission.WRITE})
    assert meta.risk is RiskLevel.MEDIUM
    assert meta.side_effect is SideEffectKind.READ_ONLY
    assert meta.approval is ApprovalMode.NEVER
    with pytest.raises(Exception):
        meta.risk = RiskLevel.HIGH  # type: ignore[misc]


def test_permission_set_difference_semantics():
    requested = frozenset({Permission.WRITE, Permission.ADMIN})
    granted = frozenset({Permission.READ})
    # Non-empty difference -> rule would DENY.
    assert requested - granted == frozenset({Permission.WRITE, Permission.ADMIN})
    # Empty difference -> rule would ALLOW.
    assert (
        frozenset({Permission.READ}) - frozenset({Permission.READ, Permission.WRITE})
        == frozenset()
    )
