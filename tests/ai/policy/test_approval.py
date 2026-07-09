#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_approval.py"""
import asyncio

from linktools.ai.policy.approval import ApprovalRule
from linktools.ai.policy.rule import (
    ApprovalMode,
    Permission,
    PolicyDecisionKind,
    RiskLevel,
    SideEffectKind,
    ToolContext,
    ToolPolicyMetadata,
    ToolRequest,
)


def _meta(side_effect: SideEffectKind) -> ToolPolicyMetadata:
    return ToolPolicyMetadata(
        permissions=frozenset({Permission.READ}),
        risk=RiskLevel.MEDIUM,
        side_effect=side_effect,
        approval=ApprovalMode.NEVER,
    )


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", session_id="s")


async def _run() -> None:
    # (i) explicit require_for -> REQUIRE_APPROVAL
    rule = ApprovalRule(require_for=frozenset({"terminal"}))
    request = ToolRequest(tool_name="terminal", arguments={})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL

    # (ii) tool not in require_for and no metadata -> ALLOW
    request = ToolRequest(tool_name="file.read", arguments={})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW

    # (iii) constructor knob: require_side_effect=DESTRUCTIVE + declared DESTRUCTIVE -> REQUIRE_APPROVAL
    rule = ApprovalRule(
        require_side_effect=SideEffectKind.DESTRUCTIVE,
        tool_metadata={"format.disk": _meta(SideEffectKind.DESTRUCTIVE)},
    )
    request = ToolRequest(tool_name="format.disk", arguments={})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL

    # (iv) side_effect=READ_ONLY vs threshold DESTRUCTIVE -> ALLOW
    rule = ApprovalRule(
        require_side_effect=SideEffectKind.DESTRUCTIVE,
        tool_metadata={"file.read": _meta(SideEffectKind.READ_ONLY)},
    )
    request = ToolRequest(tool_name="file.read", arguments={})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW


def test_approval_rule():
    asyncio.run(_run())
