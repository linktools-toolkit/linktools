#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_risk.py"""

import asyncio

from linktools.ai.policy.risk import ResourceLimitRule, RiskRule
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


def _meta(risk: RiskLevel) -> ToolPolicyMetadata:
    return ToolPolicyMetadata(
        permissions=frozenset({Permission.READ}),
        risk=risk,
        side_effect=SideEffectKind.READ_ONLY,
        approval=ApprovalMode.NEVER,
    )


def _ctx(metadata=None) -> ToolContext:
    if metadata is None:
        metadata = {}
    return ToolContext(run_id="r", session_id="s", metadata=metadata)


def _request(tool_name: str = "explosive.tool") -> ToolRequest:
    return ToolRequest(tool_name=tool_name, arguments={})


async def _run_risk() -> None:
    # declared HIGH vs max=MEDIUM -> DENY
    rule = RiskRule(
        max_allowed=RiskLevel.MEDIUM,
        tool_metadata={"explosive.tool": _meta(RiskLevel.HIGH)},
    )
    decision = await rule.evaluate(_request(), _ctx())
    assert decision.kind == PolicyDecisionKind.DENY

    # declared MEDIUM vs max=MEDIUM -> ALLOW (not strictly greater)
    rule = RiskRule(
        max_allowed=RiskLevel.MEDIUM,
        tool_metadata={"explosive.tool": _meta(RiskLevel.MEDIUM)},
    )
    decision = await rule.evaluate(_request(), _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW

    # tool not in metadata -> ALLOW
    rule = RiskRule(max_allowed=RiskLevel.MEDIUM, tool_metadata={})
    decision = await rule.evaluate(_request(), _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW


async def _run_resource() -> None:
    rule = ResourceLimitRule(limits={"max_tokens": 100})

    # tokens_used=150 > 100 -> DENY
    decision = await rule.evaluate(_request(), _ctx({"tokens_used": 150}))
    assert decision.kind == PolicyDecisionKind.DENY

    # tokens_used=50 <= 100 -> ALLOW
    decision = await rule.evaluate(_request(), _ctx({"tokens_used": 50}))
    assert decision.kind == PolicyDecisionKind.ALLOW

    # constructor knob: raise ceiling to 200 -> ALLOW at 150
    rule = ResourceLimitRule(limits={"max_tokens": 200})
    decision = await rule.evaluate(_request(), _ctx({"tokens_used": 150}))
    assert decision.kind == PolicyDecisionKind.ALLOW


def test_risk_rule():
    asyncio.run(_run_risk())


def test_resource_limit_rule():
    asyncio.run(_run_resource())
