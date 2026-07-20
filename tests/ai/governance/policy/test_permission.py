#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_permission.py"""

import asyncio

from linktools.ai.governance.policy.permission import PermissionRule
from linktools.ai.governance.policy.rule import (
    Permission,
    PolicyDecisionKind,
    ToolContext,
    ToolPolicyMetadata,
    ToolRequest,
    RiskLevel,
    SideEffectKind,
    ApprovalMode,
)


def _meta(permissions: "frozenset[Permission]") -> ToolPolicyMetadata:
    return ToolPolicyMetadata(
        permissions=permissions,
        risk=RiskLevel.MEDIUM,
        side_effect=SideEffectKind.READ_ONLY,
        approval=ApprovalMode.NEVER,
    )


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", session_id="s")


def _request() -> ToolRequest:
    return ToolRequest(tool_name="file.write", arguments={"path": "/tmp/x"})


async def _run() -> None:
    # (i) tool not in metadata -> ALLOW
    rule = PermissionRule(allowed=frozenset({Permission.READ}), tool_metadata={})
    decision = await rule.evaluate(_request(), _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW

    # (ii) declared WRITE requirement vs allowed={READ} -> DENY
    rule = PermissionRule(
        allowed=frozenset({Permission.READ}),
        tool_metadata={"file.write": _meta(frozenset({Permission.WRITE}))},
    )
    decision = await rule.evaluate(_request(), _ctx())
    assert decision.kind == PolicyDecisionKind.DENY
    assert "WRITE" in (decision.reason or "")

    # (iii) constructor knob: allow WRITE -> ALLOW same request
    rule = PermissionRule(
        allowed=frozenset({Permission.READ, Permission.WRITE}),
        tool_metadata={"file.write": _meta(frozenset({Permission.WRITE}))},
    )
    decision = await rule.evaluate(_request(), _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW


def test_permission_rule():
    asyncio.run(_run())
