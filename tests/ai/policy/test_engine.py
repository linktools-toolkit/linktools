#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_engine.py"""
import pytest

from linktools.ai.policy.engine import PolicyDecision, PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest


class _AllowRule:
    async def evaluate(self, request, context):
        return PolicyDecision(kind=PolicyDecisionKind.ALLOW, rule_id="allow-rule", reason=None)


class _DenyRule:
    async def evaluate(self, request, context):
        return PolicyDecision(kind=PolicyDecisionKind.DENY, rule_id="deny-rule", reason="blocked")


class _ApprovalRule:
    async def evaluate(self, request, context):
        return PolicyDecision(kind=PolicyDecisionKind.REQUIRE_APPROVAL, rule_id="approval-rule", reason="risky")


def _request() -> ToolRequest:
    return ToolRequest(tool_name="terminal", arguments={"command": "ls"})


def _context() -> ToolContext:
    return ToolContext(run_id="run-1", session_id="session-1")


@pytest.mark.asyncio
async def test_no_rules_allows_by_default():
    engine = PolicyEngine(rules=())
    decision = await engine.evaluate(_request(), _context())
    assert decision.kind == PolicyDecisionKind.ALLOW
    assert decision.rule_id == "default"


@pytest.mark.asyncio
async def test_deny_wins_over_allow():
    engine = PolicyEngine(rules=(_AllowRule(), _DenyRule()))
    decision = await engine.evaluate(_request(), _context())
    assert decision.kind == PolicyDecisionKind.DENY
    assert decision.rule_id == "deny-rule"


@pytest.mark.asyncio
async def test_require_approval_wins_over_allow_but_not_deny():
    engine = PolicyEngine(rules=(_AllowRule(), _ApprovalRule()))
    decision = await engine.evaluate(_request(), _context())
    assert decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL

    engine_with_deny = PolicyEngine(rules=(_ApprovalRule(), _DenyRule()))
    decision2 = await engine_with_deny.evaluate(_request(), _context())
    assert decision2.kind == PolicyDecisionKind.DENY
