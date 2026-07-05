#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/tool/test_executor.py"""
import pytest

from linktools.ai.errors import ToolApprovalRequiredError, ToolDeniedError
from linktools.ai.policy.engine import PolicyDecision, PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest
from linktools.ai.tool.executor import ToolExecutor


class _AlwaysDenyRule:
    async def evaluate(self, request, context):
        return PolicyDecision(kind=PolicyDecisionKind.DENY, rule_id="deny", reason="no")


class _AlwaysApprovalRule:
    async def evaluate(self, request, context):
        return PolicyDecision(kind=PolicyDecisionKind.REQUIRE_APPROVAL, rule_id="approval", reason="ask")


@pytest.mark.asyncio
async def test_check_allows_when_policy_allows():
    executor = ToolExecutor(policy=PolicyEngine(rules=()))
    await executor.check(ToolRequest(tool_name="file", arguments={}), ToolContext(run_id="r1", session_id="s1"))


@pytest.mark.asyncio
async def test_check_raises_tool_denied_on_deny():
    executor = ToolExecutor(policy=PolicyEngine(rules=(_AlwaysDenyRule(),)))
    with pytest.raises(ToolDeniedError):
        await executor.check(ToolRequest(tool_name="terminal", arguments={}), ToolContext(run_id="r1", session_id="s1"))


@pytest.mark.asyncio
async def test_check_raises_approval_required_on_require_approval():
    executor = ToolExecutor(policy=PolicyEngine(rules=(_AlwaysApprovalRule(),)))
    with pytest.raises(ToolApprovalRequiredError):
        await executor.check(ToolRequest(tool_name="terminal", arguments={}), ToolContext(run_id="r1", session_id="s1"))
