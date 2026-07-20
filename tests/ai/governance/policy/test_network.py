#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_network.py"""

import asyncio

from linktools.ai.governance.policy.network import NetworkRule
from linktools.ai.governance.policy.rule import PolicyDecisionKind, ToolContext, ToolRequest


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", session_id="s")


async def _run() -> None:
    rule = NetworkRule(allowed_hosts=frozenset({"api.example.com"}))

    # (i) allowed host (in url) -> ALLOW
    request = ToolRequest(
        tool_name="http.get",
        arguments={"url": "https://api.example.com/x"},
    )
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW

    # (ii) disallowed host -> DENY
    request = ToolRequest(
        tool_name="http.get",
        arguments={"url": "https://evil.com/x"},
    )
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.DENY

    # (iii) no url/host arg -> ALLOW
    request = ToolRequest(tool_name="noop.tool", arguments={})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW


async def _run_host_arg() -> None:
    # constructor knob: bare "host" argument is honored too
    rule = NetworkRule(allowed_hosts=frozenset({"api.example.com"}))
    request = ToolRequest(tool_name="dns.lookup", arguments={"host": "evil.com"})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.DENY


def test_network_rule():
    asyncio.run(_run())


def test_network_rule_host_arg():
    asyncio.run(_run_host_arg())
