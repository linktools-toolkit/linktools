#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_path.py"""
import asyncio
from pathlib import Path

from linktools.ai.policy.path import PathRule
from linktools.ai.policy.rule import PolicyDecisionKind, ToolContext, ToolRequest


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", session_id="s")


async def _run() -> None:
    rule = PathRule(allowed_roots=(Path("/safe"),))

    # (i) path under allowed root -> ALLOW
    request = ToolRequest(tool_name="file.read", arguments={"path": "/safe/x"})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW

    # (ii) path outside allowed root -> DENY
    request = ToolRequest(tool_name="file.read", arguments={"path": "/etc/passwd"})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.DENY

    # (iii) no path argument -> ALLOW
    request = ToolRequest(tool_name="noop.tool", arguments={})
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW


async def _run_terminal() -> None:
    # constructor knob: terminal command parsing -- abs path escaping -> DENY
    rule = PathRule(allowed_roots=(Path("/safe"),))
    request = ToolRequest(
        tool_name="terminal",
        arguments={"command": "cat /etc/passwd"},
    )
    decision = await rule.evaluate(request, _ctx())
    assert decision.kind == PolicyDecisionKind.DENY


def test_path_rule():
    asyncio.run(_run())


def test_path_rule_terminal_command():
    asyncio.run(_run_terminal())
