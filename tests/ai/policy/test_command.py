#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_command.py"""
import pytest

from linktools.ai.policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from linktools.ai.policy.engine import PolicyDecisionKind, ToolContext, ToolRequest


@pytest.mark.asyncio
async def test_denies_rm_rf_root():
    rule = CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS)
    request = ToolRequest(tool_name="terminal", arguments={"command": "rm -rf /"})
    decision = await rule.evaluate(request, ToolContext(run_id="run-1", session_id="session-1"))
    assert decision.kind == PolicyDecisionKind.DENY


@pytest.mark.asyncio
async def test_allows_safe_command():
    rule = CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS)
    request = ToolRequest(tool_name="terminal", arguments={"command": "ls -la"})
    decision = await rule.evaluate(request, ToolContext(run_id="run-1", session_id="session-1"))
    assert decision.kind == PolicyDecisionKind.ALLOW


@pytest.mark.asyncio
async def test_ignores_non_terminal_tools():
    rule = CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS)
    request = ToolRequest(tool_name="file", arguments={"command": "rm -rf /"})
    decision = await rule.evaluate(request, ToolContext(run_id="run-1", session_id="session-1"))
    assert decision.kind == PolicyDecisionKind.ALLOW


@pytest.mark.asyncio
async def test_category_based_match_survives_tool_rename():
    """A tool renamed away from "bash"/"terminal" is still caught by the
    denylist as long as its descriptor declares category="terminal" -- the
    rule must not be evadable by renaming."""
    rule = CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS)
    request = ToolRequest(
        tool_name="run_shell_command", category="terminal",
        arguments={"command": "rm -rf /"},
    )
    decision = await rule.evaluate(request, ToolContext(run_id="run-1", session_id="session-1"))
    assert decision.kind == PolicyDecisionKind.DENY


@pytest.mark.asyncio
async def test_category_based_match_ignores_non_terminal_category_even_if_named_bash():
    """When category info IS available, it is authoritative over the name --
    a tool named "bash" but declared as a non-terminal category is not
    subject to the command denylist."""
    rule = CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS)
    request = ToolRequest(
        tool_name="bash", category="discovery",
        arguments={"command": "rm -rf /"},
    )
    decision = await rule.evaluate(request, ToolContext(run_id="run-1", session_id="session-1"))
    assert decision.kind == PolicyDecisionKind.ALLOW
