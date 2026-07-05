#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/tool/test_capability.py — drives a real pydantic-ai Agent, matching
the FunctionModel-based pattern already used in tests/ai/security/test_hook.py.
Tool name is "terminal" (not "bash") because CommandRule (Task 4) checks
`request.tool_name != "terminal"` to decide whether to inspect the command at
all -- using any other tool name would never reach the denied-pattern check."""
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.tool.capability import build_policy_capability
from linktools.ai.tool.executor import ToolExecutor


def _tool_returns(result) -> list:
    return [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", None) == "tool-return"
    ]


def _agent_calling_terminal_with(command: str, capability) -> Agent:
    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="terminal", args={"command": command})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model_fn), capabilities=[capability])

    @agent.tool_plain
    def terminal(command: str) -> dict:
        return {"exit_code": 0, "stdout": f"ran: {command}", "stderr": ""}

    return agent


@pytest.mark.asyncio
async def test_denied_tool_call_surfaces_as_skip_not_exception():
    from linktools.ai.policy.engine import ToolContext

    executor = ToolExecutor(policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)))
    capability = build_policy_capability(executor)
    capability.current_context = ToolContext(run_id="run-1", session_id="session-1")
    agent = _agent_calling_terminal_with("rm -rf /", capability)

    result = await agent.run("do something")
    tool_returns = _tool_returns(result)
    assert tool_returns, "expected the terminal tool to have been called"
    assert "error" in str(tool_returns[0]).lower() or "denied" in str(tool_returns[0]).lower()
    assert "ran:" not in str(tool_returns[0])


@pytest.mark.asyncio
async def test_allowed_tool_call_executes_normally():
    from linktools.ai.policy.engine import ToolContext

    executor = ToolExecutor(policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)))
    capability = build_policy_capability(executor)
    capability.current_context = ToolContext(run_id="run-1", session_id="session-1")
    agent = _agent_calling_terminal_with("ls -la", capability)

    result = await agent.run("do something")
    tool_returns = _tool_returns(result)
    assert tool_returns
    assert "ran: ls -la" in str(tool_returns[0])


@pytest.mark.asyncio
async def test_current_context_defaults_to_unknown_when_unset():
    # A capability that's never had current_context set (e.g. used outside a
    # real AgentRunner-managed Run) must not raise -- it falls back to a
    # placeholder ToolContext rather than crashing on a None attribute access.
    executor = ToolExecutor(policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)))
    capability = build_policy_capability(executor)
    assert capability.current_context is None
    agent = _agent_calling_terminal_with("ls -la", capability)
    result = await agent.run("do something")
    tool_returns = _tool_returns(result)
    assert "ran: ls -la" in str(tool_returns[0])
