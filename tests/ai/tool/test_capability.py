#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/tool/test_capability.py — drives a real pydantic-ai Agent, matching
the FunctionModel-based pattern already used in tests/ai/security/test_hook.py.
Tool name is "terminal" (not "bash") because CommandRule (scenario) checks
`request.tool_name != "terminal"` to decide whether to inspect the command at
all -- using any other tool name would never reach the denied-pattern check.

Phase 1 design note refactoring: the per-Run ToolContext reaches the capability
via pydantic-ai dependency injection -- ``deps=AgentDependencies(tool_context=...)``
on ``agent.run()`` becomes ``ctx.deps.tool_context`` inside the hook. No mutable
``current_context`` field is set on the capability anymore."""
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from linktools.ai.policy.engine import PolicyEngine, ToolContext
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

    agent = Agent(FunctionModel(model_fn), capabilities=[capability], deps_type=AgentDependencies)

    @agent.tool_plain
    def terminal(command: str) -> dict:
        return {"exit_code": 0, "stdout": f"ran: {command}", "stderr": ""}

    return agent


def _deps(run_id: str = "run-1", session_id: str = "session-1") -> AgentDependencies:
    return AgentDependencies(tool_context=ToolContext(run_id=run_id, session_id=session_id))


@pytest.mark.asyncio
async def test_denied_tool_call_surfaces_as_skip_not_exception():
    executor = ToolExecutor(policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)))
    capability = build_policy_capability(executor)
    agent = _agent_calling_terminal_with("rm -rf /", capability)

    result = await agent.run("do something", deps=_deps())
    tool_returns = _tool_returns(result)
    assert tool_returns, "expected the terminal tool to have been called"
    assert "error" in str(tool_returns[0]).lower() or "denied" in str(tool_returns[0]).lower()
    assert "ran:" not in str(tool_returns[0])


@pytest.mark.asyncio
async def test_allowed_tool_call_executes_normally():
    executor = ToolExecutor(policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)))
    capability = build_policy_capability(executor)
    agent = _agent_calling_terminal_with("ls -la", capability)

    result = await agent.run("do something", deps=_deps())
    tool_returns = _tool_returns(result)
    assert tool_returns
    assert "ran: ls -la" in str(tool_returns[0])


@pytest.mark.asyncio
async def test_capability_has_no_current_context_field():
    # Phase 1 refactoring: the mutable ``current_context`` field is gone
    # entirely. The per-Run ToolContext arrives via pydantic-ai dependency
    # injection (ctx.deps.tool_context), so a fresh capability has no such
    # attribute and is identical before/after a run -- the concurrency-safety
    # invariant this refactor delivers.
    executor = ToolExecutor(policy=PolicyEngine(rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)))
    capability = build_policy_capability(executor)
    assert not hasattr(capability, "current_context")
    agent = _agent_calling_terminal_with("ls -la", capability)
    await agent.run("do something", deps=_deps())
    assert not hasattr(capability, "current_context")
