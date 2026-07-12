#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.dependencies import AgentDependencies
from linktools.ai.errors import ToolDeniedError
from linktools.ai.policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
from linktools.ai.policy.engine import (
    PolicyDecision,
    PolicyDecisionKind,
    PolicyEngine,
    ToolContext,
    ToolRequest,
)
from linktools.ai.tool.pydantic import build_policy_capability
from linktools.ai.tool.executor import ToolExecutor


def test_execute_runs_check_then_handler_and_returns_result():
    executor = ToolExecutor(policy=PolicyEngine(rules=()))

    async def _handler(value: int) -> int:
        return value * 2

    async def _run():
        return await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 21}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
        )

    assert asyncio.run(_run()) == 42


def test_execute_raises_before_handler_when_policy_denies():
    class _Deny:
        async def evaluate(self, request, context):
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY, rule_id="x", reason="no"
            )

    executor = ToolExecutor(policy=PolicyEngine(rules=(_Deny(),)))
    ran = {"handler": False}

    async def _handler(value: int) -> int:
        ran["handler"] = True
        return value

    async def _run():
        await executor.execute(
            ToolRequest(tool_name="double", arguments={"value": 1}),
            ToolContext(run_id="r1", session_id="s1"),
            _handler,
        )

    with pytest.raises(ToolDeniedError):
        asyncio.run(_run())
    assert ran["handler"] is False


def _driven_agent(capability) -> Agent:
    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="terminal", args={"command": "ls"})]
            )
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(
        FunctionModel(model_fn), capabilities=[capability], deps_type=AgentDependencies
    )

    @agent.tool_plain
    def terminal(command: str) -> str:
        raise RuntimeError("handler crashed")

    return agent


def test_policy_capability_after_tool_execute_returns_result_unchanged():
    capability = build_policy_capability(ToolExecutor(policy=PolicyEngine(rules=())))

    async def _run():
        return await capability.after_tool_execute(
            ctx=None, call=None, tool_def=None, args={}, result="some-result"
        )

    assert asyncio.run(_run()) == "some-result"


def test_policy_capability_on_tool_execute_error_surfaces_as_skip():
    capability = build_policy_capability(
        ToolExecutor(
            policy=PolicyEngine(
                rules=(CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),)
            )
        )
    )
    agent = _driven_agent(capability)

    async def _run():
        return await agent.run(
            "call it",
            deps=AgentDependencies(
                tool_context=ToolContext(run_id="r", session_id="s")
            ),
        )

    result = asyncio.run(_run())
    tool_returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", None) == "tool-return"
    ]
    assert tool_returns
    assert (
        "handler crashed" in str(tool_returns[0])
        or "error" in str(tool_returns[0]).lower()
    )
