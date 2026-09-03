#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration regressions for workspace filesystem failure semantics."""

from pathlib import Path
from typing import Any

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.runtime._agent_executor import _RuntimePersistenceBoundary
from linktools.ai.runtime._capabilities import (
    ToolOperationDecision,
    _RuntimeStepPersistence,
)
from linktools.ai.workspace import Workspace
from pydantic_ai import Agent
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.step_persistence import InMemoryStepStore

pytestmark = pytest.mark.asyncio


class _Bridge:
    def __init__(self, replay_safe: bool) -> None:
        self.replay_safe = replay_safe
        self.transitions: list[str] = []

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        assert replay_safe is self.replay_safe
        self.transitions.append("begin")
        return ToolOperationDecision("operation", "owner", 1, replay_safe)

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.transitions.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.transitions.append("fail")
        return False

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del decision, error
        self.transitions.append("unknown")


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


async def _workspace_toolset(
    tmp_path: Path,
    tool_name: str,
) -> Any:
    capabilities = workspace_capabilities(Workspace.load(tmp_path), (tool_name,))
    toolset = capabilities[0].get_toolset()
    assert toolset is not None
    return toolset


async def test_missing_workspace_file_is_failed_result_not_retry(tmp_path: Path) -> None:
    toolset = await _workspace_toolset(tmp_path, "read_file")
    bridge = _Bridge(True)
    store = InMemoryStepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        trusted_tool_classes=(("read_file", "filesystem.read"),),
    )
    context = _context()
    call = ToolCallPart(
        "read_file",
        {"path": "missing.json"},
        tool_call_id="call",
    )
    definition = ToolDefinition(
        name="read_file",
        capability_id="workspace-filesystem",
    )
    args = {"path": "missing.json"}
    await capability.before_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args=args,
    )
    tools = await toolset.get_tools(context)

    async def handler(validated_args: dict[str, Any]) -> Any:
        return await toolset.call_tool(
            "read_file",
            validated_args,
            context,
            tools["read_file"],
        )

    with pytest.raises(ToolFailed, match="TOOL_EXECUTION_FAILED"):
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args=args,
            handler=handler,
        )

    assert bridge.transitions == ["begin", "fail"]
    effect = await store.get_tool_effect(run_id="run", tool_call_id="call")
    assert effect is not None
    assert effect.status == "failed"


async def test_missing_write_parent_is_failed_result_not_unknown(tmp_path: Path) -> None:
    toolset = await _workspace_toolset(tmp_path, "write_file")
    bridge = _Bridge(False)
    store = InMemoryStepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        trusted_tool_classes=(("write_file", "filesystem.write"),),
    )
    context = _context()
    args = {"path": "worker/personnel_context/report.md", "content": "report"}
    call = ToolCallPart("write_file", args, tool_call_id="call")
    definition = ToolDefinition(
        name="write_file",
        capability_id="workspace-filesystem",
    )
    await capability.before_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args=args,
    )
    tools = await toolset.get_tools(context)

    async def handler(validated_args: dict[str, Any]) -> Any:
        return await toolset.call_tool(
            "write_file",
            validated_args,
            context,
            tools["write_file"],
        )

    with pytest.raises(ToolFailed, match="TOOL_EXECUTION_FAILED"):
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args=args,
            handler=handler,
        )

    assert bridge.transitions == ["begin", "fail"]
    effect = await store.get_tool_effect(run_id="run", tool_call_id="call")
    assert effect is not None
    assert effect.status == "failed"


async def test_repeated_workspace_failures_do_not_consume_tool_retry_budget(
    tmp_path: Path,
) -> None:
    bridge = _Bridge(True)
    store = InMemoryStepStore()
    persistence = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        trusted_tool_classes=(("read_file", "filesystem.read"),),
    )
    workspace = Workspace.load(tmp_path)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        failed_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert all(part.outcome == "failed" for part in failed_returns)
        if not failed_returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_file",
                        {"path": "missing-one.txt"},
                        tool_call_id="call-1",
                    )
                ]
            )
        if len(failed_returns) == 1:
            assert "TOOL_EXECUTION_FAILED" in str(failed_returns[0].content)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_file",
                        {"path": "missing-two.txt"},
                        tool_call_id="call-2",
                    )
                ]
            )
        assert len(failed_returns) == 2
        assert all(
            "TOOL_EXECUTION_FAILED" in str(part.content)
            for part in failed_returns
        )
        return ModelResponse(parts=[TextPart("continued after tool failures")])

    agent = Agent(
        FunctionModel(model),
        capabilities=(
            *workspace_capabilities(workspace, ("read_file",)),
            _RuntimePersistenceBoundary(persistence),
        ),
    )

    result = await agent.run("read the missing files", run_id="run")

    assert result.output == "continued after tool failures"
    assert result.usage.requests == 3
    assert bridge.transitions == ["begin", "fail", "begin", "fail"]
