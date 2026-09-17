#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent child failures remain typed through the model adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from linktools.ai.capability import SubagentCapability, ToolCallFailed, ToolCallRetry
from linktools.ai.core import ExecutionStatus, Principal, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionResult
from linktools.ai.spec import SubagentRef
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    ("status", "error_code"),
    (
        (ExecutionStatus.FAILED, ErrorCode.MODEL_TIMEOUT.value),
        (ExecutionStatus.CANCELLED, ErrorCode.EXECUTION_CANCELLED.value),
    ),
)
async def test_terminal_child_becomes_typed_tool_failure(
    status: ExecutionStatus,
    error_code: str,
) -> None:
    result = ExecutionResult(
        "child-execution",
        status,
        None,
        None,
        UsageMetrics(),
        error_code,
        {"phase": "agent_execution"},
    )
    execution = SimpleNamespace(
        replay_subagent=AsyncMock(return_value=ExecutionHandle(result.execution_id)),
        wait=AsyncMock(return_value=result),
    )
    dispatcher = SubagentDispatcher(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        execution,  # type: ignore[arg-type]
    )

    with pytest.raises(AIError) as raised:
        await dispatcher.dispatch(
            parent_execution_id="parent",
            root_execution_id="root",
            memory_scope=None,
            principal=Principal("principal", "tenant", "service"),
            ref=SubagentRef("agent", "child"),
            mode="run",
            user_prompt="do work",
            invocation_id="call",
        )

    assert raised.value.code is ErrorCode.TOOL_EXECUTION_FAILED
    assert raised.value.safe_details == {
        "phase": "subagent_execution",
        "subagent_id": "child",
        "execution_id": "child-execution",
        "status": status.value,
        "safe_error_details": {"phase": "agent_execution"},
        "error_code": error_code,
    }


async def test_existing_child_replay_does_not_require_current_definition() -> None:
    result = ExecutionResult(
        "child-execution",
        ExecutionStatus.SUCCEEDED,
        {"ok": True},
        "f" * 64,
        UsageMetrics(),
    )
    execution = SimpleNamespace(
        replay_subagent=AsyncMock(return_value=ExecutionHandle(result.execution_id)),
        wait=AsyncMock(return_value=result),
    )
    dispatcher = SubagentDispatcher(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        execution,  # type: ignore[arg-type]
    )

    value = await dispatcher.dispatch(
        parent_execution_id="parent",
        root_execution_id="root",
        memory_scope="memory",
        principal=Principal("principal", "tenant", "service"),
        ref=SubagentRef("agent", "child"),
        mode="run",
        user_prompt="do work",
        invocation_id="persisted-call",
    )

    assert value["execution_id"] == "child-execution"
    assert value["status"] == ExecutionStatus.SUCCEEDED.value
    execution.replay_subagent.assert_awaited_once()


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
        tool_name="delegate_task",
    )


async def test_subagent_adapter_returns_child_failure_to_parent_model() -> None:
    details = {
        "phase": "subagent_execution",
        "subagent_id": "child",
        "execution_id": "child-execution",
        "status": ExecutionStatus.FAILED.value,
        "error_code": ErrorCode.MODEL_TIMEOUT.value,
        "safe_error_details": {"phase": "agent_execution"},
    }

    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> "dict[str, object]":
        del ref, task, invocation_id
        assert files == ()
        raise AIError(ErrorCode.TOOL_EXECUTION_FAILED, safe_details=details)

    capability = SubagentCapability(
        (SubagentRef("agent", "child"),),
        delegate,
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallFailed) as raised:
        await toolset.call_tool(
            "delegate_task",
            {"subagent_id": "child", "task": "do work"},
            context,
            tools["delegate_task"],
        )

    assert raised.value.message == (
        "Subagent 'child' failed with MODEL_TIMEOUT and produced no result. Continue "
        "from the parent context using this failure reason."
    )


async def test_subagent_tool_retries_its_own_oversized_task() -> None:
    called = False

    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> "dict[str, object]":
        nonlocal called
        del ref, task, files, invocation_id
        called = True
        return {}

    capability = SubagentCapability(
        (SubagentRef("agent", "child"),),
        delegate,
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallRetry, match="exceeds the allowed prompt size"):
        await toolset.call_tool(
            "delegate_task",
            {"subagent_id": "child", "task": "x" * (1024 * 1024 + 1)},
            context,
            tools["delegate_task"],
        )

    assert called is False


async def test_unknown_subagent_id_gives_model_a_selection_action() -> None:
    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> "dict[str, object]":
        del ref, task, files, invocation_id
        raise AssertionError("delegate should not be called")

    capability = SubagentCapability(
        (SubagentRef("agent", "child"),),
        delegate,
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallRetry, match="Call list_subagents"):
        await toolset.call_tool(
            "delegate_task",
            {"subagent_id": "missing", "task": "do work"},
            context,
            tools["delegate_task"],
        )


async def test_subagent_downstream_prompt_error_is_not_reclassified() -> None:
    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> "dict[str, object]":
        del ref, task, files, invocation_id
        raise AIError(ErrorCode.PROMPT_TOO_LARGE)

    capability = SubagentCapability(
        (SubagentRef("agent", "child"),),
        delegate,
    )
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(AIError) as raised:
        await toolset.call_tool(
            "delegate_task",
            {"subagent_id": "child", "task": "do work"},
            context,
            tools["delegate_task"],
        )

    assert raised.value.code is ErrorCode.PROMPT_TOO_LARGE
