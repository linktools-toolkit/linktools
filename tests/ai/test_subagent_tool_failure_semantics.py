#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent failures remain child truth while becoming parent tool failures."""

import pytest
from linktools.ai.capability import SubagentCapability
from linktools.ai.core import ExecutionStatus, Principal, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime._subagent_adapter import _PydanticSubagentCapability
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionResult
from linktools.ai.spec import SubagentRef
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


class _TerminalExecution:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result

    async def replay_subagent(
        self,
        *,
        agent_id: str,
        user_prompt: str,
        principal: Principal,
        idempotency_key: str,
        memory_scope: "str | None",
        mode: str,
        parent_execution_id: str,
        root_execution_id: str,
    ) -> ExecutionHandle:
        del user_prompt, principal, idempotency_key, memory_scope, mode
        assert agent_id == "child"
        assert parent_execution_id == "parent"
        assert root_execution_id == "root"
        return ExecutionHandle(self._result.execution_id)

    async def wait(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        del principal
        assert execution_id == self._result.execution_id
        return self._result


def _dispatcher(result: ExecutionResult) -> SubagentDispatcher:
    dispatcher = object.__new__(SubagentDispatcher)
    dispatcher._execution = _TerminalExecution(result)
    dispatcher._background_failures = {}
    dispatcher._detached_tasks = set()
    return dispatcher


async def _dispatch(result: ExecutionResult) -> "dict[str, object]":
    return await _dispatcher(result).dispatch(
        parent_execution_id="parent",
        root_execution_id="root",
        memory_scope=None,
        principal=Principal("principal", "tenant", "service"),
        ref=SubagentRef("agent", "child"),
        mode="run",
        user_prompt="do work",
        invocation_id="call",
    )


async def test_failed_child_becomes_typed_tool_failure() -> None:
    result = ExecutionResult(
        "child-execution",
        ExecutionStatus.FAILED,
        None,
        None,
        UsageMetrics(),
        ErrorCode.MODEL_TIMEOUT.value,
        {"phase": "agent_execution"},
    )

    with pytest.raises(AIError) as raised:
        await _dispatch(result)

    assert raised.value.code is ErrorCode.TOOL_EXECUTION_FAILED
    assert raised.value.safe_details == {
        "phase": "subagent_execution",
        "subagent_id": "child",
        "execution_id": "child-execution",
        "status": "FAILED",
        "error_code": ErrorCode.MODEL_TIMEOUT.value,
        "safe_error_details": {"phase": "agent_execution"},
    }


async def test_cancelled_child_becomes_typed_tool_failure() -> None:
    result = ExecutionResult(
        "child-execution",
        ExecutionStatus.CANCELLED,
        None,
        None,
        UsageMetrics(),
        ErrorCode.EXECUTION_CANCELLED.value,
    )

    with pytest.raises(AIError) as raised:
        await _dispatch(result)

    assert raised.value.code is ErrorCode.TOOL_EXECUTION_FAILED
    assert raised.value.safe_details["status"] == "CANCELLED"
    assert raised.value.safe_details["error_code"] == ErrorCode.EXECUTION_CANCELLED.value


async def test_subagent_adapter_returns_child_failure_to_parent_model() -> None:
    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        invocation_id: str,
    ) -> "dict[str, object]":
        del ref, task, invocation_id
        raise AIError(
            ErrorCode.TOOL_EXECUTION_FAILED,
            safe_details={
                "phase": "subagent_execution",
                "subagent_id": "child",
                "execution_id": "child-execution",
                "status": "FAILED",
                "error_code": ErrorCode.MODEL_TIMEOUT.value,
                "safe_error_details": {"phase": "agent_execution"},
            },
        )

    capability = _PydanticSubagentCapability(
        SubagentCapability((SubagentRef("agent", "child"),), delegate)  # type: ignore[arg-type]
    )
    toolset = capability.get_toolset()
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
        tool_name="delegate_task",
    )
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolFailed) as raised:
        await toolset.call_tool(
            "delegate_task",
            {"subagent_id": "child", "task": "do work"},
            context,
            tools["delegate_task"],
        )

    assert "subagent execution failed" in raised.value.message
    assert ErrorCode.MODEL_TIMEOUT.value in raised.value.message


async def test_unknown_subagent_is_tool_failure_not_retry() -> None:
    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        invocation_id: str,
    ) -> "dict[str, object]":
        del ref, task, invocation_id
        return {}

    capability = _PydanticSubagentCapability(
        SubagentCapability((SubagentRef("agent", "child"),), delegate)  # type: ignore[arg-type]
    )
    toolset = capability.get_toolset()
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
        tool_name="delegate_task",
    )
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolFailed) as raised:
        await toolset.call_tool(
            "delegate_task",
            {"subagent_id": "missing", "task": "do work"},
            context,
            tools["delegate_task"],
        )

    assert "call list_subagents" in raised.value.message
