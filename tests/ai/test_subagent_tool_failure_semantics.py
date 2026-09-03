#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subagent child failures remain typed through the model adapter."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


async def test_subagent_adapter_returns_child_failure_to_parent_model() -> None:
    details = {
        "phase": "subagent_execution",
        "subagent_id": "child",
        "execution_id": "child-execution",
        "status": "failed",
        "error_code": ErrorCode.MODEL_TIMEOUT.value,
        "safe_error_details": {"phase": "agent_execution"},
    }

    async def delegate(
        ref: SubagentRef,
        task: str,
        *,
        invocation_id: str,
    ) -> "dict[str, object]":
        del ref, task, invocation_id
        raise AIError(ErrorCode.TOOL_EXECUTION_FAILED, safe_details=details)

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

    assert raised.value.message == "subagent execution failed: " + json.dumps(
        details,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
