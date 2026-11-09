#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed sibling tool returns survive native deferred checkpoints."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.core import ApprovalDecision, CommitPolicy, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

if TYPE_CHECKING:
    from pydantic_ai.capabilities import ValidatedToolArgs
    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, StreamFunctionDef
    from pydantic_ai.tools import RunContext, ToolDefinition


class _Binding:
    provider = "test"
    model_identity = "test:parallel-reopen"
    contract = {"version": 3, "provider": provider, "model_identity": model_identity,
                "input_mime_types": [], "settings": {}}

    def __init__(self, stream: "StreamFunctionDef") -> None:
        self.stream = stream

    def materialize(self) -> FunctionModel:
        return FunctionModel(stream_function=self.stream)


class _Approval(AbstractCapability):
    async def before_tool_execute(
        self, ctx: "RunContext[None]", *, call: "ToolCallPart",
        tool_def: "ToolDefinition", args: "ValidatedToolArgs",
    ) -> "ValidatedToolArgs":
        if call.tool_name == "approved_tool" and not ctx.tool_call_approved:
            raise ApprovalRequired()
        return args


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [ApprovalDecision.APPROVE, ApprovalDecision.DENY])
@pytest.mark.parametrize("completed_count,pending_count", [(1, 1), (3, 2)])
@pytest.mark.parametrize("buffered_trace", [False, True])
async def test_completed_parallel_return_survives_reopen(
    tmp_path: Path, decision: ApprovalDecision, completed_count: int, pending_count: int,
    buffered_trace: bool,
) -> None:
    calls = []

    async def stream(messages: "list[ModelMessage]", info: "AgentInfo") -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            yield {
                index: DeltaToolCall(name=name, json_args=json.dumps({"index": index}), tool_call_id=str(index))
                for index, name in enumerate(
                    ["completed_tool"] * completed_count + ["approved_tool"] * pending_count
                )
            }
        else:
            assert {p.tool_call_id for p in returns} == {str(i) for i in range(completed_count + pending_count)}
            for index in range(completed_count):
                assert next(p.content for p in returns if p.tool_call_id == str(index)) == {"fact": str(index)}
            yield "Complete"

    async def completed_tool(ctx: AgentContext[None], index: int) -> dict[str, str]:
        """Return a synthetic immutable fact."""
        calls.append(index)
        return {"fact": str(index)}

    async def approved_tool(ctx: AgentContext[None], index: int) -> str:
        """Run after explicit approval."""
        calls.append(index)
        return "approved"

    features = RuntimeFeatures("parallel-reopen")
    features.model("offline", _Binding(stream))
    features.agent("worker", model="offline", allow_tools=("completed_tool", "approved_tool"))
    features.tool(completed_tool, effect_policy="none")
    features.tool(approved_tool, effect_policy="none")
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    if buffered_trace:
        features.capability(TraceCapability(commit=CommitPolicy.batch(
            buffer=InMemoryCacheBackend(max_bytes=2**22), batch_size=128, flush_interval=60,
        )))
    features.capability(_Approval(), id="approval")
    storage = PersistenceStorage.from_root(tmp_path / "state")
    principal = Principal("tester", "default")
    async with Runtime.open("parallel-reopen", features=features.borrow(), storage=storage) as runtime:
        execution = await runtime.agents.get("worker").start("Run both tools", principal=principal)
        for _ in range(300):
            state = await runtime.executions.inspect(execution.execution_id, principal=principal)
            if state.status.value == "WAITING_DEFERRED":
                break
            await asyncio.sleep(0.02)
        assert state.status.value == "WAITING_DEFERRED"
        approvals = await runtime.approvals.list(execution.execution_id, principal=principal)
        assert len(approvals) == pending_count
    assert sorted(calls) == list(range(completed_count))
    async with Runtime.open("parallel-reopen", features=features.borrow(), storage=storage) as runtime:
        for approval in approvals:
            request = ApprovalDecisionRequest(principal, approval.approval_id, approval.approval_id, decision)
            await runtime.approvals.decide(execution.execution_id, request)
            try:
                await runtime.approvals.decide(execution.execution_id, request)
            except AIError as error:
                # Once the frontier advances, replay is rejected rather than re-executed.
                assert error.code is ErrorCode.APPROVAL_CONFLICT
        execution = await runtime.executions.get(execution.execution_id, principal=principal)
        result = (await execution.wait(timeout_seconds=20)).result
        assert result.status.value == "SUCCEEDED", result
    assert sorted(calls) == list(range(completed_count + (pending_count if decision is ApprovalDecision.APPROVE else 0)))
