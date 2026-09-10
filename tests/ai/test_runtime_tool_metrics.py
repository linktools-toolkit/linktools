#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool metrics must describe actual handler execution, never durable replay."""

from __future__ import annotations

from typing import Any

import pytest
from linktools.ai.observe import Observation
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.runtime._tool_metrics import _ToolMetricContext
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _Bridge:
    def __init__(self, decision: ToolOperationDecision) -> None:
        self.decision = decision
        self.calls: list[str] = []

    async def begin(
        self,
        ctx: RunContext[None],
        call: Any,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        assert replay_safe is self.decision.replay_safe
        self.calls.append("begin")
        return self.decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(
        self,
        decision: ToolOperationDecision,
        error: BaseException,
    ) -> None:
        del decision, error
        self.calls.append("unknown")


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


def _metric_context(recorder: _Recorder) -> _ToolMetricContext:
    return _ToolMetricContext(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id="session",
        step_run_id="run",
        agent_id="agent",
    )


async def _boundary(
    decision: ToolOperationDecision,
    recorder: _Recorder,
    handler: Any,
) -> tuple[RuntimeToolBoundaryToolset, _Bridge, RunContext[None], Any]:
    async def tool() -> object:
        return await handler()

    bridge = _Bridge(decision)
    raw = FunctionToolset([tool])
    boundary = RuntimeToolBoundaryToolset(
        (raw,),
        {
            "tool": ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="replay_safe",
                tool_class="business",
            )
        },
        id="boundary",
        tool_operations=bridge,
        tool_metrics=_metric_context(recorder),
    )
    context = _context()
    tools = await boundary.get_tools(context)
    return boundary, bridge, context, tools["tool"]


async def test_cached_tool_result_does_not_emit_execution_metric() -> None:
    recorder = _Recorder()
    entered = False

    async def handler() -> object:
        nonlocal entered
        entered = True
        return {"live": True}

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision(
            "operation",
            "owner",
            1,
            True,
            cached_result={"cached": True},
            has_cached_result=True,
        ),
        recorder,
        handler,
    )

    result = await boundary.call_tool("tool", {}, context, tool)

    assert result == {"cached": True}
    assert entered is False
    assert bridge.calls == ["begin"]
    assert recorder.observations == []


async def test_actual_tool_handler_emits_one_execution_metric() -> None:
    recorder = _Recorder()

    async def handler() -> object:
        return {"ok": True}

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True),
        recorder,
        handler,
    )

    result = await boundary.call_tool("tool", {}, context, tool)

    assert result == {"ok": True}
    assert bridge.calls == ["begin", "complete"]
    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.tool.execution"
    assert observation.status == "SUCCEEDED"
    assert observation.error_code is None
    assert observation.dimensions == {"agent_id": "agent", "tool_name": "tool"}
    assert dict(observation.correlation) == {
        "execution_id": "execution",
        "session_id": "session",
        "step_run_id": "run",
        "tool_call_id": "call",
    }
    assert len(observation.measurements) == 1
    assert observation.measurements[0].name == "latency_ns"
    assert observation.measurements[0].value >= 0


async def test_skip_tool_execution_emits_success_metric_and_durable_completion() -> None:
    recorder = _Recorder()

    async def handler() -> object:
        raise SkipToolExecution({"skipped": True})

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True),
        recorder,
        handler,
    )

    with pytest.raises(SkipToolExecution) as raised:
        await boundary.call_tool("tool", {}, context, tool)

    assert raised.value.result == {"skipped": True}
    assert bridge.calls == ["begin", "complete"]
    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.tool.execution"
    assert observation.status == "SUCCEEDED"
    assert observation.error_code is None
