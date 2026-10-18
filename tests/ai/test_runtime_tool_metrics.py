#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool metrics must describe actual handler execution, never durable replay."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from linktools.ai.capability import ToolCallFailed, ToolCallRetry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Observation
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    BoundaryToolset,
)
from linktools.ai.runtime._tool_metrics import (
    ToolMetricsCapability,
    ToolMetricContext,
)
from ._runtime_test_helpers import tool_run_context, tool_with_metadata
from pydantic_ai.exceptions import CallDeferred, SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition

pytestmark = pytest.mark.asyncio


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    async def try_record(self, observation: Observation) -> bool:
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

    async def defer(self, decision: ToolOperationDecision) -> bool:
        del decision
        self.calls.append("defer")
        return False

    async def unknown(
        self,
        decision: ToolOperationDecision,
        error: BaseException,
    ) -> None:
        del decision, error
        self.calls.append("unknown")


def _metric_context(recorder: _Recorder) -> ToolMetricContext:
    return ToolMetricContext(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id="session",
        agent_run_id="run",
        agent_id="agent",
    )


async def _boundary(
    decision: ToolOperationDecision,
    recorder: _Recorder,
    handler: Any,
) -> tuple[BoundaryToolset, _Bridge, RunContext[None], Any]:
    async def tool() -> object:
        return await handler()

    bridge = _Bridge(decision)
    descriptor = ManagedToolDescriptor(
        effect_owner="tool_operation",
        effect_policy="replay_safe",
        tool_class="business",
    )
    raw = FunctionToolset([tool_with_metadata(tool, descriptor)])
    boundary = BoundaryToolset(
        (raw,),
        {"tool": descriptor},
        id="boundary",
        tool_operations=bridge,
        tool_metrics=_metric_context(recorder),
    )
    context = tool_run_context()
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
        "agent_run_id": "run",
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


@pytest.mark.parametrize(
    ("signal", "expected_code"),
    (
        (ToolCallRetry("retry"), ErrorCode.TOOL_RETRY_REQUIRED.value),
        (ToolCallFailed("failed"), ErrorCode.TOOL_EXECUTION_FAILED.value),
    ),
)
async def test_tool_signal_metric_uses_stable_failure_code(
    signal: ToolCallRetry | ToolCallFailed,
    expected_code: str,
) -> None:
    recorder = _Recorder()

    async def handler() -> object:
        raise signal

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True),
        recorder,
        handler,
    )

    with pytest.raises(type(signal)):
        await boundary.call_tool("tool", {}, context, tool)

    assert bridge.calls == ["begin", "fail"]
    assert recorder.observations[0].error_code == expected_code


@pytest.mark.parametrize(
    ("signal", "expected_code"),
    (
        (ToolCallRetry("retry"), ErrorCode.TOOL_RETRY_REQUIRED.value),
        (ToolCallFailed("failed"), ErrorCode.TOOL_EXECUTION_FAILED.value),
    ),
)
async def test_capability_tool_signal_is_observed_before_control_conversion(
    signal: ToolCallRetry | ToolCallFailed,
    expected_code: str,
) -> None:
    recorder = _Recorder()
    capability = ToolMetricsCapability(_metric_context(recorder))
    call = ToolCallPart("capability_tool", args={}, tool_call_id="call")
    tool_def = ToolDefinition(name="capability_tool")

    async def handler(_: dict[str, Any]) -> object:
        raise signal

    with pytest.raises(type(signal)):
        await capability.wrap_tool_execute(
            tool_run_context(),
            call=call,
            tool_def=tool_def,
            args={},
            handler=handler,
        )

    assert recorder.observations[0].error_code == expected_code


async def test_tool_metric_preserves_ai_error_and_defaults_other_errors() -> None:
    from linktools.ai.runtime._tool_metrics import _tool_error_code

    assert (
        _tool_error_code(AIError(ErrorCode.SANDBOX_BUSY))
        == ErrorCode.SANDBOX_BUSY.value
    )
    assert (
        _tool_error_code(RuntimeError("failed"))
        == ErrorCode.TOOL_EXECUTION_FAILED.value
    )


@pytest.mark.parametrize(
    ("outcome", "settlement", "status"),
    (
        (None, "complete", "SUCCEEDED"),
        (ToolCallFailed("failure"), "fail", "FAILED"),
        (CallDeferred(), "defer", "DEFERRED"),
        (RuntimeError("unknown effect"), "unknown", "FAILED"),
    ),
)
async def test_tool_metric_cancellation_preserves_settled_effect(
    outcome: BaseException | None,
    settlement: str,
    status: str,
) -> None:
    entered = asyncio.Event()
    attempted: list[Observation] = []

    class BlockingRecorder(_Recorder):
        async def try_record(self, observation: Observation) -> bool:
            attempted.append(observation)
            entered.set()
            await asyncio.Event().wait()
            return True

    async def handler() -> object:
        if outcome is not None:
            raise outcome
        return {"done": True}

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True),
        BlockingRecorder(),
        handler,
    )
    task = asyncio.create_task(boundary.call_tool("tool", {}, context, tool))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert bridge.calls == ["begin", settlement]
    assert attempted[0].status == status
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge.calls == ["begin", settlement]


async def test_tool_metric_failure_preserves_original_leaf_error() -> None:
    class FailingRecorder(_Recorder):
        async def try_record(self, observation: Observation) -> bool:
            raise OSError("metric sink unavailable")

    error = ToolCallFailed("original leaf failure")

    async def handler() -> object:
        raise error

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True),
        FailingRecorder(),
        handler,
    )
    with pytest.raises(ToolCallFailed) as raised:
        await boundary.call_tool("tool", {}, context, tool)
    assert raised.value is error
    assert bridge.calls == ["begin", "fail"]


async def test_original_tool_cancellation_settles_before_interruptible_metric() -> None:
    leaf_entered = asyncio.Event()
    metric_entered = asyncio.Event()

    class BlockingRecorder(_Recorder):
        async def try_record(self, observation: Observation) -> bool:
            assert observation.status == "CANCELLED"
            metric_entered.set()
            await asyncio.Event().wait()
            return True

    async def handler() -> object:
        leaf_entered.set()
        await asyncio.Event().wait()

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True),
        BlockingRecorder(),
        handler,
    )
    task = asyncio.create_task(boundary.call_tool("tool", {}, context, tool))
    await asyncio.wait_for(leaf_entered.wait(), timeout=1)
    task.cancel()
    await asyncio.wait_for(metric_entered.wait(), timeout=1)
    assert bridge.calls == ["begin", "unknown"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge.calls == ["begin", "unknown"]


async def test_tool_latency_excludes_durable_settlement_and_metric_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime import _tool_metrics

    now = 100
    class Recorder(_Recorder):
        async def try_record(self, observation: Observation) -> bool:
            nonlocal now
            now = 20_000
            return await super().try_record(observation)

    recorder = Recorder()

    def monotonic_ns() -> int:
        return now

    monkeypatch.setattr(_tool_metrics, "monotonic_ns", monotonic_ns)

    async def handler() -> object:
        nonlocal now
        now = 150
        return "done"

    boundary, bridge, context, tool = await _boundary(
        ToolOperationDecision("operation", "owner", 1, True), recorder, handler,
    )
    original_complete = bridge.complete

    async def complete(decision: ToolOperationDecision, result: object) -> bool:
        nonlocal now
        now = 10_000
        return await original_complete(decision, result)

    bridge.complete = complete
    assert await boundary.call_tool("tool", {}, context, tool) == "done"
    assert recorder.observations[0].measurements[0].value == 50
