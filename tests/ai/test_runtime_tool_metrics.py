#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool metrics must describe actual handler execution, never durable replay."""

from __future__ import annotations

from typing import Any

import pytest
from linktools.ai.observe import Observation
from linktools.ai.runtime._capabilities import ToolOperationDecision, _RuntimeStepPersistence
from linktools.ai.runtime._tool_metrics import _ToolMetricContext
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class _StepStore:
    async def record_tool_effect(self, effect: Any) -> None:
        del effect

    async def append_event(self, event: Any) -> None:
        del event

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> Any:
        del run_id, tool_call_id
        return None


class _Bridge:
    def __init__(self, decision: ToolOperationDecision) -> None:
        self.decision = decision
        self.calls: list[str] = []

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        assert replay_safe is self.decision.replay_safe
        self.calls.append("begin")
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del decision, error
        self.calls.append("unknown")

    async def existing_call_ids(self, tool_call_ids: tuple[str, ...]) -> frozenset[str]:
        del tool_call_ids
        return frozenset()


def _context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id="run")


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


async def _capability(
    decision: ToolOperationDecision,
    recorder: _Recorder,
) -> tuple[_RuntimeStepPersistence, _Bridge, RunContext[None], ToolCallPart, ToolDefinition]:
    bridge = _Bridge(decision)
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=_StepStore(),
        agent_name="agent",
        run_id="run",
        tool_metrics=_metric_context(recorder),
    )
    context = _context()
    call = ToolCallPart("tool", {}, tool_call_id="call")
    definition = ToolDefinition(
        name="tool",
        metadata={"linktools.ai.replay_safe": decision.replay_safe},
    )
    await capability.before_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
    )
    return capability, bridge, context, call, definition


async def test_cached_tool_result_does_not_emit_execution_metric() -> None:
    recorder = _Recorder()
    capability, bridge, context, call, definition = await _capability(
        ToolOperationDecision(
            "operation",
            "owner",
            1,
            True,
            cached_result={"cached": True},
            has_cached_result=True,
        ),
        recorder,
    )
    entered = False

    async def handler(_args: dict[str, Any]) -> object:
        nonlocal entered
        entered = True
        return {"live": True}

    result = await capability.wrap_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
        handler=handler,
    )

    assert result == {"cached": True}
    assert entered is False
    assert bridge.calls == ["begin"]
    assert recorder.observations == []


async def test_actual_tool_handler_emits_one_execution_metric() -> None:
    recorder = _Recorder()
    capability, bridge, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, True),
        recorder,
    )
    entered = 0

    async def handler(_args: dict[str, Any]) -> object:
        nonlocal entered
        entered += 1
        return {"ok": True}

    result = await capability.wrap_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
        handler=handler,
    )

    assert result == {"ok": True}
    assert entered == 1
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
    capability, bridge, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, True),
        recorder,
    )
    skipped = {"skipped": True}

    async def handler(_args: dict[str, Any]) -> object:
        raise SkipToolExecution(skipped)

    with pytest.raises(SkipToolExecution) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    assert raised.value.result == skipped
    assert bridge.calls == ["begin", "complete"]
    assert len(recorder.observations) == 1
    observation = recorder.observations[0]
    assert observation.kind == "linktools.tool.execution"
    assert observation.status == "SUCCEEDED"
    assert observation.error_code is None
