#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fault coverage for tool terminal ownership and local task waiters."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest
from linktools.ai.agent._capabilities import (
    ToolOperationDecision,
    _RuntimeStepPersistence,
)
from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task._graph import (
    TaskGraph,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)
from linktools.ai.task._local import LocalTaskGraphLauncher, TaskNodeRunResult
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.asyncio


@dataclass
class _Effect:
    run_id: str
    tool_call_id: str
    status: str
    effect_summary: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    idempotency_key: str | None = None


class _StepStore:
    def __init__(self) -> None:
        self.effects: list[_Effect] = []

    async def record_tool_effect(self, effect: Any) -> None:
        self.effects.append(
            _Effect(effect.run_id, effect.tool_call_id, effect.status, effect.effect_summary)
        )

    async def append_event(self, event: Any) -> None:
        del event

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> Any:
        for effect in reversed(self.effects):
            if effect.run_id == run_id and effect.tool_call_id == tool_call_id:
                return effect
        return None


class _ToolBridge:
    def __init__(self, decision: ToolOperationDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str]] = []

    async def begin(self, *args: Any) -> ToolOperationDecision:
        del args
        self.calls.append(("begin", ""))
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        self.calls.append(("renew", decision.operation_id))
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> None:
        del result
        self.calls.append(("complete", decision.operation_id))

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del error
        self.calls.append(("fail", decision.operation_id))

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del error
        self.calls.append(("unknown", decision.operation_id))


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


def _call() -> ToolCallPart:
    return ToolCallPart("tool", {}, tool_call_id="call")


def _definition(replay_safe: bool) -> ToolDefinition:
    return ToolDefinition(
        name="tool",
        metadata={"linktools.ai.replay_safe": replay_safe},
    )


async def _capability(
    decision: ToolOperationDecision,
) -> tuple[_RuntimeStepPersistence, _ToolBridge, _StepStore, RunContext[None], ToolCallPart, ToolDefinition]:
    bridge = _ToolBridge(decision)
    store = _StepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
    )
    context = _context()
    call = _call()
    definition = _definition(decision.replay_safe)
    await capability.before_tool_execute(context, call=call, tool_def=definition, args={})
    return capability, bridge, store, context, call, definition


async def test_tool_success_terminalizes_once_before_effect_completion() -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, True)
    )

    async def handler(_args: dict[str, Any]) -> dict[str, bool]:
        return {"ok": True}

    result = await capability.wrap_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
        handler=handler,
    )
    await capability.after_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
        result=result,
    )

    assert [kind for kind, _ in bridge.calls] == ["begin", "complete"]
    assert [effect.status for effect in store.effects] == ["started", "completed"]


@pytest.mark.parametrize("replay_safe", [True, False])
async def test_retry_error_uses_replay_safety_not_exception_type(replay_safe: bool) -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, replay_safe)
    )

    async def handler(_args: dict[str, Any]) -> None:
        raise ModelRetry("retry")

    with pytest.raises((ModelRetry, AIError)) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    if replay_safe:
        assert isinstance(raised.value, ModelRetry)
        assert [kind for kind, _ in bridge.calls] == ["begin", "fail"]
        assert [effect.status for effect in store.effects] == ["started", "failed"]
    else:
        assert isinstance(raised.value, AIError)
        assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        with pytest.raises(AIError):
            await capability.on_tool_execute_error(
                context,
                call=call,
                tool_def=definition,
                args={},
                error=raised.value,
            )
        assert [kind for kind, _ in bridge.calls] == ["begin", "unknown"]
        assert [effect.status for effect in store.effects] == ["started"]


async def test_generic_replay_safe_failure_is_failed_by_error_hook_once() -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, True)
    )

    async def handler(_args: dict[str, Any]) -> None:
        raise ValueError("failure")

    with pytest.raises(ValueError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )
    with pytest.raises(ValueError):
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=raised.value,
        )

    assert [kind for kind, _ in bridge.calls] == ["begin", "fail"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]
    assert not capability._calls


async def test_generic_replay_unsafe_failure_preserves_started_effect() -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, False)
    )

    async def handler(_args: dict[str, Any]) -> None:
        raise ValueError("unknown effect")

    with pytest.raises(ValueError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )
    with pytest.raises(ValueError):
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=raised.value,
        )

    assert [kind for kind, _ in bridge.calls] == ["begin", "unknown"]
    assert [effect.status for effect in store.effects] == ["started"]


async def test_cached_failed_replays_effect_without_terminal_mutation() -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision(
            "operation",
            "owner",
            1,
            True,
            cached_error=AIError(ErrorCode.EXECUTION_FAILED),
        )
    )
    handler_calls = 0

    async def handler(_args: dict[str, Any]) -> None:
        nonlocal handler_calls
        handler_calls += 1

    with pytest.raises(AIError) as raised:
        await capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )

    with pytest.raises(AIError):
        await capability.on_tool_execute_error(
            context,
            call=call,
            tool_def=definition,
            args={},
            error=raised.value,
        )

    assert raised.value.code is ErrorCode.EXECUTION_FAILED
    assert handler_calls == 0
    assert [kind for kind, _ in bridge.calls] == ["begin"]
    assert [effect.status for effect in store.effects] == ["started", "failed"]


async def test_cached_completed_replays_without_terminal_mutation() -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision(
            "operation",
            "owner",
            1,
            True,
            cached_result={"ok": True},
            has_cached_result=True,
        )
    )
    handler_calls = 0

    async def handler(_args: dict[str, Any]) -> None:
        nonlocal handler_calls
        handler_calls += 1

    result = await capability.wrap_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
        handler=handler,
    )
    await capability.after_tool_execute(
        context,
        call=call,
        tool_def=definition,
        args={},
        result=result,
    )

    assert result == {"ok": True}
    assert handler_calls == 0
    assert [kind for kind, _ in bridge.calls] == ["begin"]
    assert [effect.status for effect in store.effects] == ["started", "completed"]


@pytest.mark.parametrize("replay_safe", [True, False])
async def test_cancellation_preserves_started_by_replay_safety(replay_safe: bool) -> None:
    capability, bridge, store, context, call, definition = await _capability(
        ToolOperationDecision("operation", "owner", 1, replay_safe)
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_args: dict[str, Any]) -> None:
        entered.set()
        await release.wait()

    task = asyncio.create_task(
        capability.wrap_tool_execute(
            context,
            call=call,
            tool_def=definition,
            args={},
            handler=handler,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [effect.status for effect in store.effects] == ["started"]
    assert [kind for kind, _ in bridge.calls] == ["begin"] if replay_safe else ["begin", "unknown"]
    assert not capability._calls


class _TaskRepository:
    def __init__(self, status: TaskStatus = TaskStatus.RUNNING) -> None:
        self.status = status
        self.failure: BaseException | None = None

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        del graph_id, tenant_id
        if self.failure is not None:
            raise self.failure
        return TaskGraphView("graph", self.status, ())

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        del graph_id, tenant_id
        return TaskGraphView("graph", self.status, ())

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[object, ...]:
        del graph_id, tenant_id
        return ()

    async def claim(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("terminal graph must not claim a node")

    async def renew(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("terminal graph must not renew a node")

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("terminal graph must not complete a node")

    async def fail(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("terminal graph must not fail a node")


class _TaskRunner:
    async def run(self, *args: Any, **kwargs: Any) -> TaskNodeRunResult:
        del args, kwargs
        return TaskNodeRunResult("0" * 64)


def _task_request() -> TaskGraphRequest:
    return TaskGraphRequest(
        TaskGraph("graph", (TaskNode("node"),)),
        Principal("user", "tenant"),
        idempotency_key="request",
    )


async def test_local_scheduler_terminal_exit_wakes_waiter_and_cleans_entry() -> None:
    launcher = LocalTaskGraphLauncher(
        _TaskRepository(TaskStatus.SUCCEEDED),
        _TaskRunner(),
        owner="launcher",
    )
    await launcher.start(_task_request())
    waiter = asyncio.create_task(
        launcher.wait_graph_activity("graph", tenant_id="tenant")
    )
    await asyncio.wait_for(waiter, timeout=1)
    await asyncio.sleep(0)
    assert not launcher._graphs
    await launcher.shutdown()


async def test_local_scheduler_failure_wakes_waiter_with_infrastructure_error() -> None:
    repository = _TaskRepository()
    repository.failure = RuntimeError("scheduler failure")
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    await launcher.start(_task_request())
    await asyncio.sleep(0)
    assert launcher.owns_graph("graph", tenant_id="tenant") is True
    with pytest.raises(AIError) as raised:
        await asyncio.wait_for(
            launcher.wait_graph_activity("graph", tenant_id="tenant"),
            timeout=1,
        )
    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
    await asyncio.sleep(0)
    assert not launcher._graphs
    await launcher.shutdown()


async def test_launcher_shutdown_wakes_active_waiter() -> None:
    launcher = LocalTaskGraphLauncher(_TaskRepository(), _TaskRunner(), owner="launcher")
    await launcher.start(_task_request())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(
        launcher.wait_graph_activity("graph", tenant_id="tenant")
    )
    await asyncio.sleep(0)
    await launcher.shutdown()
    await asyncio.wait_for(waiter, timeout=1)
