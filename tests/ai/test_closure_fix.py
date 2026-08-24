#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fault coverage for tool terminal ownership and local task waiters."""

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import pytest
from linktools.ai.agent._capabilities import (
    ToolOperationDecision,
    _RuntimeStepPersistence,
)
from linktools.ai.core import Principal, TaskStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.runtime.state import ToolOperationAdmission
from linktools.ai.storage import PayloadPolicy, StoredPayload
from linktools.ai.task._graph import (
    CancelGraphRequest,
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


class _OperationRepository:
    def __init__(self) -> None:
        self.request: Any = None

    async def admit(self, request: Any) -> ToolOperationRecord:
        self.request = request
        now = datetime.now(timezone.utc)
        return ToolOperationRecord(
            tool_operation_id=request.tool_operation_id,
            tenant_id=request.tenant_id,
            step_run_id=request.step_run_id,
            tool_call_id=request.tool_call_id,
            idempotency_key_digest=request.idempotency_key_digest,
            tool_name=request.tool_name,
            arguments_digest=request.arguments_digest,
            binding_fingerprint=request.binding_fingerprint,
            replay_safe=request.replay_safe,
            status=ToolOperationStatus.CLAIMED,
            owner=request.owner,
            fence=1,
            lease_expires_at=now,
            error_code=None,
            created_at=now,
            updated_at=now,
        )


class _TerminalRepository:
    def __init__(self) -> None:
        self.record: ToolOperationRecord | None = None

    async def get_operation(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord | None:
        if self.record is None:
            return None
        if self.record.tool_operation_id != tool_operation_id:
            return None
        if self.record.tenant_id != tenant_id:
            return None
        return self.record


class _TerminalCommands:
    def __init__(self, repository: _TerminalRepository) -> None:
        self._repository = repository
        self.calls: list[tuple[str, bool, bool]] = []

    async def commit_tool_admission(
        self,
        request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        now = datetime.now(timezone.utc)
        self._repository.record = ToolOperationRecord(
            tool_operation_id=request.tool_operation_id,
            tenant_id=request.tenant_id,
            step_run_id=request.step_run_id,
            tool_call_id=request.tool_call_id,
            idempotency_key_digest=request.idempotency_key_digest,
            tool_name=request.tool_name,
            arguments_digest=request.arguments_digest,
            binding_fingerprint=request.binding_fingerprint,
            replay_safe=request.replay_safe,
            status=ToolOperationStatus.CLAIMED,
            owner=request.owner,
            fence=1,
            lease_expires_at=None,
            error_code=None,
            created_at=now,
            updated_at=now,
        )
        return self._repository.record

    async def commit_tool_terminal(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> ToolOperationRecord:
        record = await self._repository.get_operation(
            tool_operation_id,
            tenant_id=tenant_id,
        )
        assert record is not None
        assert record.owner == owner
        assert record.fence == fence
        self.calls.append(
            (tool_operation_id, result_payload is not None, error_payload is not None)
        )
        status = (
            ToolOperationStatus.COMPLETED
            if result_payload is not None
            else ToolOperationStatus.FAILED
        )
        self._repository.record = replace(
            record,
            status=status,
            owner=None,
            lease_expires_at=None,
            error_code=error_code,
            result_payload=result_payload,
            error_payload=error_payload,
            updated_at=datetime.now(timezone.utc),
        )
        return self._repository.record


def _context(run_id: str = "run") -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id=run_id,
    )


def _call(tool_call_id: str = "call") -> ToolCallPart:
    return ToolCallPart("tool", {}, tool_call_id=tool_call_id)


def _definition(replay_safe: bool) -> ToolDefinition:
    return ToolDefinition(
        name="tool",
        metadata={"linktools.ai.replay_safe": replay_safe},
    )


async def test_tool_operation_uses_runtime_step_id_for_admission() -> None:
    repository = _OperationRepository()
    bridge = RuntimeToolOperationBridge(
        repository,
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="runtime-step",
        binding_fingerprint="binding",
        owner="owner",
        payload_policy=PayloadPolicy(),
    )

    await bridge.begin(_context(), _call(), _definition(replay_safe=True), {})

    assert repository.request.step_run_id == "runtime-step"


async def test_tool_terminal_bridge_only_commits_operation_state() -> None:
    repository = _TerminalRepository()
    commands = _TerminalCommands(repository)
    bridge = RuntimeToolOperationBridge(
        repository,
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="runtime-step",
        binding_fingerprint="binding",
        owner="owner",
        payload_policy=PayloadPolicy(),
        terminal_commands=commands,
    )

    success = await bridge.begin(
        _context(),
        _call("success-call"),
        _definition(replay_safe=True),
        {},
    )
    await bridge.complete(success, {"ok": True})
    completed = repository.record
    assert completed is not None
    assert completed.status is ToolOperationStatus.COMPLETED

    failed = await bridge.begin(
        _context(),
        _call("failed-call"),
        _definition(replay_safe=True),
        {"failed": True},
    )
    await bridge.fail(failed, ValueError("failure"))
    assert repository.record is not None
    assert repository.record.status is ToolOperationStatus.FAILED
    assert len(commands.calls) == 2
    assert all(result or error for _, result, error in commands.calls)


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
        self.reconcile_gate: asyncio.Event | None = None

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        del graph_id, tenant_id
        if self.reconcile_gate is not None:
            gate = self.reconcile_gate
            self.reconcile_gate = None
            await gate.wait()
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
    await asyncio.sleep(0)
    assert launcher.owns_graph("graph", tenant_id="tenant") is True
    with pytest.raises(AIError) as raised:
        await asyncio.wait_for(
            launcher.wait_graph_activity("graph", tenant_id="tenant"),
            timeout=1,
        )
    assert raised.value.code is ErrorCode.INTERNAL_ERROR
    with pytest.raises(AIError) as late_raised:
        await asyncio.wait_for(
            launcher.wait_graph_activity("graph", tenant_id="tenant"),
            timeout=1,
        )
    assert late_raised.value.code is ErrorCode.INTERNAL_ERROR
    assert repository.status is TaskStatus.RUNNING
    assert launcher._graphs
    await launcher.shutdown()


async def test_local_scheduler_failure_wakes_existing_waiter() -> None:
    repository = _TaskRepository()
    gate = asyncio.Event()
    repository.reconcile_gate = gate
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    await launcher.start(_task_request())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(
        launcher.wait_graph_activity("graph", tenant_id="tenant")
    )
    await asyncio.sleep(0)

    repository.failure = RuntimeError("scheduler failure")
    gate.set()

    with pytest.raises(AIError) as raised:
        await asyncio.wait_for(waiter, timeout=1)
    assert raised.value.code is ErrorCode.INTERNAL_ERROR
    assert launcher._graphs
    await launcher.shutdown()


async def test_local_scheduler_rearms_after_retained_failure() -> None:
    repository = _TaskRepository()
    repository.failure = RuntimeError("scheduler failure")
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    request = _task_request()
    await launcher.start(request)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    failed_run = launcher._graphs[("tenant", "graph")]

    repository.failure = None
    await launcher.start(request)
    replacement = launcher._graphs[("tenant", "graph")]

    assert replacement is not failed_run
    assert replacement.failure is None
    assert replacement.task is not None
    await launcher.shutdown()
    assert not launcher._graphs


async def test_launcher_shutdown_clears_retained_failure() -> None:
    repository = _TaskRepository()
    repository.failure = RuntimeError("scheduler failure")
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    await launcher.start(_task_request())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert launcher._graphs

    await launcher.shutdown()

    assert not launcher._graphs


async def test_launcher_cancel_clears_retained_failure() -> None:
    repository = _TaskRepository()
    repository.failure = RuntimeError("scheduler failure")
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    await launcher.start(_task_request())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert launcher._graphs

    await launcher.cancel(
        "graph",
        CancelGraphRequest(Principal("user", "tenant"), "cancel-request"),
    )

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
