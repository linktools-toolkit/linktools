#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fault coverage for Runtime tool terminal ownership and local task waiters."""

import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import pytest
from linktools.ai.core import Principal, TaskStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._capabilities import (
    ToolOperationDecision,
    _RuntimeStepPersistence,
    _tool_effect_policy,
)
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.runtime.state import ToolOperationAdmission
from linktools.ai.storage import PayloadPolicy, StoredPayload
from linktools.ai.task._graph import CancelGraphRequest, TaskGraph, TaskGraphRequest, TaskGraphView, TaskNode
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
    idempotency_key: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class _StepStore:
    def __init__(self) -> None:
        self.effects: list[_Effect] = []

    async def record_tool_effect(self, effect: Any) -> None:
        self.effects.append(
            _Effect(
                effect.run_id,
                effect.tool_call_id,
                effect.status,
                effect.effect_summary,
                effect.idempotency_key,
            )
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
        self.calls: list[str] = []

    async def begin(self, ctx, call, tool_def, args, replay_safe):
        del ctx, call, tool_def, args
        self.calls.append("begin")
        return replace(self.decision, replay_safe=replay_safe)

    async def renew(self, decision):
        self.calls.append("renew")
        return decision

    async def complete(self, decision, result):
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision, error):
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(self, decision, error):
        del decision, error
        self.calls.append("unknown")


class _OperationRepository:
    def __init__(self) -> None:
        self.request: ToolOperationAdmission | None = None

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord:
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
            binding_digest=request.binding_digest,
            replay_safe=request.replay_safe,
            status=ToolOperationStatus.CLAIMED,
            owner=request.owner,
            fence=1,
            lease_expires_at=now,
            error_code=None,
            created_at=now,
            updated_at=now,
        )


def _context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id="run")


def _definition(replay_safe: bool) -> ToolDefinition:
    return ToolDefinition(name="tool", metadata={"linktools.ai.replay_safe": replay_safe})


async def test_tool_operation_admission_uses_runtime_step_and_binding_digest() -> None:
    repository = _OperationRepository()
    bridge = RuntimeToolOperationBridge(
        repository,
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="runtime-step",
        binding_digest="binding",
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )
    await bridge.begin(
        _context(),
        ToolCallPart("tool", {}, tool_call_id="call"),
        _definition(True),
        {},
        True,
    )
    assert repository.request is not None
    assert repository.request.step_run_id == "runtime-step"
    assert repository.request.binding_digest == "binding"
    assert not hasattr(repository.request, "binding_fingerprint")


@pytest.mark.parametrize(
    ("name", "capability_id", "tool_class", "replay_safe", "effect_free"),
    [
        ("read_file", "workspace-filesystem", "filesystem.read", True, True),
        ("write_file", "workspace-filesystem", "filesystem.write", False, False),
        ("check_command", "workspace-shell", "shell", True, True),
        ("run_command", "workspace-shell", "shell", False, False),
        ("read_memory", "linktools-memory", "memory.read", True, True),
        ("write_memory", "linktools-memory", "memory.write", True, False),
        ("list_skills", "linktools-skill", "control", True, True),
        ("write_plan", "linktools-planning", "control", True, False),
        ("delegate_task", "linktools-subagent", "control", True, False),
    ],
)
async def test_trusted_tool_effect_policy_matrix(
    name: str,
    capability_id: str,
    tool_class: str,
    replay_safe: bool,
    effect_free: bool,
) -> None:
    policy = _tool_effect_policy(
        ToolDefinition(name=name, capability_id=capability_id),
        trusted_tool_classes=((name, tool_class),),
    )
    assert policy.replay_safe is replay_safe
    assert policy.effect_free is effect_free


async def test_trusted_tool_effect_policy_rejects_spoofed_capability() -> None:
    with pytest.raises(AIError) as raised:
        _tool_effect_policy(
            ToolDefinition(name="read_file", capability_id="custom"),
            trusted_tool_classes=(("read_file", "filesystem.read"),),
        )
    assert raised.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT


async def test_custom_tool_replay_metadata_remains_explicit_opt_in() -> None:
    safe = _tool_effect_policy(
        ToolDefinition(name="custom", metadata={"linktools.ai.replay_safe": True}),
        trusted_tool_classes=(),
    )
    unsafe = _tool_effect_policy(ToolDefinition(name="custom"), trusted_tool_classes=())
    assert (safe.replay_safe, safe.effect_free) == (True, False)
    assert (unsafe.replay_safe, unsafe.effect_free) == (False, False)


async def _capability(replay_safe: bool):
    bridge = _ToolBridge(ToolOperationDecision("operation", "owner", 1, replay_safe))
    store = _StepStore()
    capability = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
    )
    context = _context()
    call = ToolCallPart("tool", {}, tool_call_id="call")
    definition = _definition(replay_safe)
    await capability.before_tool_execute(context, call=call, tool_def=definition, args={})
    return capability, bridge, store, context, call, definition


@pytest.mark.parametrize("replay_safe", [True, False])
async def test_model_retry_respects_effect_replay_safety(replay_safe: bool) -> None:
    capability, bridge, store, context, call, definition = await _capability(replay_safe)

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
        assert bridge.calls == ["begin", "fail"]
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
        assert bridge.calls == ["begin", "unknown"]
        assert [effect.status for effect in store.effects] == ["started"]


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
    waiter = asyncio.create_task(launcher.wait_graph_activity("graph", tenant_id="tenant"))
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
    with pytest.raises(AIError) as raised:
        await asyncio.wait_for(
            launcher.wait_graph_activity("graph", tenant_id="tenant"),
            timeout=1,
        )
    assert raised.value.code is ErrorCode.INTERNAL_ERROR
    await launcher.shutdown()


async def test_launcher_cancel_clears_retained_failure() -> None:
    repository = _TaskRepository()
    repository.failure = RuntimeError("scheduler failure")
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    await launcher.start(_task_request())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await launcher.cancel(
        "graph",
        CancelGraphRequest(Principal("user", "tenant"), "cancel-request"),
    )
    assert not launcher._graphs
    await launcher.shutdown()
