#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fault coverage for Runtime tool terminal ownership and local task waiters."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from linktools.ai.core import Principal, TaskStatus, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.runtime.state._contracts import (
    ToolOperationAdmission,
    ToolOperationRecord,
)
from linktools.ai.storage import PayloadPolicy
from linktools.ai.task._graph import (
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphRequest,
    TaskGraphView,
    TaskLease,
    TaskNode,
)
from linktools.ai.task._local import (
    LocalTaskGraphLauncher,
    TaskNodeRunResult,
    _LeaseState,
)
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage


class _StepStore:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def append_event(self, event: Any) -> None:
        self.events.append(event)


class _ToolBridge:
    def __init__(self, decision: ToolOperationDecision) -> None:
        self.decision = decision
        self.calls: list[str] = []

    async def effective_args(self, ctx, call, tool_def, args):
        del ctx, call, tool_def
        return args

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
            execution_id=request.execution_id,
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
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


def _definition(replay_safe: bool) -> ToolDefinition:
    return ToolDefinition(
        name="tool", metadata={"linktools.ai.replay_safe": replay_safe}
    )


@pytest.mark.asyncio
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
    assert repository.request.execution_id == "execution"
    assert repository.request.step_run_id == "runtime-step"
    assert repository.request.binding_digest == "binding"
    assert not hasattr(repository.request, "binding_fingerprint")


@pytest.mark.asyncio
async def test_tool_operation_identity_is_scoped_to_step_run() -> None:
    call = ToolCallPart("tool", {}, tool_call_id="call")
    first_repository = _OperationRepository()
    first_bridge = RuntimeToolOperationBridge(
        first_repository,
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="first-step",
        binding_digest="binding",
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )
    second_repository = _OperationRepository()
    second_bridge = RuntimeToolOperationBridge(
        second_repository,
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="second-step",
        binding_digest="binding",
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )

    await first_bridge.begin(_context(), call, _definition(True), {}, True)
    await second_bridge.begin(_context(), call, _definition(True), {}, True)

    assert first_repository.request is not None
    assert second_repository.request is not None
    assert (
        first_repository.request.tool_operation_id
        != second_repository.request.tool_operation_id
    )


@pytest.mark.asyncio
async def test_tool_operation_cache_rejects_changed_call_fingerprint() -> None:
    repository = _OperationRepository()
    bridge = RuntimeToolOperationBridge(
        repository,
        object(),
        namespace="namespace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="step",
        binding_digest="binding",
        owner="owner",
        background_tasks=set(),
        payload_policy=PayloadPolicy(),
    )
    context = _context()
    definition = _definition(True)
    call = ToolCallPart("tool", {}, tool_call_id="call")

    await bridge.begin(context, call, definition, {}, True)

    with pytest.raises(AIError) as raised:
        await bridge.begin(context, call, definition, {"changed": True}, True)

    assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


def test_managed_tool_descriptor_keeps_effect_ownership_explicit() -> None:
    descriptor = ManagedToolDescriptor(
        effect_owner="tool_operation",
        effect="non_replay_safe",
        tool_class="filesystem.write",
    )
    assert descriptor.effect_owner == "tool_operation"
    assert descriptor.effect == "non_replay_safe"


def test_managed_tool_descriptor_rejects_effect_free_mismatch() -> None:
    with pytest.raises(ValueError):
        ManagedToolDescriptor(
            effect_owner="none",
            effect="replay_safe",
            tool_class="business",
        )


@pytest.mark.asyncio
async def test_replay_safe_model_retry_is_a_known_failure() -> None:
    bridge = _ToolBridge(ToolOperationDecision("operation", "owner", 1, True))

    async def retry_tool() -> None:
        raise ModelRetry("retry")

    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([retry_tool]),),
        {
            "retry_tool": ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="replay_safe",
                tool_class="business",
            )
        },
        id="business",
        tool_operations=bridge,  # type: ignore[arg-type]
    )
    context = _context()
    tools = await boundary.get_tools(context)
    with pytest.raises(ModelRetry):
        await boundary.call_tool(
            "retry_tool",
            {},
            context,
            tools["retry_tool"],
        )
    assert bridge.calls == ["begin", "fail"]


@pytest.mark.asyncio
async def test_non_replay_safe_model_retry_requires_effect_verification() -> None:
    bridge = _ToolBridge(ToolOperationDecision("operation", "owner", 1, False))

    async def retry_tool() -> None:
        raise ModelRetry("retry")

    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([retry_tool]),),
        {
            "retry_tool": ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="business",
            )
        },
        id="business",
        tool_operations=bridge,  # type: ignore[arg-type]
    )
    context = _context()
    tools = await boundary.get_tools(context)
    with pytest.raises(ToolFailed, match="TOOL_EFFECT_UNKNOWN"):
        await boundary.call_tool(
            "retry_tool",
            {},
            context,
            tools["retry_tool"],
        )
    assert bridge.calls == ["begin", "unknown"]


class _TaskRepository:
    def __init__(self, status: TaskStatus = TaskStatus.RUNNING) -> None:
        self.status = status
        self.failure: BaseException | None = None
        self.recovery: dict[str, object] | None = None

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        del graph_id, tenant_id
        if self.failure is not None:
            raise self.failure
        return TaskGraphView("graph", self.status, ())

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        del graph_id, tenant_id
        return TaskGraphView("graph", self.status, (TaskNode("node"),))

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[object, ...]:
        del graph_id, tenant_id
        return ()

    async def claim(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("terminal graph must not claim a node")

    async def mark_recovery_required(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> object:
        self.recovery = {
            "lease": lease,
            "tenant_id": tenant_id,
            "error_code": error_code,
            "error_digest": error_digest,
            "execution_id": execution_id,
        }
        return object()


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


@pytest.mark.parametrize(
    "code",
    (
        ErrorCode.STORAGE_COMMIT_UNKNOWN,
        ErrorCode.EXECUTION_START_UNKNOWN,
        ErrorCode.TOOL_EFFECT_UNKNOWN,
    ),
)
@pytest.mark.asyncio
async def test_task_recovery_persists_bounded_original_error_code(code: ErrorCode) -> None:
    repository = _TaskRepository()
    launcher = object.__new__(LocalTaskGraphLauncher)
    launcher._lock = asyncio.Lock()
    launcher._repository = repository
    request = _task_request()
    run = SimpleNamespace(
        request=request,
        condition=asyncio.Condition(),
        generation=0,
        failure=None,
        closed=False,
    )
    launcher._graphs = {(request.principal.tenant_id, request.graph.graph_id): run}
    cause = AIError(code, safe_details={"source": "must-not-be-copied"})
    lease = TaskLease(
        "graph",
        "node",
        "tenant",
        "owner",
        1,
        datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    await launcher._defer_recovery(
        run,
        request.graph.nodes[0],
        _LeaseState(lease),
        cause=cause,
    )

    assert run.failure is None
    assert run.closed is True
    assert repository.recovery == {
        "lease": lease,
        "tenant_id": "tenant",
        "error_code": code.value,
        "error_digest": canonical_sha256(
            {"graph_id": "graph", "node_id": "node", "code": code.value}
        ),
        "execution_id": None,
    }


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_launcher_cancel_clears_retained_failure() -> None:
    repository = _TaskRepository()
    repository.failure = RuntimeError("scheduler failure")
    launcher = LocalTaskGraphLauncher(repository, _TaskRunner(), owner="launcher")
    request = _task_request()
    launch = TaskGraphLaunch(
        request.graph, request.principal, request.limits, request.correlation
    )
    await launcher.start(launch)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await launcher.cancel(launch)
    assert not launcher._graphs
    await launcher.shutdown()
