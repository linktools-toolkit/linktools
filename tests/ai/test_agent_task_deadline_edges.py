#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deadline outcome races and bounded cancellation preserve authoritative results."""
from linktools.ai.model import create_model

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, RunBudget, TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.execution import ExecutionView
from linktools.ai.task_engine import _agent as _agent_task
from linktools.ai.task_engine._agent import AgentTaskRunner
from linktools.ai.task import TaskNode, TaskNodeInvocation, TaskNodeRunError


_PRINCIPAL = Principal("deadline", "tenant", "service")


class _Execution:
    def __init__(self, view: ExecutionView) -> None:
        self.view = view
        self.waits = 0
        self.cancels = 0
        self.cancel_started = asyncio.Event()
        self.cancel_finished = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def wait(self, *, timeout_seconds: float | None = None) -> object:
        del timeout_seconds
        self.waits += 1
        if self.view.terminal_at is None:
            self.view = replace(
                self.view, status=ExecutionStatus.SUCCEEDED,
                terminal_at=datetime.now(timezone.utc),
            )
        return SimpleNamespace(result=SimpleNamespace(
            status=self.view.status, output="done", error_code=None,
        ))

    async def cancel(self) -> None:
        self.cancels += 1
        self.cancel_started.set()
        try:
            await self.release_cancel.wait()
        finally:
            self.cancel_finished.set()


def _view(started_at: datetime, terminal_at: datetime | None = None) -> ExecutionView:
    return ExecutionView(
        "execution", "agent",
        ExecutionStatus.STARTED if terminal_at is None else ExecutionStatus.SUCCEEDED,
        ExecutionLineageKind.RUN, None, "execution", None,
        started_at=started_at, terminal_at=terminal_at,
    )


def _runner(execution: _Execution) -> AgentTaskRunner:
    # A controlled execution boundary makes result/readback races deterministic;
    # full Runtime construction and real cancellation are covered by integration tests.
    runner = AgentTaskRunner.__new__(AgentTaskRunner)

    async def inspect(execution_id: str, principal: Principal) -> ExecutionView:
        assert execution_id == "execution" and principal == _PRINCIPAL
        return execution.view

    async def get(execution_id: str, principal: Principal) -> _Execution:
        assert execution_id == "execution" and principal == _PRINCIPAL
        return execution

    runner._inspect_execution = inspect
    runner._get_execution = get
    runner._record_invocation = None
    return runner


def _invocation(timeout: float = 1.0) -> TaskNodeInvocation:
    return TaskNodeInvocation(
        TaskNode("agent", timeout_seconds=timeout), "graph", _PRINCIPAL, {}, {},
    )


@pytest.mark.asyncio
async def test_timed_success_does_not_repeat_result_observation() -> None:
    execution = _Execution(_view(datetime.now(timezone.utc)))
    result = await _runner(execution).wait_bound(_invocation(), "execution")
    assert result.execution_id == "execution"
    # Each SDK wait captures/replays the execution tree: duplicate consumption
    # would regress the accepted single-observation IO budget for normal completion.
    assert execution.waits == 1
    assert execution.cancels == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("on_time", (True, False))
async def test_late_read_uses_durable_completion_time(on_time: bool) -> None:
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    execution = _Execution(_view(
        started, started + timedelta(seconds=0.5 if on_time else 2),
    ))
    if on_time:
        result = await _runner(execution).wait_bound(_invocation(), "execution")
        assert result.execution_id == "execution"
    else:
        with pytest.raises(TaskNodeRunError) as caught:
            await _runner(execution).wait_bound(_invocation(), "execution")
        assert caught.value.code is ErrorCode.EXECUTION_WAIT_TIMEOUT
        assert caught.value.error_origin == "node"
    assert execution.cancels == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", (False, True))
async def test_unsettled_cancel_has_fixed_bound(
    monkeypatch: pytest.MonkeyPatch, interrupt: bool,
) -> None:
    monkeypatch.setattr(_agent_task, "_DEADLINE_CANCEL_TIMEOUT_SECONDS", 0.04)
    execution = _Execution(_view(datetime.now(timezone.utc) - timedelta(seconds=10)))
    waiting = asyncio.create_task(_runner(execution).wait_bound(_invocation(), "execution"))
    await asyncio.wait_for(execution.cancel_started.wait(), 1)
    if interrupt:
        for _ in range(3):
            waiting.cancel()
            await asyncio.sleep(0.005)
    try:
        done, _ = await asyncio.wait({waiting}, timeout=0.5)
        assert waiting in done, "caller cancellation must not renew the settlement deadline"
        with pytest.raises(TaskNodeRunError) as caught:
            waiting.result()
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert caught.value.safe_details["reason"] == "settlement_timeout"
        await asyncio.wait_for(execution.cancel_finished.wait(), 1)
    finally:
        execution.release_cancel.set()
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


def test_recovery_deadline_uses_original_start_and_tighter_parent() -> None:
    started = datetime(2020, 1, 1, tzinfo=timezone.utc)
    node = TaskNode("agent", timeout_seconds=3)
    policy = AgentTaskRunner.deadline_policy(node, started, None)
    assert policy is not None
    assert policy[0] == started + timedelta(seconds=3)
    assert policy[1] is ErrorCode.EXECUTION_WAIT_TIMEOUT
    budget = RunBudget(deadline_at=started + timedelta(seconds=2))
    parent_policy = AgentTaskRunner.deadline_policy(node, started, ("graph:graph", budget))
    assert parent_policy is not None
    assert parent_policy[0] == budget.deadline_at
    assert parent_policy[1] is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    assert parent_policy[2]["scope_id"] == "graph:graph"


@pytest.mark.asyncio
async def test_recovery_deadline_guard_does_not_block_pending_cancellation() -> None:
    from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
    from linktools.ai.persistence.contracts import RecoveryCheckpointState, RecoveryHandoffPhase
    from .test_local_execution_recovery import _record

    execution = replace(_record(), status=ExecutionStatus.CANCELLING)
    port = SimpleNamespace(
        tenant_id="tenant",
        load_execution=AsyncMock(return_value=execution),
        _reconcile_tool_effects=AsyncMock(return_value=((), 0)),
        _pending_cancel_operations=AsyncMock(return_value=()),
        validate_binding=Mock(),
        _reconcile_session_recovery=AsyncMock(return_value=True),
        _recovery_idempotency=AsyncMock(return_value=None),
        _commit_terminal=AsyncMock(),
    )
    coordinator = _RecoveryCoordinator(port, None)
    guard = AsyncMock(return_value=False)
    coordinator.bind_relaunch_guard(guard)
    launched = await coordinator.reconcile_checkpoint(SimpleNamespace(
        execution_id=execution.execution_id, state=RecoveryCheckpointState.ACTIVE,
        handoff_phase=RecoveryHandoffPhase.NONE,
    ))
    assert not launched
    guard.assert_not_awaited()
    assert port._commit_terminal.await_args.args[1] is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_standalone_recovery_does_not_read_graph_deadline_state() -> None:
    from linktools.ai.inputs.recovery import TaskRecoveryGuard
    from .test_local_execution_recovery import _record

    # No graph repositories exist on this boundary double: standalone recovery
    # must not reach graph ownership or input-capture reads.
    guard = TaskRecoveryGuard(SimpleNamespace(), SimpleNamespace())
    assert await guard(_record())


@pytest.mark.asyncio
async def test_task_recovery_without_retained_invocation_does_not_relaunch() -> None:
    from linktools.ai.inputs.recovery import TaskRecoveryGuard
    from .test_local_execution_recovery import _record

    guard = TaskRecoveryGuard(
        SimpleNamespace(tenant_id="tenant"),
        SimpleNamespace(read_invocation_node=AsyncMock(return_value=None)),
    )
    assert not await guard(replace(_record(), requires_task_invocation_capture=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", (False, True))
async def test_rebound_deadline_preserves_invocation_before_failure(
    monkeypatch: pytest.MonkeyPatch, unknown: bool,
) -> None:
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    execution = _Execution(_view(started, None if unknown else started + timedelta(seconds=2)))
    runner = _runner(execution)
    captured = asyncio.Event()

    async def capture(execution_id: str, invocation: TaskNodeInvocation) -> None:
        assert execution_id == "execution" and invocation.node.node_id == "agent"
        captured.set()

    runner._record_invocation = capture
    monkeypatch.setattr(_agent_task, "_DEADLINE_CANCEL_TIMEOUT_SECONDS", 0.02)
    try:
        with pytest.raises(TaskNodeRunError) as caught:
            await runner.wait_bound(_invocation(), "execution")
        assert captured.is_set()
        assert caught.value.code is (ErrorCode.STORAGE_RECOVERY_REQUIRED if unknown else ErrorCode.EXECUTION_WAIT_TIMEOUT)
    finally:
        execution.release_cancel.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_subagent_recovery_inherits_task_ancestor_deadline() -> None:
    from linktools.ai.inputs.recovery import TaskRecoveryGuard
    from .test_local_execution_recovery import _record

    parent = replace(_record(), execution_id="parent", root_execution_id="parent",
                     requires_task_invocation_capture=True,
                     started_at=datetime.now(timezone.utc) - timedelta(seconds=10))
    child = replace(_record(), parent_execution_id="parent", root_execution_id="parent",
                    parent_invocation_id="delegate", lineage_kind=ExecutionLineageKind.SUBAGENT,
                    started_at=datetime.now(timezone.utc))
    principal = Principal(parent.principal_id, "tenant", parent.principal_kind)
    node = TaskNode("node", timeout_seconds=1)
    persistence = SimpleNamespace(
        tenant_id="tenant",
        execution=SimpleNamespace(executions=SimpleNamespace(get=AsyncMock(return_value=parent))),
        task=SimpleNamespace(
            operations=SimpleNamespace(list_pending=AsyncMock(return_value=())),
            tasks=SimpleNamespace(graph_state=AsyncMock(return_value=SimpleNamespace(
                status=TaskStatus.RUNNING, nodes=(node,),
                node_states=(SimpleNamespace(node_id="node", execution_id="parent", status=TaskStatus.RUNNING),),
            ))),
            admissions=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
                principal=principal, budget_scope_id=None, budget=None,
            ))),
        ),
    )
    captures = SimpleNamespace(read_invocation_node=AsyncMock(return_value=("graph", node)))
    assert not await TaskRecoveryGuard(persistence, captures)(child)



@pytest.mark.asyncio
async def test_recovery_binding_candidate_ignores_model_connection_changes() -> None:
    from linktools.ai.agent import AgentCatalog, AgentCompiler

    from linktools.ai.runtime._agent_binding_resolver import _AgentBindingResolver
    from linktools.ai.spec import AgentSpec

    specs = {
        "parent": AgentSpec("parent", allow_tools=(), allow_skills=(), allow_subagents=("child",)),
        "child": AgentSpec("child", allow_tools=(), allow_skills=(), allow_subagents=()),
    }
    contracts = []
    for endpoint in ("https://first.example/v1", "https://second.example/v1"):
        compiler = AgentCompiler(
            models={"default": create_model(protocol='openai',
                model="gpt-test", base_url=endpoint, api_key="test-placeholder",
            )},
            candidates=(), agents=specs,
        )
        catalog = AgentCatalog({key: compiler.compile(spec) for key, spec in specs.items()})
        resolver = _AgentBindingResolver(catalog, compiler)
        contracts.append(await resolver.resolve_root("parent"))
    assert contracts[0].binding_digest == contracts[1].binding_digest
    assert contracts[0] == contracts[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_status,node_status", [
    (TaskStatus.CANCELLED, TaskStatus.RUNNING),
    (TaskStatus.FAILED, TaskStatus.RUNNING),
    (TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED),
    (TaskStatus.RUNNING, TaskStatus.CANCELLED),
    (TaskStatus.RUNNING, TaskStatus.FAILED),
    (TaskStatus.RUNNING, TaskStatus.RUNNING),
])
async def test_recovery_guard_respects_durable_graph_and_node_outcome(
    graph_status: TaskStatus, node_status: TaskStatus,
) -> None:
    from linktools.ai.inputs.recovery import TaskRecoveryGuard
    from .test_local_execution_recovery import _record

    execution = replace(_record(), requires_task_invocation_capture=True)
    node = TaskNode("node")
    persistence = SimpleNamespace(
        tenant_id="tenant",
        task=SimpleNamespace(
            operations=SimpleNamespace(list_pending=AsyncMock(return_value=())),
            tasks=SimpleNamespace(graph_state=AsyncMock(return_value=SimpleNamespace(
                status=graph_status, nodes=(node,),
                node_states=(SimpleNamespace(node_id="node", execution_id=execution.execution_id,
                                             status=node_status),),
            ))),
            admissions=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
                principal=Principal(execution.principal_id, "tenant", execution.principal_kind),
                budget_scope_id=None, budget=None,
            ))),
        ),
    )
    captures = SimpleNamespace(read_invocation_node=AsyncMock(return_value=("graph", node)))
    assert await TaskRecoveryGuard(persistence, captures)(execution) is (
        graph_status is TaskStatus.RUNNING and node_status is TaskStatus.RUNNING
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_target,expected", [(None, False), ("execution", False), ("other", True)])
async def test_recovery_guard_honors_pending_cancel_before_graph_state_changes(
    cancel_target: str | None, expected: bool,
) -> None:
    from linktools.ai.core import OperationKind
    from linktools.ai.inputs.recovery import TaskRecoveryGuard
    from .test_local_execution_recovery import _record

    execution = replace(_record(), execution_id="execution", root_execution_id="execution",
                        requires_task_invocation_capture=True)
    node = TaskNode("node")
    operations = SimpleNamespace(list_pending=AsyncMock(return_value=(SimpleNamespace(
        operation_kind=OperationKind.TASK_CANCEL, execution_id=cancel_target,
    ),)))
    persistence = SimpleNamespace(
        tenant_id="tenant",
        task=SimpleNamespace(
            operations=operations,
            tasks=SimpleNamespace(graph_state=AsyncMock(return_value=SimpleNamespace(
                status=TaskStatus.RUNNING, nodes=(node,),
                node_states=(SimpleNamespace(node_id="node", execution_id=execution.execution_id,
                                             status=TaskStatus.RUNNING),),
            ))),
            admissions=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
                principal=Principal(execution.principal_id, "tenant", execution.principal_kind),
                budget_scope_id=None, budget=None,
            ))),
        ),
    )
    captures = SimpleNamespace(read_invocation_node=AsyncMock(return_value=("graph", node)))
    assert await TaskRecoveryGuard(persistence, captures)(execution) is expected


@pytest.mark.asyncio
async def test_recovery_guard_checks_cancellation_after_first_ledger_page() -> None:
    from linktools.ai.core import OperationKind
    from linktools.ai.inputs.recovery import TaskRecoveryGuard

    pending = AsyncMock(side_effect=[
        tuple(SimpleNamespace(operation_kind=OperationKind.TASK_CANCEL,
                              execution_id="other", sequence=index) for index in range(1, 129)),
        (SimpleNamespace(operation_kind=OperationKind.TASK_CANCEL, execution_id=None, sequence=129),),
    ])
    guard = TaskRecoveryGuard(SimpleNamespace(
        tenant_id="tenant", task=SimpleNamespace(operations=SimpleNamespace(list_pending=pending)),
    ), SimpleNamespace())
    assert await guard._has_cancel_intent("graph", "execution")
    assert [call.kwargs["after_sequence"] for call in pending.await_args_list] == [None, 128]
