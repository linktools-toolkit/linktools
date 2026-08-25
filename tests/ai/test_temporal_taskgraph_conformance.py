#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression evidence for Temporal TaskGraph replay and cancellation."""

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import linktools.ai.temporal._activity as temporal_activity
import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    ExecutionStatus,
    Principal,
    TaskStatus,
    UsageMetrics,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    ExecutionRequest,
    ExecutionResult,
    RuntimeObjectKeyFactory,
    RuntimeState,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import (
    TaskDependencyResult,
    TaskGraph,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskLease,
    TaskNode,
    TaskNodeRunResult,
    TaskNodeView,
)
from linktools.ai.temporal._activity import TaskActivity
from linktools.ai.temporal._request import put_task_request
from linktools.ai.temporal.workflow import (
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
    TaskWorkflow,
    TaskWorkflowInput,
    TaskWorkflowNode,
)


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "model"),
        agent_digest="c" * 64,
        output_schema_id="test-output",
        output_schema_revision=1,
        output_schema_fingerprint="b" * 64,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
    )


def _request(graph_id: str = "graph") -> TaskGraphRequest:
    principal = Principal("principal", "tenant", "service")
    graph = TaskGraph(
        graph_id,
        (TaskNode("node", input={"kind": "test", "binding": _binding().to_payload()}),),
    )
    return TaskGraphRequest(graph, principal, idempotency_key=f"request-{graph_id}")


@pytest.fixture(autouse=True)
def temporal_activity_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        temporal_activity._temporal_activity,
        "info",
        lambda: SimpleNamespace(workflow_run_id="test-workflow-run"),
    )


class _ReplayRunner:
    def __init__(self, failure: AIError | None = None) -> None:
        self.failure = failure
        self.result_digest = canonical_sha256({"output": "value"})
        self.result_calls = 0
        self.terminal_status = ExecutionStatus.FAILED

    async def prepare(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> tuple[str, ExecutionRequest]:
        del node, graph_id, dependency_results
        if self.failure is not None:
            raise self.failure
        return _binding().binding_digest, ExecutionRequest(
            "prompt",
            principal,
            "key",
        )

    async def terminal_result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        del principal
        error_code = (
            ErrorCode.EXECUTION_CANCELLED
            if self.terminal_status is ExecutionStatus.CANCELLED
            else ErrorCode.EXECUTION_FAILED
        )
        return ExecutionResult(
            execution_id,
            self.terminal_status,
            None,
            None,
            None,
            None,
            UsageMetrics(),
            error_code.value,
        )

    async def result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> TaskNodeRunResult:
        del principal
        self.result_calls += 1
        return TaskNodeRunResult(self.result_digest, execution_id)


async def _runtime_activity(
    runner: _ReplayRunner,
) -> tuple[RuntimeState, TaskActivity, TaskWorkflowInput]:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="temporal-conformance", tenant_id="tenant")
    request = _request()
    await state.task.tasks.create_graph(request.graph, tenant_id="tenant")
    store = InMemoryObjectStore("requests")
    request_ref = await put_task_request(
        store,
        RuntimeObjectKeyFactory("temporal-conformance"),
        request,
    )
    workflow_request = TaskWorkflowInput.from_request(
        request,
        request_ref=request_ref,
        worker_build="worker",
    )
    activity = TaskActivity.from_runtime(
        task_state=state.task,
        runner=runner,
        request_store=store,
        namespace="temporal-conformance",
    )
    return state, activity, workflow_request


@pytest.mark.asyncio
async def test_prepare_requires_temporal_activity_run_context(monkeypatch) -> None:
    state, activity, request = await _runtime_activity(_ReplayRunner())

    def missing_context():
        raise RuntimeError("not in activity context")

    monkeypatch.setattr(temporal_activity._temporal_activity, "info", missing_context)
    try:
        with pytest.raises(AIError) as error:
            await activity.prepare(request, "node", {})
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_new_workflow_run_does_not_take_foreign_active_lease(monkeypatch) -> None:
    state, activity, request = await _runtime_activity(_ReplayRunner())
    try:
        first = await activity.prepare(request, "node", {})
        assert isinstance(first, tuple)
        lease, _ = first

        monkeypatch.setattr(
            temporal_activity._temporal_activity,
            "info",
            lambda: SimpleNamespace(workflow_run_id="new-workflow-run"),
        )
        current = await activity.prepare(request, "node", {})

        assert isinstance(current, TaskNodeView)
        assert current.status is TaskStatus.RUNNING
        assert current.owner == lease.owner
        assert current.fence == lease.fence
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_prepare_replays_failed_node_after_reconcile_failure(monkeypatch) -> None:
    state, activity, request = await _runtime_activity(
        _ReplayRunner(AIError(ErrorCode.PROMPT_TOO_LARGE))
    )
    original_reconcile = state.task.tasks.reconcile_graph
    reconcile_calls = 0

    async def fail_reconcile(graph_id: str, *, tenant_id: str):
        nonlocal reconcile_calls
        reconcile_calls += 1
        if reconcile_calls == 2:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original_reconcile(graph_id, tenant_id=tenant_id)

    monkeypatch.setattr(state.task.tasks, "reconcile_graph", fail_reconcile)
    try:
        with pytest.raises(AIError) as error:
            await activity.prepare(request, "node", {})
        assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE

        replayed = await activity.prepare(request, "node", {})
        assert isinstance(replayed, TaskNodeView)
        assert replayed.status is TaskStatus.FAILED
        nodes = await state.task.tasks.list_nodes("graph", tenant_id="tenant")
        assert nodes[0].status is TaskStatus.FAILED
        assert nodes[0].fence == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_prepare_rejects_malformed_failed_terminal(monkeypatch) -> None:
    state, activity, request = await _runtime_activity(
        _ReplayRunner(AIError(ErrorCode.PROMPT_TOO_LARGE))
    )
    original_list = state.task.tasks.list_nodes

    async def malformed_list(graph_id: str, *, tenant_id: str):
        nodes = await original_list(graph_id, tenant_id=tenant_id)
        return (replace(nodes[0], error_digest="A" * 64),)

    try:
        first = await activity.prepare(request, "node", {})
        assert isinstance(first, TaskNodeView)
        assert first.status is TaskStatus.FAILED
        monkeypatch.setattr(state.task.tasks, "list_nodes", malformed_list)
        with pytest.raises(AIError) as error:
            await activity.prepare(request, "node", {})
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_settle_replays_succeeded_node_without_second_complete(
    monkeypatch,
) -> None:
    runner = _ReplayRunner()
    state, activity, request = await _runtime_activity(runner)
    original_reconcile = state.task.tasks.reconcile_graph
    original_complete = state.task.tasks.complete
    reconcile_calls = 0
    complete_calls = 0

    async def fail_reconcile(graph_id: str, *, tenant_id: str):
        nonlocal reconcile_calls
        reconcile_calls += 1
        if reconcile_calls == 2:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original_reconcile(graph_id, tenant_id=tenant_id)

    async def count_complete(*args, **kwargs):
        nonlocal complete_calls
        complete_calls += 1
        return await original_complete(*args, **kwargs)

    monkeypatch.setattr(state.task.tasks, "reconcile_graph", fail_reconcile)
    monkeypatch.setattr(state.task.tasks, "complete", count_complete)
    try:
        prepared = await activity.prepare(request, "node", {})
        assert prepared is not None
        lease, execution = prepared
        child_result = ExecutionWorkflowResult(
            execution.execution_id,
            "SUCCEEDED",
            None,
            0,
        )

        with pytest.raises(AIError) as error:
            await activity.settle(request, lease, child_result)
        assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE

        replayed = await activity.settle(request, lease, child_result)
        assert isinstance(replayed, TaskNodeView)
        assert replayed.status is TaskStatus.SUCCEEDED
        assert replayed.result_digest == runner.result_digest
        assert replayed.execution_id == execution.execution_id
        assert complete_calls == 1
        assert runner.result_calls == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_settle_replay_rejects_mismatched_execution_result(monkeypatch) -> None:
    runner = _ReplayRunner()
    state, activity, request = await _runtime_activity(runner)
    original_reconcile = state.task.tasks.reconcile_graph
    reconcile_calls = 0

    async def fail_reconcile(graph_id: str, *, tenant_id: str):
        nonlocal reconcile_calls
        reconcile_calls += 1
        if reconcile_calls == 2:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original_reconcile(graph_id, tenant_id=tenant_id)

    monkeypatch.setattr(state.task.tasks, "reconcile_graph", fail_reconcile)
    try:
        prepared = await activity.prepare(request, "node", {})
        assert prepared is not None
        lease, execution = prepared
        child_result = ExecutionWorkflowResult(
            execution.execution_id,
            "SUCCEEDED",
            None,
            0,
        )
        with pytest.raises(AIError) as error:
            await activity.settle(request, lease, child_result)
        assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE

        runner.result_digest = canonical_sha256({"output": "changed"})
        with pytest.raises(AIError) as error:
            await activity.settle(request, lease, child_result)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_settle_replays_failed_node_without_second_fail(monkeypatch) -> None:
    state, activity, request = await _runtime_activity(_ReplayRunner())
    original_reconcile = state.task.tasks.reconcile_graph
    original_fail = state.task.tasks.fail
    reconcile_calls = 0
    fail_calls = 0

    async def fail_reconcile(graph_id: str, *, tenant_id: str):
        nonlocal reconcile_calls
        reconcile_calls += 1
        if reconcile_calls == 2:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original_reconcile(graph_id, tenant_id=tenant_id)

    async def count_fail(*args, **kwargs):
        nonlocal fail_calls
        fail_calls += 1
        return await original_fail(*args, **kwargs)

    monkeypatch.setattr(state.task.tasks, "reconcile_graph", fail_reconcile)
    monkeypatch.setattr(state.task.tasks, "fail", count_fail)
    try:
        prepared = await activity.prepare(request, "node", {})
        assert prepared is not None
        lease, execution = prepared
        child_result = ExecutionWorkflowResult(
            execution.execution_id,
            "FAILED",
            None,
            0,
        )

        with pytest.raises(AIError) as error:
            await activity.settle(request, lease, child_result)
        assert error.value.code is ErrorCode.STORAGE_UNAVAILABLE
        replayed = await activity.settle(request, lease, child_result)
        assert replayed.status is TaskStatus.FAILED
        assert fail_calls == 1
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("durable_status", "child_status"),
    [
        (TaskStatus.SUCCEEDED, "FAILED"),
        (TaskStatus.SUCCEEDED, "CANCELLED"),
        (TaskStatus.FAILED, "SUCCEEDED"),
        (TaskStatus.CANCELLED, "SUCCEEDED"),
    ],
)
async def test_terminal_settle_replay_uses_durable_status(
    monkeypatch,
    durable_status: TaskStatus,
    child_status: str,
) -> None:
    runner = _ReplayRunner()
    state, activity, request = await _runtime_activity(runner)
    try:
        prepared = await activity.prepare(request, "node", {})
        assert isinstance(prepared, tuple)
        lease, _ = prepared
        if durable_status is TaskStatus.SUCCEEDED:
            await state.task.tasks.complete(
                lease,
                tenant_id="tenant",
                execution_id="durable-execution",
                result_digest=runner.result_digest,
            )
        elif durable_status is TaskStatus.FAILED:
            await state.task.tasks.fail(
                lease,
                tenant_id="tenant",
                error_code=ErrorCode.EXECUTION_FAILED.value,
                error_digest=canonical_sha256({"type": "ExecutionResult"}),
            )
        else:
            await state.task.tasks.cancel_graph("graph", tenant_id="tenant")

        complete_calls = 0
        fail_calls = 0
        original_complete = state.task.tasks.complete
        original_fail = state.task.tasks.fail

        async def count_complete(*args, **kwargs):
            nonlocal complete_calls
            complete_calls += 1
            return await original_complete(*args, **kwargs)

        async def count_fail(*args, **kwargs):
            nonlocal fail_calls
            fail_calls += 1
            return await original_fail(*args, **kwargs)

        monkeypatch.setattr(state.task.tasks, "complete", count_complete)
        monkeypatch.setattr(state.task.tasks, "fail", count_fail)

        replayed = await activity.settle(
            request,
            lease,
            ExecutionWorkflowResult(
                "old-child-execution",
                child_status,
                None,
                0,
            ),
        )

        assert replayed.status is durable_status
        assert complete_calls == 0
        assert fail_calls == 0
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("durable_status", "child_status"),
    [
        (TaskStatus.SUCCEEDED, "FAILED"),
        (TaskStatus.SUCCEEDED, "CANCELLED"),
        (TaskStatus.FAILED, "SUCCEEDED"),
        (TaskStatus.CANCELLED, "SUCCEEDED"),
    ],
)
async def test_stale_settle_replay_uses_higher_fence_durable_status(
    monkeypatch,
    durable_status: TaskStatus,
    child_status: str,
) -> None:
    runner = _ReplayRunner()
    if child_status in {"FAILED", "CANCELLED"}:
        runner.terminal_status = ExecutionStatus(child_status)
    state, activity, request = await _runtime_activity(runner)
    original_list_nodes = state.task.tasks.list_nodes
    original_reconcile = state.task.tasks.reconcile_graph
    list_calls = 0
    mutation_calls = 0
    try:
        prepared = await activity.prepare(request, "node", {})
        assert isinstance(prepared, tuple)
        lease, _ = prepared
        durable = _durable_terminal_view(request, durable_status, runner)

        async def list_nodes(graph_id: str, *, tenant_id: str):
            nonlocal list_calls
            list_calls += 1
            if list_calls == 1:
                return await original_list_nodes(graph_id, tenant_id=tenant_id)
            return (durable,)

        async def no_op_reconcile(graph_id: str, *, tenant_id: str):
            return await original_reconcile(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(state.task.tasks, "list_nodes", list_nodes)
        monkeypatch.setattr(state.task.tasks, "reconcile_graph", no_op_reconcile)

        async def stale_complete(*args, **kwargs):
            nonlocal mutation_calls
            mutation_calls += 1
            raise AIError(ErrorCode.TASK_FENCE_STALE)

        async def stale_fail(*args, **kwargs):
            nonlocal mutation_calls
            mutation_calls += 1
            raise AIError(ErrorCode.TASK_FENCE_STALE)

        if child_status == "SUCCEEDED":
            monkeypatch.setattr(state.task.tasks, "complete", stale_complete)
        else:
            monkeypatch.setattr(state.task.tasks, "fail", stale_fail)

        replayed = await activity.settle(
            request,
            lease,
            ExecutionWorkflowResult(
                f"{request.graph_id}:{lease.node_id}",
                child_status,
                None,
                0,
            ),
        )

        assert replayed == durable
        assert mutation_calls == 1
        assert list_calls == 3
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_stale_settle_replay_keeps_running_newer_fence_stale(
    monkeypatch,
) -> None:
    runner = _ReplayRunner()
    state, activity, request = await _runtime_activity(runner)
    original_list_nodes = state.task.tasks.list_nodes
    list_calls = 0
    mutation_calls = 0
    try:
        prepared = await activity.prepare(request, "node", {})
        assert isinstance(prepared, tuple)
        lease, _ = prepared
        current = TaskNodeView(
            request.graph_id,
            lease.node_id,
            (),
            TaskStatus.RUNNING,
            "new-owner",
            lease.fence + 1,
            datetime.now(timezone.utc) + timedelta(seconds=30),
            None,
            None,
            None,
        )

        async def list_nodes(graph_id: str, *, tenant_id: str):
            nonlocal list_calls
            list_calls += 1
            if list_calls == 1:
                return await original_list_nodes(graph_id, tenant_id=tenant_id)
            return (current,)

        async def stale_complete(*args, **kwargs):
            nonlocal mutation_calls
            mutation_calls += 1
            raise AIError(ErrorCode.TASK_FENCE_STALE)

        monkeypatch.setattr(state.task.tasks, "list_nodes", list_nodes)
        monkeypatch.setattr(state.task.tasks, "complete", stale_complete)

        with pytest.raises(AIError) as error:
            await activity.settle(
                request,
                lease,
                ExecutionWorkflowResult(
                    f"{request.graph_id}:{lease.node_id}",
                    "SUCCEEDED",
                    None,
                    0,
                ),
            )

        assert error.value.code is ErrorCode.TASK_FENCE_STALE
        assert mutation_calls == 1
        assert list_calls == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_stale_settle_replay_rejects_blocked_durable_node(monkeypatch) -> None:
    runner = _ReplayRunner()
    state, activity, request = await _runtime_activity(runner)
    original_list_nodes = state.task.tasks.list_nodes
    list_calls = 0
    try:
        prepared = await activity.prepare(request, "node", {})
        assert isinstance(prepared, tuple)
        lease, _ = prepared
        current = _durable_terminal_view(request, TaskStatus.BLOCKED, runner)

        async def list_nodes(graph_id: str, *, tenant_id: str):
            nonlocal list_calls
            list_calls += 1
            if list_calls == 1:
                return await original_list_nodes(graph_id, tenant_id=tenant_id)
            return (current,)

        async def stale_complete(*args, **kwargs):
            raise AIError(ErrorCode.TASK_FENCE_STALE)

        monkeypatch.setattr(state.task.tasks, "list_nodes", list_nodes)
        monkeypatch.setattr(state.task.tasks, "complete", stale_complete)

        with pytest.raises(AIError) as error:
            await activity.settle(
                request,
                lease,
                ExecutionWorkflowResult(
                    f"{request.graph_id}:{lease.node_id}",
                    "SUCCEEDED",
                    None,
                    0,
                ),
            )

        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert list_calls == 2
    finally:
        await state.close()


def _durable_terminal_view(
    request: TaskWorkflowInput,
    status: TaskStatus,
    runner: _ReplayRunner,
) -> TaskNodeView:
    if status is TaskStatus.SUCCEEDED:
        return TaskNodeView(
            request.graph_id,
            "node",
            (),
            status,
            None,
            2,
            None,
            runner.result_digest,
            None,
            None,
            "higher-fence-execution",
        )
    if status is TaskStatus.FAILED:
        return TaskNodeView(
            request.graph_id,
            "node",
            (),
            status,
            None,
            2,
            None,
            None,
            ErrorCode.EXECUTION_FAILED.value,
            canonical_sha256({"type": "ExecutionResult"}),
            None,
        )
    if status is TaskStatus.CANCELLED:
        return TaskNodeView(
            request.graph_id,
            "node",
            (),
            status,
            None,
            2,
            None,
            None,
            None,
            None,
            None,
        )
    return TaskNodeView(
        request.graph_id,
        "node",
        (),
        status,
        None,
        2,
        None,
        None,
        ErrorCode.TASK_DEPENDENCY_FAILED.value,
        canonical_sha256({"type": "TaskDependency"}),
        None,
    )


def _workflow_request(
    nodes: tuple[TaskNode, ...],
    *,
    max_concurrency: int = 8,
) -> TaskWorkflowInput:
    request = TaskGraphRequest(
        TaskGraph("graph", nodes),
        Principal("principal", "tenant", "service"),
        idempotency_key="request-graph",
        limits=TaskGraphLimits(max_concurrency=max_concurrency),
    )
    return TaskWorkflowInput.from_request(
        request,
        request_ref="request-ref",
        worker_build="worker",
    )


class _GraphActivity:
    def __init__(self, *, failing_node: str | None = None) -> None:
        self.failing_node = failing_node
        self.settled: list[ExecutionWorkflowResult] = []
        self.renew_gate = asyncio.Event()

    async def prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> tuple[TaskLease, ExecutionWorkflowInput]:
        del dependency_results
        owner = f"owner-{node_id}"
        lease = TaskLease(
            request.graph_id,
            node_id,
            request.tenant_id,
            owner,
            1,
            datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        execution = ExecutionWorkflowInput(
            execution_id=f"{request.graph_id}:{node_id}",
            tenant_id=request.tenant_id,
            binding_digest="a" * 64,
            request_ref=request.request_ref,
            worker_build=request.worker_build,
            owner=owner,
            fence=1,
            operation_id=f"operation-{node_id}",
        )
        return lease, execution

    async def renew(self, lease: TaskLease) -> TaskLease:
        if lease.node_id == self.failing_node:
            raise RuntimeError("renew infrastructure failure")
        await self.renew_gate.wait()
        return lease

    async def settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskNodeView:
        del request, lease
        self.settled.append(result)
        if result.status == "SUCCEEDED":
            return TaskNodeView(
                "graph",
                result.execution_id.removeprefix("graph:"),
                (),
                TaskStatus.SUCCEEDED,
                None,
                1,
                None,
                canonical_sha256({"node": result.execution_id}),
                None,
                None,
                result.execution_id,
            )
        return TaskNodeView(
            "graph",
            result.execution_id.removeprefix("graph:"),
            (),
            TaskStatus.FAILED,
            None,
            1,
            None,
            None,
            ErrorCode.EXECUTION_FAILED.value,
            canonical_sha256({"type": "ExecutionResult", "code": result.status}),
            None,
        )


class _Child:
    def __init__(
        self,
        result: ExecutionWorkflowResult | None = None,
        error: BaseException | None = None,
        *,
        wait_for_cancel: bool = False,
    ) -> None:
        self.result_value = result
        self.error = error
        self.wait_for_cancel = wait_for_cancel
        self.cancel_event = asyncio.Event()
        self.cancel_calls = 0

    def cancel(self) -> bool:
        self.cancel_calls += 1
        self.cancel_event.set()
        return True

    async def result(self) -> ExecutionWorkflowResult:
        if self.wait_for_cancel:
            await self.cancel_event.wait()
            raise asyncio.CancelledError
        if self.error is not None:
            raise self.error
        assert self.result_value is not None
        return self.result_value


class _TestWorkflow(TaskWorkflow):
    def __init__(
        self,
        activity: _GraphActivity,
        children: Mapping[str, _Child],
    ) -> None:
        super().__init__(activity)
        self.children = children

    def _start_child(
        self,
        execution_input: ExecutionWorkflowInput,
        request: TaskWorkflowInput,
        node: TaskWorkflowNode,
    ) -> _Child:
        del execution_input, request
        return self.children[node.node_id]


@pytest.mark.asyncio
async def test_child_failure_is_settled_as_failed() -> None:
    activity = _GraphActivity()
    request = _workflow_request((TaskNode("node", input={"kind": "test"}),))
    child = _Child(error=RuntimeError("child failed"))
    result = await _TestWorkflow(activity, {"node": child}).run(request)

    assert result.status == "FAILED"
    assert [item.status for item in activity.settled] == ["FAILED"]
    assert child.cancel_calls == 0


@pytest.mark.asyncio
async def test_child_cancellation_propagates_without_remote_recancel() -> None:
    activity = _GraphActivity()
    request = _workflow_request((TaskNode("node", input={"kind": "test"}),))
    prepared = await activity.prepare(request, "node", {})
    lease, execution = prepared
    workflow = TaskWorkflow(activity)
    child = _Child(error=asyncio.CancelledError())

    _, result = await workflow._wait_for_child(execution, lease, child)

    assert result.status == "CANCELLED"
    assert child.cancel_calls == 0


@pytest.mark.asyncio
async def test_workflow_cancellation_cleans_local_waiters() -> None:
    activity = _GraphActivity()
    request = _workflow_request((TaskNode("node", input={"kind": "test"}),))
    prepared = await activity.prepare(request, "node", {})
    lease, execution = prepared
    workflow = TaskWorkflow(activity)
    child = _Child(wait_for_cancel=True)
    wait_task = asyncio.create_task(workflow._wait_for_child(execution, lease, child))

    await asyncio.sleep(0)
    wait_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait_task

    assert child.cancel_calls == 1


@pytest.mark.asyncio
async def test_renew_failure_race_with_cancel_returns_cancelled() -> None:
    activity = _GraphActivity(failing_node="node")
    request = _workflow_request((TaskNode("node", input={"kind": "test"}),))
    prepared = await activity.prepare(request, "node", {})
    lease, execution = prepared
    workflow = TaskWorkflow(activity)
    workflow._cancelled = True
    child = _Child(wait_for_cancel=True)

    with pytest.raises(asyncio.CancelledError):
        await workflow._wait_for_child(execution, lease, child)

    assert child.cancel_calls == 1


@pytest.mark.asyncio
async def test_renew_failure_cancels_child_and_skips_settle() -> None:
    activity = _GraphActivity(failing_node="node")
    request = _workflow_request((TaskNode("node", input={"kind": "test"}),))
    child = _Child(wait_for_cancel=True)

    with pytest.raises(RuntimeError, match="renew infrastructure failure"):
        await _TestWorkflow(activity, {"node": child}).run(request)

    assert child.cancel_calls >= 1
    assert activity.settled == []


@pytest.mark.asyncio
async def test_graph_cancel_during_child_wait_returns_cancelled() -> None:
    activity = _GraphActivity()
    request = _workflow_request((TaskNode("node", input={"kind": "test"}),))
    child = _Child(wait_for_cancel=True)
    workflow = _TestWorkflow(activity, {"node": child})
    run_task = asyncio.create_task(workflow.run(request))

    for _ in range(100):
        if workflow._active_children:
            break
        await asyncio.sleep(0)
    assert workflow._active_children
    workflow.cancel("cancel-request")
    result = await run_task

    assert result.status == "CANCELLED"
    assert child.cancel_calls >= 1


@pytest.mark.asyncio
async def test_batch_failure_cleans_sibling_child_workflow() -> None:
    activity = _GraphActivity(failing_node="bad")
    request = _workflow_request(
        (
            TaskNode("bad", input={"kind": "test"}),
            TaskNode("sibling", input={"kind": "test"}),
        ),
        max_concurrency=2,
    )
    bad = _Child(wait_for_cancel=True)
    sibling = _Child(wait_for_cancel=True)

    with pytest.raises(RuntimeError, match="renew infrastructure failure"):
        await _TestWorkflow(activity, {"bad": bad, "sibling": sibling}).run(request)

    assert bad.cancel_calls >= 1
    assert sibling.cancel_calls >= 1
    assert activity.settled == []


@pytest.mark.asyncio
async def test_multi_dependency_graph_keeps_dependency_results() -> None:
    activity = _GraphActivity()
    request = _workflow_request(
        (
            TaskNode("left", input={"kind": "test"}),
            TaskNode("right", input={"kind": "test"}),
            TaskNode("join", ("left", "right"), input={"kind": "test"}),
        ),
        max_concurrency=2,
    )
    children = {
        node.node_id: _Child(
            ExecutionWorkflowResult(f"graph:{node.node_id}", "SUCCEEDED", None, 0)
        )
        for node in (request.nodes[0], request.nodes[1], request.nodes[2])
    }

    result = await _TestWorkflow(activity, children).run(request)

    assert result.status == "SUCCEEDED"
    assert result.completed_node_ids == ("join", "left", "right")
