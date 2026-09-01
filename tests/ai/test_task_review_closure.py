#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph review closure invariants."""

import asyncio
from types import SimpleNamespace

import pytest

from linktools.ai.core import OperationStatus, Principal, TaskStatus, idempotency_key_digest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._planner import _AgentTaskNodeHandler
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskService,
    LocalTaskGraphLauncher,
    TaskFunction,
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskNode,
    TaskNodeContext,
    TaskNodeRunResult,
)
from linktools.ai.workspace import trusted_workspace_principal


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _NoopRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, *args: object, **kwargs: object) -> TaskNodeRunResult:
        del args, kwargs
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _FailingCancelLauncher:
    async def start(self, launch: object) -> object:
        del launch
        return SimpleNamespace()

    async def cancel(self, graph_id: str, request: object) -> object:
        del graph_id, request
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)


@pytest.mark.asyncio
async def test_natural_failure_is_visible_before_explicit_cancel_override() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-natural-aggregate", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("natural-aggregate", (TaskNode("failed"), TaskNode("active")))
        await repository.create_graph(graph, tenant_id="tenant")
        lease = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="worker",
            lease_seconds=30,
        )
        await repository.fail(
            lease,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="a" * 64,
        )

        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert before is not None
        assert before.status is TaskStatus.FAILED

        view = await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        after = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert after is not None
        states = {item.node_id: item for item in after.node_states}
        assert view.status is TaskStatus.CANCELLED
        assert after.status is TaskStatus.CANCELLED
        assert states["failed"].status is TaskStatus.FAILED
        assert states["active"].status is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_explicit_cancel_preserves_blocked_terminal_node() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cancel-blocked", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "cancel-blocked",
            (
                TaskNode("failed"),
                TaskNode("blocked", ("failed",)),
                TaskNode("active"),
            ),
        )
        await repository.create_graph(graph, tenant_id="tenant")
        await repository.claim(
            graph.graph_id,
            "active",
            tenant_id="tenant",
            owner="active-worker",
            lease_seconds=30,
        )
        failed = await repository.claim(
            graph.graph_id,
            "failed",
            tenant_id="tenant",
            owner="failed-worker",
            lease_seconds=30,
        )
        await repository.fail(
            failed,
            tenant_id="tenant",
            error_code=ErrorCode.TASK_NODE_FAILED.value,
            error_digest="b" * 64,
        )
        await repository.reconcile_graph(graph.graph_id, tenant_id="tenant")

        before = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert before is not None
        before_states = {item.node_id: item for item in before.node_states}
        assert before_states["blocked"].status is TaskStatus.BLOCKED
        assert before_states["active"].status is TaskStatus.RUNNING

        await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        after = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert after is not None
        states = {item.node_id: item for item in after.node_states}
        assert after.status is TaskStatus.CANCELLED
        assert states["failed"].status is TaskStatus.FAILED
        assert states["blocked"].status is TaskStatus.BLOCKED
        assert states["active"].status is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_launcher_counts_remote_live_lease_in_max_concurrency() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-global-capacity", tenant_id="tenant")
    launcher = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("global-capacity", (TaskNode("remote"), TaskNode("local")))
        await repository.create_graph(graph, tenant_id="tenant")
        await repository.claim(
            graph.graph_id,
            "remote",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        runner = _NoopRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                trusted_workspace_principal("tenant"),
                TaskGraphLimits(max_concurrency=1),
            )
        )

        await asyncio.sleep(0.05)
        assert not runner.started.is_set()
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        states = {item.node_id: item for item in snapshot.node_states}
        assert states["local"].status is TaskStatus.READY
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


def test_task_node_context_input_is_deeply_detached() -> None:
    context = TaskNodeContext(
        None,
        Principal("tester", "tenant"),
        "graph",
        "node",
        {"nested": {"value": 1}},
        {},
        "context-idempotency",
    )
    nested = context.input["nested"]
    assert isinstance(nested, dict)
    nested["value"] = 2
    assert context.input["nested"] == {"value": 1}


def test_task_function_uses_slots() -> None:
    async def run(context: TaskNodeContext[None]) -> None:
        del context

    function = TaskFunction("example.slots", 1, run)
    assert not hasattr(function, "__dict__")


@pytest.mark.asyncio
async def test_cancel_cleanup_failure_marks_operation_effect_unknown() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cancel-ledger", tenant_id="tenant")
    try:
        graph = TaskGraph("cancel-ledger", (TaskNode("node"),))
        await state.task.tasks.create_graph(graph, tenant_id="tenant")
        service = DefaultTaskService(
            state.task,
            _AllowAuthorization(),
            _FailingCancelLauncher(),
        )
        key = "cancel-ledger-request-0001"
        with pytest.raises(AIError) as error:
            await service.cancel_graph(
                graph.graph_id,
                CancelGraphRequest(
                    trusted_workspace_principal("tenant"),
                    key,
                ),
            )
        assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        operation = await state.task.operations.get(
            idempotency_key_digest(key),
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.status is OperationStatus.EFFECT_UNKNOWN
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_detached_bind_unknown_surfaces_background_failure() -> None:
    handler = _AgentTaskNodeHandler(object(), object(), object())
    started = asyncio.Event()
    release = asyncio.Event()

    class Control:
        async def bind_execution(self, execution_id: str) -> None:
            del execution_id
            started.set()
            await release.wait()
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    task = asyncio.create_task(
        handler._bind_execution(
            Control(),
            "execution",
            key=("tenant", "graph", "node"),
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    pending = handler.pending_background_tasks
    assert pending
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.sleep(0)
    failure = handler.background_failure
    assert failure is not None
    assert failure.code is ErrorCode.STORAGE_RECOVERY_REQUIRED


@pytest.mark.asyncio
async def test_bind_after_launch_ignores_expected_ownership_loss() -> None:
    handler = _AgentTaskNodeHandler(object(), object(), object())

    async def launched() -> object:
        return SimpleNamespace(execution_id="execution")

    class Control:
        async def bind_execution(self, execution_id: str) -> None:
            del execution_id
            raise AIError(ErrorCode.TASK_FENCE_STALE)

    launch_task = asyncio.create_task(launched())
    await handler._bind_after_launch(
        launch_task,
        ("tenant", "graph", "node"),
        Control(),
    )
    assert handler.background_failure is None
