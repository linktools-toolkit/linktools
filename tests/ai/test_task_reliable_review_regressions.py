#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for TaskGraph lease authority and explicit cancellation."""

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import linktools.ai.task._local as task_local
from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import (
    CancelGraphRequest,
    LocalTaskGraphLauncher,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphLaunch,
    TaskGraphLimits,
    TaskNode,
    TaskNodeRunControl,
    TaskNodeRunResult,
)
from linktools.ai.task._service_impl import DefaultTaskService
from linktools.ai.workspace import trusted_workspace_principal


class _RecordingRunner:
    def __init__(self) -> None:
        self.cancelled_nodes: list[str] = []

    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del node, graph_id, principal, dependency_results, control
        raise AssertionError("runner must not start during explicit remote cancellation")

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> None:
        del graph_id, principal, dependency_results
        self.cancelled_nodes.append(node.node_id)


class _BlockingRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        del node, graph_id, principal, dependency_results, control
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> None:
        del node, graph_id, principal, dependency_results


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _FailingLocalWaiter:
    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        del graph_id, tenant_id
        return True

    def graph_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int:
        del graph_id, tenant_id
        return 0

    async def wait_graph_activity(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_generation: "int | None" = None,
    ) -> None:
        del graph_id, tenant_id, after_generation
        raise AIError(ErrorCode.INTERNAL_ERROR)


@pytest.mark.asyncio
async def test_explicit_cancel_keeps_failed_node_but_marks_graph_cancelled() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-cancel-after-failure", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "cancel-after-failure",
            (TaskNode("failed"), TaskNode("active")),
        )
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
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        by_id = {node.node_id: node for node in snapshot.node_states}
        assert view.status is TaskStatus.CANCELLED
        assert snapshot.status is TaskStatus.CANCELLED
        assert by_id["failed"].status is TaskStatus.FAILED
        assert by_id["active"].status is TaskStatus.CANCELLED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_explicit_cancel_cleans_running_node_without_local_scheduler_ownership() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-remote-cancel", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("remote-cancel-graph", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
        await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        runner = _RecordingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        principal = trusted_workspace_principal("tenant")
        await repository.cancel_graph(graph.graph_id, tenant_id="tenant")

        view = await launcher.cancel(
            graph.graph_id,
            CancelGraphRequest(principal, "remote-cancel-request-0001"),
        )

        assert view.status is TaskStatus.CANCELLED
        assert runner.cancelled_nodes == ["node"]
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.node_states[0].status is TaskStatus.CANCELLED
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_heartbeat_lease_loss_cancels_runner_without_terminal_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-heartbeat-loss", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("heartbeat-loss-graph", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
        principal = trusted_workspace_principal("tenant")
        runner = _BlockingRunner()

        async def fail_renew(
            lease: object,
            *,
            tenant_id: str,
            lease_seconds: int,
        ) -> object:
            del lease, tenant_id, lease_seconds
            raise AIError(ErrorCode.TASK_FENCE_STALE)

        monkeypatch.setattr(task_local, "_HEARTBEAT_SECONDS", 0.01)
        monkeypatch.setattr(repository, "renew", fail_renew)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(TaskGraphLaunch(graph, principal, TaskGraphLimits()))

        await asyncio.wait_for(runner.entered.wait(), 1)
        await asyncio.wait_for(runner.cancelled.wait(), 1)
        await asyncio.sleep(0)

        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        node = snapshot.node_states[0]
        assert node.status is TaskStatus.RUNNING
        assert node.result_digest is None
        assert node.error_code is None
        assert node.error_digest is None
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_nonterminal_waiter_failure_propagates_after_fresh_snapshot() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-waiter-failure", tenant_id="tenant")
    try:
        repository = state.task.tasks
        graph = TaskGraph("waiter-failure-graph", (TaskNode("node"),))
        await repository.create_graph(graph, tenant_id="tenant")
        await repository.claim(
            graph.graph_id,
            "node",
            tenant_id="tenant",
            owner="local-worker",
            lease_seconds=30,
        )
        principal = trusted_workspace_principal("tenant")
        service = DefaultTaskService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
            local_waiter=_FailingLocalWaiter(),
        )

        with pytest.raises(AIError) as error:
            await service.wait_graph(
                graph.graph_id,
                principal=principal,
                timeout_seconds=1,
            )

        assert error.value.code is ErrorCode.INTERNAL_ERROR
        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.status is TaskStatus.RUNNING
    finally:
        await state.close()
