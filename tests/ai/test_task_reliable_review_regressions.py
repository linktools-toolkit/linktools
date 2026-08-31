#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for TaskGraph lease authority and explicit cancellation."""

import asyncio
from collections.abc import Mapping

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
