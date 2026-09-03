#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for TaskGraph lease authority and explicit cancellation."""

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from ._task_test_helpers import admit_graph
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
        raise AssertionError(
            "runner must not start during explicit remote cancellation"
        )

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
        await admit_graph(state, graph)
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
async def test_explicit_cancel_cleans_running_node_without_local_scheduler_ownership() -> (
    None
):
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-remote-cancel", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("remote-cancel-graph", (TaskNode("node"),))
        await admit_graph(state, graph)
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
        await admit_graph(state, graph)
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
        await admit_graph(state, graph)
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


@pytest.mark.asyncio
async def test_inflight_node_does_not_suppress_durable_terminal_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(
        namespace="task-inflight-terminal-recheck", tenant_id="tenant"
    )
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("inflight-terminal-recheck", (TaskNode("node"),))
        await admit_graph(state, graph)
        runner = _BlockingRunner()
        monkeypatch.setattr(task_local, "_SCHEDULER_RECHECK_SECONDS", 0.01)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                trusted_workspace_principal("tenant"),
                TaskGraphLimits(),
            )
        )

        await asyncio.wait_for(runner.entered.wait(), 1)
        await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        await asyncio.wait_for(runner.cancelled.wait(), 1)

        snapshot = await repository.snapshot_graph(graph.graph_id, tenant_id="tenant")
        assert snapshot is not None
        assert snapshot.status is TaskStatus.CANCELLED
        assert snapshot.node_states[0].status is TaskStatus.CANCELLED
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_inflight_node_does_not_suppress_expired_foreign_lease_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReclaimRunner:
        def __init__(self) -> None:
            self.local_entered = asyncio.Event()
            self.local_cancelled = asyncio.Event()
            self.foreign_reclaimed = asyncio.Event()

        async def run(
            self,
            node: TaskNode,
            *,
            graph_id: str,
            principal: Principal,
            dependency_results: Mapping[str, TaskDependencyResult],
            control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del graph_id, principal, dependency_results, control
            if node.node_id == "foreign":
                self.foreign_reclaimed.set()
                return TaskNodeRunResult("a" * 64)
            self.local_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.local_cancelled.set()
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

    state = RuntimeState.in_memory()
    await state.initialize(
        namespace="task-inflight-foreign-reclaim", tenant_id="tenant"
    )
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "inflight-foreign-reclaim",
            (TaskNode("local"), TaskNode("foreign")),
        )
        await admit_graph(state, graph)
        stale = await repository.claim(
            graph.graph_id,
            "foreign",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=1,
        )
        runner = ReclaimRunner()
        monkeypatch.setattr(task_local, "_SCHEDULER_RECHECK_SECONDS", 0.01)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                trusted_workspace_principal("tenant"),
                TaskGraphLimits(max_concurrency=2),
            )
        )

        await asyncio.wait_for(runner.local_entered.wait(), 1)
        await asyncio.wait_for(runner.foreign_reclaimed.wait(), 2)

        foreign = None
        for _ in range(100):
            states = await repository.list_nodes(graph.graph_id, tenant_id="tenant")
            foreign = next(node for node in states if node.node_id == "foreign")
            if foreign.status is TaskStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)
        assert foreign is not None
        assert foreign.status is TaskStatus.SUCCEEDED
        assert foreign.fence == stale.fence + 1
        assert foreign.owner is None

        await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
        await asyncio.wait_for(runner.local_cancelled.wait(), 1)
    finally:
        if launcher is not None:
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_event_stream_rechecks_foreign_update_while_local_node_is_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-stream-foreign-recheck", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    stream = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "stream-foreign-recheck",
            (TaskNode("local"), TaskNode("foreign")),
        )
        await admit_graph(state, graph)
        foreign_lease = await repository.claim(
            graph.graph_id,
            "foreign",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        principal = trusted_workspace_principal("tenant")
        runner = _BlockingRunner()
        monkeypatch.setattr(task_local, "_SCHEDULER_RECHECK_SECONDS", 0.01)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                principal,
                TaskGraphLimits(max_concurrency=2),
            )
        )
        await asyncio.wait_for(runner.entered.wait(), 1)

        service = DefaultTaskService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
            local_waiter=launcher,
        )
        history = await service.list_graph_events(
            graph.graph_id,
            principal=principal,
            limit=100,
        )
        assert history.items
        stream = service.stream_graph_events(
            graph.graph_id,
            principal=principal,
            after_sequence=history.items[-1].sequence,
        )
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)

        await repository.complete(
            foreign_lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="b" * 64,
        )

        event = await asyncio.wait_for(pending, 1)
        assert event.node_id == "foreign"
        assert event.previous_status is TaskStatus.RUNNING
        assert event.status is TaskStatus.SUCCEEDED
        assert event.result_digest == "b" * 64
    finally:
        if stream is not None:
            await stream.aclose()
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await asyncio.wait_for(runner.cancelled.wait(), 1)
            await launcher.shutdown()
        await state.close()
