#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph activity notifications."""

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import linktools.ai.task._local as task_local
import linktools.ai.task._service_impl as task_service_impl
from linktools.ai.core import Principal, TaskStatus
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import (
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


class _AllowAuthorization:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _BlockingRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
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
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return TaskNodeRunResult("c" * 64)

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
async def test_scheduler_timeout_boundary_does_not_lose_completion_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-boundary-activity", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "boundary-activity",
            (TaskNode("local"), TaskNode("foreign")),
        )
        await repository.create_graph(graph, tenant_id="tenant")
        await repository.claim(
            graph.graph_id,
            "foreign",
            tenant_id="tenant",
            owner="remote-worker",
            lease_seconds=30,
        )
        runner = _BlockingRunner()
        scheduler_wait_entered = asyncio.Event()
        boundary_returned = asyncio.Event()
        original_wait = asyncio.wait

        async def boundary_wait(
            tasks: object,
            *,
            timeout: float | None = None,
            return_when: str = asyncio.ALL_COMPLETED,
        ) -> tuple[set[asyncio.Task[object]], set[asyncio.Task[object]]]:
            selected = tuple(tasks)  # type: ignore[arg-type]
            scheduler_wait = any(
                task.get_name().startswith("task-node-") for task in selected
            )
            if scheduler_wait and not boundary_returned.is_set():
                scheduler_wait_entered.set()
                while True:
                    states = await repository.list_nodes(
                        graph.graph_id,
                        tenant_id="tenant",
                    )
                    local = next(node for node in states if node.node_id == "local")
                    if local.status is TaskStatus.SUCCEEDED:
                        boundary_returned.set()
                        return set(), set(selected)
                    await asyncio.sleep(0)
            return await original_wait(
                selected,
                timeout=timeout,
                return_when=return_when,
            )

        monkeypatch.setattr(task_local.asyncio, "wait", boundary_wait)
        monkeypatch.setattr(task_local, "_SCHEDULER_RECHECK_SECONDS", 0.01)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                trusted_workspace_principal("tenant"),
                TaskGraphLimits(max_concurrency=2),
            )
        )

        await asyncio.wait_for(runner.entered.wait(), 1)
        await asyncio.wait_for(scheduler_wait_entered.wait(), 1)
        generation = launcher.graph_activity_generation(
            graph.graph_id,
            tenant_id="tenant",
        )
        assert generation is not None
        activity = asyncio.create_task(
            launcher.wait_graph_activity(
                graph.graph_id,
                tenant_id="tenant",
                after_generation=generation,
            )
        )
        runner.release.set()

        await asyncio.wait_for(boundary_returned.wait(), 1)
        await asyncio.wait_for(activity, 1)
        states = await repository.list_nodes(graph.graph_id, tenant_id="tenant")
        by_id = {node.node_id: node for node in states}
        assert by_id["local"].status is TaskStatus.SUCCEEDED
        assert by_id["foreign"].status is TaskStatus.RUNNING
    finally:
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_local_stream_observers_do_not_poll_durable_snapshots_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-observer-idle", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    streams = []
    pending: list[asyncio.Task[object]] = []
    try:
        repository = state.task.tasks
        graph = TaskGraph("observer-idle", (TaskNode("local"),))
        await repository.create_graph(graph, tenant_id="tenant")
        principal = trusted_workspace_principal("tenant")
        runner = _BlockingRunner()
        monkeypatch.setattr(task_local, "_SCHEDULER_RECHECK_SECONDS", 0.01)
        monkeypatch.setattr(task_service_impl, "_GRAPH_OBSERVATION_RECHECK_SECONDS", 0.01)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(TaskGraphLaunch(graph, principal, TaskGraphLimits()))
        await asyncio.wait_for(runner.entered.wait(), 1)
        await asyncio.sleep(0.05)

        snapshot_calls = 0
        original_snapshot = repository.snapshot_graph

        async def counting_snapshot(
            graph_id: str,
            *,
            tenant_id: str,
        ):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return await original_snapshot(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "snapshot_graph", counting_snapshot)
        service = DefaultTaskService(
            SimpleNamespace(tasks=repository),
            _AllowAuthorization(),
            local_waiter=launcher,
        )
        streams = [
            service.stream_graph(graph.graph_id, principal=principal),
            service.stream_graph(graph.graph_id, principal=principal),
        ]
        first = [await asyncio.wait_for(anext(stream), 1) for stream in streams]
        assert all(snapshot.node_states[0].status is TaskStatus.RUNNING for snapshot in first)
        assert snapshot_calls == 2

        pending = [asyncio.create_task(anext(stream)) for stream in streams]
        await asyncio.sleep(0.05)
        assert snapshot_calls == 2
        assert all(not task.done() for task in pending)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for stream in streams:
            await stream.aclose()
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await asyncio.wait_for(runner.cancelled.wait(), 1)
            await launcher.shutdown()
        await state.close()


@pytest.mark.asyncio
async def test_local_stream_observes_foreign_update_via_scheduler_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-observer-scheduler-notify", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    stream = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "observer-scheduler-notify",
            (TaskNode("local"), TaskNode("foreign")),
        )
        await repository.create_graph(graph, tenant_id="tenant")
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
        monkeypatch.setattr(task_service_impl, "_GRAPH_OBSERVATION_RECHECK_SECONDS", 60.0)
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
        stream = service.stream_graph(graph.graph_id, principal=principal)
        first = await asyncio.wait_for(anext(stream), 1)
        first_by_id = {node.node_id: node for node in first.node_states}
        assert first_by_id["local"].status is TaskStatus.RUNNING
        assert first_by_id["foreign"].status is TaskStatus.RUNNING

        await repository.complete(
            foreign_lease,
            tenant_id="tenant",
            execution_id=None,
            result_digest="b" * 64,
        )

        second = await asyncio.wait_for(anext(stream), 1)
        second_by_id = {node.node_id: node for node in second.node_states}
        assert second_by_id["local"].status is TaskStatus.RUNNING
        assert second_by_id["foreign"].status is TaskStatus.SUCCEEDED
    finally:
        if stream is not None:
            await stream.aclose()
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await asyncio.wait_for(runner.cancelled.wait(), 1)
            await launcher.shutdown()
        await state.close()
