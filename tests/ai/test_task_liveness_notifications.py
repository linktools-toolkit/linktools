#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for TaskGraph activity notifications."""

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
async def test_scheduler_retries_transient_reconcile_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="task-reconcile-retry", tenant_id="tenant")
    launcher: LocalTaskGraphLauncher | None = None
    try:
        repository = state.task.tasks
        graph = TaskGraph("reconcile-retry", (TaskNode("node"),))
        await admit_graph(state, graph)
        original_reconcile = repository.reconcile_graph
        attempts = 0

        async def reconcile(graph_id: str, *, tenant_id: str):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await original_reconcile(graph_id, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "reconcile_graph", reconcile)
        runner = _BlockingRunner()
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(
            TaskGraphLaunch(
                graph,
                trusted_workspace_principal("tenant"),
                TaskGraphLimits(),
            )
        )

        await asyncio.wait_for(runner.entered.wait(), 1)

        assert attempts >= 2
        assert launcher.owns_graph(graph.graph_id, tenant_id="tenant")
    finally:
        if launcher is not None:
            await repository.cancel_graph(graph.graph_id, tenant_id="tenant")
            await launcher.shutdown()
        await state.close()


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
        await admit_graph(state, graph)
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
async def test_local_event_stream_observers_do_not_poll_durable_snapshots_when_idle(
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
        await admit_graph(state, graph)
        principal = trusted_workspace_principal("tenant")
        runner = _BlockingRunner()
        monkeypatch.setattr(task_local, "_SCHEDULER_RECHECK_SECONDS", 0.01)
        launcher = LocalTaskGraphLauncher(repository, runner, owner="local-worker")
        await launcher.start(TaskGraphLaunch(graph, principal, TaskGraphLimits()))
        await asyncio.wait_for(runner.entered.wait(), 1)
        await asyncio.sleep(0.05)

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
        after_sequence = history.items[-1].sequence

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
        streams = [
            service.stream_graph_events(
                graph.graph_id,
                principal=principal,
                after_sequence=after_sequence,
            ),
            service.stream_graph_events(
                graph.graph_id,
                principal=principal,
                after_sequence=after_sequence,
            ),
        ]
        pending = [asyncio.create_task(anext(stream)) for stream in streams]
        await asyncio.sleep(0.05)

        assert snapshot_calls == 0
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
async def test_local_event_stream_observes_foreign_update_via_scheduler_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(
        namespace="task-observer-scheduler-notify", tenant_id="tenant"
    )
    launcher: LocalTaskGraphLauncher | None = None
    stream = None
    try:
        repository = state.task.tasks
        graph = TaskGraph(
            "observer-scheduler-notify",
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
