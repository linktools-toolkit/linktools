#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completion during a scheduler read must not delay ready dependents."""

from linktools.ai.runtime.state import RuntimeState
import asyncio

import pytest

from linktools.ai.core import Principal, PrincipalKind, TaskStatus, canonical_sha256
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.task import (
    LocalTaskGraphLauncher, TaskGraph, TaskGraphAdmission, TaskGraphLaunch,
    TaskGraphLimits, TaskGraphRequest, TaskGraphState, TaskNode, TaskNodeInvocation, TaskNodeView,
    TaskNodeRunControl, TaskNodeRunResult,
)
from linktools.ai.task import _local as task_local


@pytest.mark.asyncio
async def test_completed_dependency_during_state_read_starts_consumer_before_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    state = RuntimeState(state)
    await state.initialize(namespace="completion-race", tenant_id="tenant")
    release = asyncio.Event()
    consumer_started = asyncio.Event()
    stale_read_returned = asyncio.Event()
    idle_started = asyncio.Event()
    launcher: LocalTaskGraphLauncher | None = None

    class Runner:
        async def run(
            self, invocation: TaskNodeInvocation, *, control: TaskNodeRunControl,
        ) -> TaskNodeRunResult:
            del control
            if invocation.node.node_id == "hold":
                await release.wait()
            else:
                consumer_started.set()
            return TaskNodeRunResult(canonical_sha256("done"), f"execution-{invocation.node.node_id}")

        async def cancel(self, invocation: TaskNodeInvocation) -> None:
            del invocation

    try:
        graph = TaskGraph("completion-graph", (
            TaskNode("hold"), TaskNode("consumer", dependencies=("hold",)),
        ))
        principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
        request = TaskGraphRequest(graph, principal, idempotency_key="completion-race-1")
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
        repository = state.task.tasks
        launcher = LocalTaskGraphLauncher(repository, Runner(), owner="worker")
        original_read = repository.scheduler_state
        original_idle = launcher._wait_scheduler

        async def read(graph_id: str, *, tenant_id: str) -> TaskGraphState:
            snapshot = await original_read(graph_id, tenant_id=tenant_id)
            if not stale_read_returned.is_set() and any(
                node.node_id == "hold" and node.status is TaskStatus.RUNNING
                for node in snapshot.node_states
            ):
                # Finish the dependency after the read captured its running state.
                run = launcher._graphs[(tenant_id, graph_id)]
                release.set()
                await run.inflight["hold"].task
                stale_read_returned.set()
            return snapshot

        async def idle(run: task_local._GraphRun, states: tuple[TaskNodeView, ...]) -> None:
            if stale_read_returned.is_set() and not consumer_started.is_set():
                idle_started.set()
            await original_idle(run, states)

        monkeypatch.setattr(repository, "scheduler_state", read)
        monkeypatch.setattr(launcher, "_wait_scheduler", idle)
        await launcher.start(TaskGraphLaunch(graph.graph_id, principal, TaskGraphLimits()))
        graph_run = launcher._graphs[("tenant", graph.graph_id)]
        await asyncio.wait_for(stale_read_returned.wait(), 5)
        consumer = asyncio.create_task(consumer_started.wait())
        idle_wait = asyncio.create_task(idle_started.wait())
        try:
            await asyncio.wait({consumer, idle_wait}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            assert not idle_started.is_set(), "ready consumer was left behind an idle scheduler wait"
            assert consumer_started.is_set()
            assert graph_run.task is not None
            await asyncio.wait_for(asyncio.shield(graph_run.task), 5)
            completed = await original_read(graph.graph_id, tenant_id="tenant")
            assert completed.status is TaskStatus.SUCCEEDED
            assert all(node.status is TaskStatus.SUCCEEDED for node in completed.node_states)
        finally:
            consumer.cancel()
            idle_wait.cancel()
            await asyncio.gather(consumer, idle_wait, return_exceptions=True)
    finally:
        release.set()
        if launcher is not None:
            await launcher.shutdown()
        await state.close()
