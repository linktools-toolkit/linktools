#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A local invocation always joins the heartbeat it owns before returning."""

import asyncio

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import LocalTaskGraphLauncher, TaskGraph, TaskGraphLaunch, TaskGraphLimits, TaskNode, TaskNodeInvocation, TaskNodeRunControl, TaskNodeRunResult
from linktools.ai.task._local import _GraphRun, _LeaseState

from ._task_test_helpers import admit_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("handoff_error", "missing_execution", "deferred", "cancel"))
async def test_invocation_joins_owned_heartbeat_on_every_deferred_exit(
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    started, stopped = asyncio.Event(), asyncio.Event()
    heartbeat_tasks = []
    released = []
    failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)

    class Runner:
        async def run(self, invocation: TaskNodeInvocation, *, control: TaskNodeRunControl) -> TaskNodeRunResult:
            await started.wait()
            if outcome == "cancel":
                raise asyncio.CancelledError("runner cancelled")
            return TaskNodeRunResult("a" * 64, None if outcome == "missing_execution" else "execution", deferred=True)

    class Launcher(LocalTaskGraphLauncher):
        async def _heartbeat(self, *args, **kwargs) -> None:
            heartbeat_tasks.append(asyncio.current_task())
            started.set()
            try:
                await super()._heartbeat(*args, **kwargs)
            finally:
                stopped.set()

    async def release(execution_id, **kwargs):
        released.append(execution_id)

    async with Persistence.open("heartbeat-cleanup", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        node = TaskNode("node")
        await admit_graph(state, TaskGraph("graph", (node,)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="owner", lease_seconds=30)

        async def handoff(*args, **kwargs):
            raise failure

        if outcome == "handoff_error":
            monkeypatch.setattr(repository, "handoff_execution", handoff)
        launcher = Launcher(repository, Runner(), owner="owner", release_execution_hold=release)
        run = _GraphRun(TaskGraphLaunch("graph", Principal("actor", "tenant"), TaskGraphLimits()), "owner")
        try:
            if outcome == "deferred":
                await launcher._run_node(run, node, _LeaseState(lease))
            else:
                expected_type = asyncio.CancelledError if outcome == "cancel" else AIError
                with pytest.raises(expected_type) as raised:
                    await launcher._run_node(run, node, _LeaseState(lease))
                if outcome == "handoff_error":
                    assert raised.value is failure
                elif outcome == "missing_execution":
                    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert stopped.is_set()
            assert heartbeat_tasks and all(task.done() for task in heartbeat_tasks)
            current = (await repository.list_nodes("graph", tenant_id="tenant"))[0]
            assert current.fence == lease.fence
            assert current.status is (TaskStatus.WAITING if outcome == "deferred" else TaskStatus.RUNNING)
            assert released == []
        finally:
            for task in heartbeat_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*heartbeat_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_during_heartbeat_join_still_finishes_settled_hold() -> None:
    heartbeat_started, cleanup_started = asyncio.Event(), asyncio.Event()
    join_started, finish_cleanup = asyncio.Event(), asyncio.Event()
    released = []

    class Launcher(LocalTaskGraphLauncher):
        async def _heartbeat(self, *args, **kwargs) -> None:
            heartbeat_started.set()
            try:
                await super()._heartbeat(*args, **kwargs)
            finally:
                cleanup_started.set()
                while not finish_cleanup.is_set():
                    try:
                        await finish_cleanup.wait()
                    except asyncio.CancelledError:
                        join_started.set()

    async def release(execution_id, **kwargs):
        released.append(execution_id)

    async with Persistence.open("heartbeat-hold", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        node = TaskNode("node")
        await admit_graph(state, TaskGraph("graph", (node,)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="owner", lease_seconds=30)

        class Runner:
            async def run(self, invocation: TaskNodeInvocation, *, control: TaskNodeRunControl) -> TaskNodeRunResult:
                await heartbeat_started.wait()
                await control.handoff_execution("execution")
                await cleanup_started.wait()
                await repository.complete(
                    None, tenant_id="tenant", graph_id="graph", node_id="node",
                    expected_fence=lease.fence, execution_id="execution", result_digest="a" * 64,
                )
                return TaskNodeRunResult("a" * 64, "different-execution", deferred=True)

        launcher = Launcher(repository, Runner(), owner="owner", release_execution_hold=release)
        run = _GraphRun(TaskGraphLaunch("graph", Principal("actor", "tenant"), TaskGraphLimits()), "owner")
        pending = asyncio.create_task(launcher._run_node(run, node, _LeaseState(lease)))
        try:
            await asyncio.wait_for(join_started.wait(), 2)
            pending.cancel()
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
            assert released == ["execution"]
            current = (await repository.list_nodes("graph", tenant_id="tenant"))[0]
            assert current.status is TaskStatus.SUCCEEDED
            assert current.execution_id == "execution"
            assert current.fence == lease.fence
        finally:
            finish_cleanup.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
