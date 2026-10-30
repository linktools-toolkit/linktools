#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task result holds follow durable settlement, not observer lifetime."""

import asyncio

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import (
    LocalTaskGraphLauncher, TaskGraphLaunch, TaskGraphLimits, TaskGraphState,
    TaskNode, TaskNodeRunResult, TaskNodeView, TaskRef,
)
from linktools.ai.task._local import _GraphRun
from linktools.ai.task.engine import TaskEngine


NODE = TaskNode("node", task=TaskRef("example.task", 1))
PRINCIPAL = Principal("actor", "tenant")


def _state(status: TaskStatus, *, fence: int = 1, execution_id: str | None = "execution") -> TaskGraphState:
    recovery = status is TaskStatus.RECOVERY_REQUIRED
    view = TaskNodeView(
        "graph", "node", (), status, None, fence, None,
        "a" * 64 if status is TaskStatus.SUCCEEDED else None,
        ErrorCode.STORAGE_RECOVERY_REQUIRED.value if recovery else None,
        "b" * 64 if recovery else None, execution_id=execution_id,
    )
    graph_status = status if status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED,
                                       TaskStatus.BLOCKED, TaskStatus.RECOVERY_REQUIRED} else TaskStatus.RUNNING
    return TaskGraphState("graph", graph_status, (NODE,), (view,))


class _Repository:
    def __init__(self, state: TaskGraphState) -> None:
        self.state = state
        self.completed = 0
        self.recovery = 0

    async def graph_state(self, graph_id: str, *, tenant_id: str) -> TaskGraphState:
        assert (graph_id, tenant_id) == ("graph", "tenant")
        return self.state

    async def complete(self, lease, **kwargs) -> None:
        assert lease is None
        assert kwargs["execution_id"] == "execution"
        self.completed += 1
        self.state = _state(TaskStatus.SUCCEEDED)

    async def mark_recovery_required(self, lease, **kwargs) -> None:
        assert lease is None
        assert kwargs["execution_id"] == "execution"
        self.recovery += 1
        self.state = _state(TaskStatus.RECOVERY_REQUIRED)


class _Runner:
    def __init__(self, outcome=None) -> None:
        self.outcome = outcome
        self.inspections = 0
        self.waits = 0
        self.entered = asyncio.Event()

    async def inspect_bound(self, invocation, execution_id):
        assert execution_id == "execution"
        assert invocation.principal == PRINCIPAL
        self.inspections += 1
        if isinstance(self.outcome, AIError):
            raise self.outcome
        return self.outcome

    async def wait_bound(self, invocation, execution_id):
        self.waits += 1
        self.entered.set()
        await asyncio.Event().wait()

    async def run(self, *args, **kwargs):
        pytest.fail("settling a bound execution must not run a new invocation")


def _run() -> _GraphRun:
    return _GraphRun(TaskGraphLaunch("graph", PRINCIPAL, TaskGraphLimits()), "owner")


@pytest.mark.asyncio
@pytest.mark.parametrize("status,fence,execution_id,released", (
    (TaskStatus.SUCCEEDED, 1, "execution", True),
    (TaskStatus.FAILED, 1, "execution", True),
    (TaskStatus.WAITING, 1, "execution", False),
    (TaskStatus.RECOVERY_REQUIRED, 1, "execution", False),
    (TaskStatus.READY, 2, "execution", False),
    (TaskStatus.READY, 2, None, False),
    (TaskStatus.READY, 2, "replacement", True),
))
async def test_hold_release_requires_durable_settlement(status, fence, execution_id, released) -> None:
    repository = _Repository(_state(TaskStatus.WAITING))
    calls = []

    async def release(execution_id: str, **kwargs) -> None:
        calls.append((execution_id, kwargs))

    launcher = LocalTaskGraphLauncher(repository, _Runner(), owner="owner", release_execution_hold=release)
    # The cleanup must read the current repository state, not its launch snapshot.
    repository.state = _state(status, fence=fence, execution_id=execution_id)
    await launcher._finish_execution_hold("execution", tenant_id="tenant", hold_id="task:graph:node",
                                          graph_id="graph", node_id="node", expected_fence=1)
    assert len(calls) == int(released)
    if released:
        assert calls == [("execution", {"tenant_id": "tenant", "hold_id": "task:graph:node"})]


@pytest.mark.asyncio
async def test_cancelled_bound_observer_keeps_waiting_execution_hold() -> None:
    repository = _Repository(_state(TaskStatus.WAITING))
    runner = _Runner()
    released = []

    async def release(execution_id: str, **kwargs) -> None:
        released.append(execution_id)

    launcher = LocalTaskGraphLauncher(repository, runner, owner="owner", release_execution_hold=release)
    observer = asyncio.create_task(launcher._wait_bound_node(_run(), NODE, "execution", 1))
    await asyncio.wait_for(runner.entered.wait(), 5)
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    assert released == []
    assert repository.state.node_states[0].status is TaskStatus.WAITING


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", (None, "unrelated_conflict"))
async def test_general_hold_conflict_is_not_treated_as_closed_retention(reason: str | None) -> None:
    error = AIError(ErrorCode.STORAGE_CONFLICT, safe_details={} if reason is None else {"reason": reason})

    async def acquire(*args, **kwargs) -> None:
        raise error

    repository, runner = _Repository(_state(TaskStatus.WAITING)), _Runner()
    launcher = LocalTaskGraphLauncher(repository, runner, owner="owner", acquire_execution_hold=acquire)
    with pytest.raises(AIError) as raised:
        await launcher._wait_bound_node(_run(), NODE, "execution", 1)
    assert raised.value is error
    assert runner.inspections == runner.waits == repository.completed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("available", "missing", "denied", "not-terminal"))
async def test_closed_retention_only_settles_existing_result(outcome: str) -> None:
    result = TaskNodeRunResult("a" * 64, "execution") if outcome == "available" else None
    if outcome in {"missing", "denied"}:
        result = AIError(ErrorCode.STORAGE_NOT_FOUND if outcome == "missing" else ErrorCode.AUTHORIZATION_DENIED)
    repository, runner = _Repository(_state(TaskStatus.WAITING)), _Runner(result)
    releases = []

    async def acquire(*args, **kwargs) -> None:
        raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"reason": "execution_retention_closed"})

    async def release(*args, **kwargs) -> None:
        releases.append(args)

    launcher = LocalTaskGraphLauncher(repository, runner, owner="owner",
                                     acquire_execution_hold=acquire, release_execution_hold=release)
    await launcher._wait_bound_node(_run(), NODE, "execution", 1)
    assert runner.inspections == 1
    assert runner.waits == 0
    assert releases == []
    assert repository.completed == int(outcome == "available")
    assert repository.recovery == int(outcome != "available")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (None, ErrorCode.STORAGE_NOT_FOUND, ErrorCode.STORAGE_CONFLICT,
                                     ErrorCode.AUTHORIZATION_DENIED))
async def test_terminal_graph_own_hold_cleanup_is_idempotent_without_hiding_other_errors(
    monkeypatch: pytest.MonkeyPatch, failure: ErrorCode | None,
) -> None:
    async with Persistence.open("hold-cleanup", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as persistence:
        async with TaskEngine.open(persistence=persistence) as engine:
            executor = engine._task_node_runtime
            calls = []

            async def release(execution_id: str, *, tenant_id: str, hold_id: str) -> None:
                calls.append((execution_id, tenant_id, hold_id))
                if failure is not None:
                    raise AIError(failure)

            monkeypatch.setattr(executor._execution_reads, "release_dependency_hold", release)
            state = _state(TaskStatus.SUCCEEDED)
            if failure in {None, ErrorCode.STORAGE_NOT_FOUND}:
                await executor.release_graph_dependencies(state, tenant_id="tenant")
                await executor.release_graph_dependencies(state, tenant_id="tenant")
                assert calls == [("execution", "tenant", "task:graph:node")] * 2
            else:
                with pytest.raises(AIError) as raised:
                    await executor.release_graph_dependencies(state, tenant_id="tenant")
                assert raised.value.code is failure
