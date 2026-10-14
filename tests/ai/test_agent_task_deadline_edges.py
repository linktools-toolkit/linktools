#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deadline outcome races and bounded cancellation preserve authoritative results."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, RunBudget
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import ExecutionView
from linktools.ai.runtime import _agent_task
from linktools.ai.runtime._agent_task import RuntimeAgentTaskRunner
from linktools.ai.task import TaskNode, TaskNodeInvocation, TaskNodeRunError


_PRINCIPAL = Principal("deadline", "tenant", "service")


class _Execution:
    def __init__(self, view: ExecutionView) -> None:
        self.view = view
        self.waits = 0
        self.cancels = 0
        self.cancel_started = asyncio.Event()
        self.cancel_finished = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def wait(self, *, timeout_seconds: float | None = None) -> object:
        del timeout_seconds
        self.waits += 1
        if self.view.terminal_at is None:
            self.view = replace(
                self.view, status=ExecutionStatus.SUCCEEDED,
                terminal_at=datetime.now(timezone.utc),
            )
        return SimpleNamespace(result=SimpleNamespace(
            status=self.view.status, output="done", error_code=None,
        ))

    async def cancel(self) -> None:
        self.cancels += 1
        self.cancel_started.set()
        try:
            await self.release_cancel.wait()
        finally:
            self.cancel_finished.set()


def _view(started_at: datetime, terminal_at: datetime | None = None) -> ExecutionView:
    return ExecutionView(
        "execution", "agent",
        ExecutionStatus.STARTED if terminal_at is None else ExecutionStatus.SUCCEEDED,
        ExecutionLineageKind.RUN, None, "execution", None,
        started_at=started_at, terminal_at=terminal_at,
    )


def _runner(execution: _Execution) -> RuntimeAgentTaskRunner:
    # A controlled execution boundary makes result/readback races deterministic;
    # full Runtime construction and real cancellation are covered by integration tests.
    runner = RuntimeAgentTaskRunner.__new__(RuntimeAgentTaskRunner)

    async def inspect(execution_id: str, principal: Principal) -> ExecutionView:
        assert execution_id == "execution" and principal == _PRINCIPAL
        return execution.view

    async def get(execution_id: str, principal: Principal) -> _Execution:
        assert execution_id == "execution" and principal == _PRINCIPAL
        return execution

    runner._inspect_execution = inspect
    runner._get_execution = get
    runner._record_invocation = None
    return runner


def _invocation(timeout: float = 1.0) -> TaskNodeInvocation:
    return TaskNodeInvocation(
        TaskNode("agent", timeout_seconds=timeout), "graph", _PRINCIPAL, {}, {},
    )


@pytest.mark.asyncio
async def test_timed_success_does_not_repeat_result_observation() -> None:
    execution = _Execution(_view(datetime.now(timezone.utc)))
    result = await _runner(execution).wait_bound(_invocation(), "execution")
    assert result.execution_id == "execution"
    # Each SDK wait captures/replays the execution tree: duplicate consumption
    # would regress the accepted single-observation IO budget for normal completion.
    assert execution.waits == 1
    assert execution.cancels == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("on_time", (True, False))
async def test_late_read_uses_durable_completion_time(on_time: bool) -> None:
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    execution = _Execution(_view(
        started, started + timedelta(seconds=0.5 if on_time else 2),
    ))
    if on_time:
        result = await _runner(execution).wait_bound(_invocation(), "execution")
        assert result.execution_id == "execution"
    else:
        with pytest.raises(TaskNodeRunError) as caught:
            await _runner(execution).wait_bound(_invocation(), "execution")
        assert caught.value.code is ErrorCode.EXECUTION_WAIT_TIMEOUT
        assert caught.value.error_origin == "node"
    assert execution.cancels == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", (False, True))
async def test_unsettled_cancel_has_fixed_bound(
    monkeypatch: pytest.MonkeyPatch, interrupt: bool,
) -> None:
    monkeypatch.setattr(_agent_task, "_DEADLINE_CANCEL_TIMEOUT_SECONDS", 0.04)
    execution = _Execution(_view(datetime.now(timezone.utc) - timedelta(seconds=10)))
    waiting = asyncio.create_task(_runner(execution).wait_bound(_invocation(), "execution"))
    await asyncio.wait_for(execution.cancel_started.wait(), 1)
    if interrupt:
        for _ in range(3):
            waiting.cancel()
            await asyncio.sleep(0.005)
    try:
        done, _ = await asyncio.wait({waiting}, timeout=0.5)
        assert waiting in done, "caller cancellation must not renew the settlement deadline"
        with pytest.raises(TaskNodeRunError) as caught:
            waiting.result()
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert caught.value.safe_details["reason"] == "settlement_timeout"
        await asyncio.wait_for(execution.cancel_finished.wait(), 1)
    finally:
        execution.release_cancel.set()
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


def test_recovery_deadline_uses_original_start_and_tighter_parent() -> None:
    started = datetime(2020, 1, 1, tzinfo=timezone.utc)
    node = TaskNode("agent", timeout_seconds=3)
    policy = RuntimeAgentTaskRunner.deadline_policy(node, started, None)
    assert policy is not None
    assert policy[0] == started + timedelta(seconds=3)
    assert policy[1] is ErrorCode.EXECUTION_WAIT_TIMEOUT
    budget = RunBudget(deadline_at=started + timedelta(seconds=2))
    parent_policy = RuntimeAgentTaskRunner.deadline_policy(node, started, ("graph:graph", budget))
    assert parent_policy is not None
    assert parent_policy[0] == budget.deadline_at
    assert parent_policy[1] is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    assert parent_policy[2]["scope_id"] == "graph:graph"
