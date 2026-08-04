#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure and cleanup convergence tests for task-graph execution."""

import pytest

from linktools.ai.errors import (
    ChildExecutionPlatformError,
    StorageError,
    TaskGraphCleanupError,
)
from linktools.ai.tasks.models import TaskStatus, TaskUsage
from linktools.ai.tasks.persistence.local import LocalTaskBackend
from linktools.ai.tasks.swarm.engine import NodeRunRequest, NodeRunResult, TaskGraphEngine

from tests.ai.tasks.swarm._support import NoopGate, make_plan, ready_executions
from tests.ai.tasks.swarm.test_engine import _limits


class _PlatformFailureRunner:
    def __init__(self, *, cancel_fails: bool = False) -> None:
        self.cancel_fails = cancel_fails
        self.cancelled = []

    async def run(self, request: NodeRunRequest) -> NodeRunResult:
        raise ChildExecutionPlatformError(
            child_run_id=request.child_run_id,
            usage=TaskUsage(input_tokens=7, output_tokens=3),
            error_type="ProviderError",
            safe_message="child execution failed",
            cause=RuntimeError("provider detail"),
        )

    async def request_cancel(self, *, child_run_id, principal, reason) -> None:
        self.cancelled.append((child_run_id, reason))
        if self.cancel_fails:
            raise RuntimeError("cancel request failed")

    async def read_usage(self, *, child_run_id: str) -> TaskUsage:
        return TaskUsage(input_tokens=7, output_tokens=3)


@pytest.mark.asyncio
async def test_platform_failure_keeps_primary_error_and_converges_claim():
    store = LocalTaskBackend()
    plan = make_plan(("a",))
    await store.create_plan(plan, ready_executions(plan))
    runner = _PlatformFailureRunner()
    engine = TaskGraphEngine(
        store=store,
        runner=runner,
        gate=NoopGate(),
        limits=_limits(max_concurrency=1),
        owner="scheduler",
        parent_run_id="parent",
        parent_owner="scheduler",
        parent_fence=0,
    )

    with pytest.raises(ChildExecutionPlatformError) as raised:
        await engine.execute(plan)

    execution = (await store.list_executions(plan.id))[0]
    assert execution.status is TaskStatus.CANCELLED
    assert execution.usage == TaskUsage(input_tokens=7, output_tokens=3)
    assert engine.active_count == 0
    assert runner.cancelled == [(execution.active_run_id, "parent_failed")]
    assert raised.value.cause.args == ("provider detail",)


@pytest.mark.asyncio
async def test_cleanup_error_is_attached_without_masking_primary_error():
    store = LocalTaskBackend()
    plan = make_plan(("a",))
    await store.create_plan(plan, ready_executions(plan))
    runner = _PlatformFailureRunner(cancel_fails=True)
    engine = TaskGraphEngine(
        store=store,
        runner=runner,
        gate=NoopGate(),
        limits=_limits(max_concurrency=1),
        owner="scheduler",
        parent_run_id="parent",
        parent_owner="scheduler",
        parent_fence=0,
    )

    with pytest.raises(TaskGraphCleanupError) as raised:
        await engine.execute(plan)

    assert isinstance(raised.value.primary_error, ChildExecutionPlatformError)
    assert isinstance(raised.value.cleanup_error, StorageError)
    assert engine.active_count == 0
