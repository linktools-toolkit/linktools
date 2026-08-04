#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskGraphEngine scheduling tests. A controllable fake runner records the
high-water mark of in-flight nodes so the concurrency cap is asserted directly;
asyncio.Event drives timing so no test sleeps to guess a schedule."""

import asyncio
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from linktools.ai.errors import (
    StorageError,
    SwarmLimitExceededError,
    TaskGraphCleanupError,
    TaskGraphInvariantError,
)
from linktools.ai.execution.domain import RunError
from linktools.ai.tasks.models import (
    DependencyFailurePolicy,
    TaskDependency,
    TaskExecution,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)
from linktools.ai.tasks.persistence.local import LocalTaskBackend
from linktools.ai.tasks.swarm.engine import (
    ControlGate,
    NodeRunRequest,
    NodeRunResult,
    TaskGraphEngine,
)
from linktools.ai.tasks.swarm.limits import SwarmLimits

from tests.ai.tasks.swarm._support import (
    NoopGate,
    RecordingRunner,
    make_plan,
    ready_executions,
)


def _limits(max_concurrency: int = 4, max_tasks: int = 50) -> SwarmLimits:
    return SwarmLimits(
        max_rounds=1,
        max_tasks=max_tasks,
        max_delegations=0,
        max_depth=0,
        max_concurrency=max_concurrency,
        max_total_tokens=None,
        max_total_cost=None,
        timeout_seconds=None,
    )


def _engine(store, runner, *, limits=None, owner="scheduler") -> "tuple[TaskGraphEngine, SwarmLimits]":
    lim = limits or _limits()
    gate = NoopGate()
    return TaskGraphEngine(
        store=store,
        runner=runner,
        gate=gate,
        limits=lim,
        owner=owner,
        parent_run_id="parent-run",
        parent_owner=owner,
        parent_fence=0,
    ), gate


@pytest.mark.asyncio
async def test_empty_plan_completes_immediately():
    store = LocalTaskBackend()
    runner = RecordingRunner()
    plan = TaskPlan("empty", ())
    await store.create_plan(plan, ())
    engine, _ = _engine(store, runner)
    usage = await engine.execute(plan)
    assert usage.input_tokens == 0
    assert runner.max_seen == 0


@pytest.mark.asyncio
async def test_linear_chain_runs_in_order():
    store = LocalTaskBackend()
    runner = RecordingRunner(results={"a": 1, "b": 2, "c": 3})
    plan = make_plan(
        ("a", "b", "c"),
        edges={"b": ("a",), "c": ("b",)},
    )
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner)
    await engine.execute(plan)
    order = [n for n, _ in runner.order]
    assert order.index("a") < order.index("b") < order.index("c")
    execs = {e.node_id: e for e in await store.list_executions(plan.id)}
    assert all(e.status is TaskStatus.COMPLETED for e in execs.values())


@pytest.mark.asyncio
async def test_diamond_runs_once_per_node():
    store = LocalTaskBackend()
    runner = RecordingRunner(results={"a": 1, "b": 2, "c": 3, "d": 4})
    plan = make_plan(
        ("a", "b", "c", "d"),
        edges={"b": ("a",), "c": ("a",), "d": ("b", "c")},
    )
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner)
    await engine.execute(plan)
    assert runner.counts == {"a": 1, "b": 1, "c": 1, "d": 1}


@pytest.mark.asyncio
async def test_skip_propagates_to_fixpoint():
    store = LocalTaskBackend()
    # node a fails; b depends on a (skip) -> skipped; c depends on b -> skipped.
    runner = RecordingRunner(
        results={"a": None},
        failures={"a": RunError("boom", "a failed")},
    )
    plan = make_plan(
        ("a", "b", "c"),
        edges={"b": ("a",), "c": ("b",)},
    )
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner)
    await engine.execute(plan)
    execs = {e.node_id: e for e in await store.list_executions(plan.id)}
    assert execs["a"].status is TaskStatus.FAILED
    assert execs["b"].status is TaskStatus.SKIPPED
    assert execs["b"].blocked_by == ("a",)
    assert execs["c"].status is TaskStatus.SKIPPED
    assert execs["c"].blocked_by == ("b",)


@pytest.mark.asyncio
async def test_proceed_degraded_runs_after_dependency_failure():
    store = LocalTaskBackend()
    runner = RecordingRunner(
        results={"a": None, "b": 2},
        failures={"a": RunError("boom", "a failed")},
    )
    plan = make_plan(
        ("a", "b"),
        edges={"b": (TaskDependency("a", DependencyFailurePolicy.PROCEED_DEGRADED),)},
    )
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner)
    await engine.execute(plan)
    execs = {e.node_id: e for e in await store.list_executions(plan.id)}
    assert execs["a"].status is TaskStatus.FAILED
    assert execs["b"].status is TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_skip_priority_over_proceed_degraded_in_mixed_set():
    store = LocalTaskBackend()
    runner = RecordingRunner(
        results={"a": None, "b": None, "c": 3},
        failures={
            "a": RunError("boom", "a"),
            "b": RunError("boom", "b"),
        },
    )
    # c depends on a (proceed_degraded) and b (skip) -> skip wins.
    plan = make_plan(
        ("a", "b", "c"),
        edges={
            "c": (
                TaskDependency("a", DependencyFailurePolicy.PROCEED_DEGRADED),
                TaskDependency("b", DependencyFailurePolicy.SKIP),
            )
        },
    )
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner)
    await engine.execute(plan)
    execs = {e.node_id: e for e in await store.list_executions(plan.id)}
    assert execs["c"].status is TaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_concurrency_cap_never_exceeded():
    store = LocalTaskBackend()
    runner = RecordingRunner(results={n: n for n in "abcdefgh"})
    plan = make_plan(tuple("abcdefgh"))
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner, limits=_limits(max_concurrency=2))
    await engine.execute(plan)
    assert runner.max_seen <= 2, runner.max_seen
    assert runner.max_seen == 2  # at least one pair runs concurrently


@pytest.mark.asyncio
async def test_completion_order_independent_of_node_order():
    store = LocalTaskBackend()
    plan = make_plan(("a", "b"))
    await store.create_plan(plan, ready_executions(plan))
    # runner finishes b before a, but collect encodes in plan order a,b
    runner = RecordingRunner(results={"a": 1, "b": 2}, fast={"b"})
    engine, _ = _engine(store, runner)
    await engine.execute(plan)
    from linktools.ai.tasks.swarm.aggregation import collect

    execs = {e.node_id: e for e in await store.list_executions(plan.id)}
    projection = collect(
        plan.id,
        tuple(
            {
                "node_id": node.id,
                "agent_id": node.payload.agent_id,
                "status": execs[node.id].status,
                "output": execs[node.id].result,
                "error": execs[node.id].error,
                "blocked_by": execs[node.id].blocked_by,
                "reason": execs[node.id].terminal_reason,
                "attempts": execs[node.id].attempt,
                "child_run_id": execs[node.id].active_run_id,
                "usage": {
                    "input_tokens": execs[node.id].usage.input_tokens,
                    "output_tokens": execs[node.id].usage.output_tokens,
                    "total_cost": execs[node.id].usage.total_cost,
                },
            }
            for node in plan.nodes
        ),
    )
    assert list(projection["nodes"].keys()) == ["a", "b"]


@pytest.mark.asyncio
async def test_usage_sums_all_terminal_attempts():
    store = LocalTaskBackend()
    runner = RecordingRunner(
        results={"a": 1, "b": None},
        usage={"a": TaskUsage(10, 5), "b": TaskUsage(3, 2)},
        failures={"b": RunError("boom", "b failed")},
    )
    plan = make_plan(("a", "b"))
    await store.create_plan(plan, ready_executions(plan))
    engine, _ = _engine(store, runner)
    usage = await engine.execute(plan)
    assert usage.input_tokens == 13
    assert usage.output_tokens == 7


@pytest.mark.asyncio
async def test_token_limit_aborts_via_accumulated_usage():
    from tests.ai.tasks.swarm._support import LimitGate

    store = LocalTaskBackend()
    runner = RecordingRunner(results={"a": 1, "b": 2}, usage={"a": TaskUsage(60, 10)})
    plan = make_plan(("a", "b"), edges={"b": ("a",)})
    await store.create_plan(plan, ready_executions(plan))
    gate = LimitGate(max_total_tokens=50)
    engine = TaskGraphEngine(
        store=store,
        runner=runner,
        gate=gate,
        limits=SwarmLimits(
            max_rounds=1,
            max_tasks=50,
            max_delegations=0,
            max_depth=0,
            max_concurrency=4,
            max_total_tokens=50,
            max_total_cost=None,
            timeout_seconds=None,
        ),
        owner="sch",
        parent_run_id="pr",
        parent_owner="sch",
        parent_fence=0,
    )
    with pytest.raises(SwarmLimitExceededError):
        await engine.execute(plan)
    # node a alone exceeds 50 input tokens (60), so b never runs
    assert runner.counts.get("b", 0) == 0


@pytest.mark.asyncio
async def test_no_busy_loop_raises_invariant_error():
    # A node stuck in a non-terminal state whose dependency never reaches a
    # terminal (impossible by construction; simulate by leaving a dependency
    # CLAIMED permanently with an active lease and never completing it).
    from datetime import datetime, timedelta, timezone
    from linktools.ai.storage.coordination.lease import Lease

    store = LocalTaskBackend()
    plan = make_plan(("a", "b"), edges={"b": ("a",)})
    executions = ready_executions(plan)
    now = datetime.now(timezone.utc)
    leased = Lease(
        owner="holder",
        fence=7,
        expires_at=now + timedelta(hours=1),
    )
    claimed = replace(executions[0], status=TaskStatus.CLAIMED, attempt=1, lease=leased)
    await store.create_plan(plan, (claimed, executions[1]))
    runner = RecordingRunner(results={"a": 1, "b": 2})
    engine, _ = _engine(store, runner, limits=_limits(max_concurrency=1))
    with pytest.raises(TaskGraphCleanupError) as raised:
        await engine.execute(plan)
    assert isinstance(raised.value.primary_error, TaskGraphInvariantError)
    assert isinstance(raised.value.cleanup_error, StorageError)
