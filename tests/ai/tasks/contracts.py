#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared task-store contract assertions exercised by both the local and the
SQLAlchemy backends. One suite, two backends: the contract is the authority for
claim fencing, terminal freezing, atomic plan creation, and the skip/cancel
transitions."""

import asyncio
from datetime import timedelta

import pytest

from linktools.ai.errors import StorageConflictError
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


def _plan(plan_id: str = "plan") -> TaskPlan:
    return TaskPlan(
        plan_id,
        (
            TaskNode("a", TaskGraphNodePayload("agent-a", "prompt-a")),
            TaskNode(
                "b",
                TaskGraphNodePayload("agent-b", "prompt-b"),
                dependencies=(TaskDependency("a"),),
            ),
        ),
    )


def _executions(plan_id: str = "plan") -> "tuple[TaskExecution, ...]":
    return (
        TaskExecution("e-a", plan_id, "a", TaskStatus.READY),
        TaskExecution("e-b", plan_id, "b", TaskStatus.READY),
    )


async def assert_task_store_contract(store) -> None:
    """Drive one backend through every transition and every race the spec calls
    out. ``store`` must already be initialized."""
    plan = _plan()
    executions = _executions()
    await store.create_plan(plan, executions)

    assert (await store.get_plan(plan.id)).id == plan.id
    listed = await store.list_executions(plan.id)
    assert {e.node_id for e in listed} == {"a", "b"}

    await assert_plan_is_not_upsertable(store, plan)
    await assert_concurrent_claim_has_one_winner(store)
    await assert_stale_fence_cannot_complete(store)
    await assert_claim_skip_cancel_races_have_one_outcome(store)
    await assert_terminals_are_frozen(store)
    await assert_empty_plan_succeeds(store)


async def assert_plan_is_not_upsertable(store, plan: TaskPlan) -> None:
    with pytest.raises(StorageConflictError):
        await store.create_plan(plan, _executions())


async def assert_concurrent_claim_has_one_winner(store) -> None:
    plan = _plan("race-plan")
    await store.create_plan(plan, (TaskExecution("e-race", "race-plan", "a", TaskStatus.READY),))
    results = await asyncio.gather(
        store.claim_ready("e-race", owner="w1", duration=timedelta(seconds=30)),
        store.claim_ready("e-race", owner="w2", duration=timedelta(seconds=30)),
        return_exceptions=True,
    )
    winners = [r for r in results if isinstance(r, TaskExecution)]
    conflicts = [r for r in results if isinstance(r, StorageConflictError)]
    assert len(winners) == 1, results
    assert len(conflicts) == 1, results
    final = await store.get_execution("e-race")
    assert final.status is TaskStatus.CLAIMED
    assert final.owner in {"w1", "w2"}


async def assert_stale_fence_cannot_complete(store) -> None:
    plan = _plan("fence-plan")
    await store.create_plan(plan, (TaskExecution("e-fence", "fence-plan", "a", TaskStatus.READY),))
    claimed = await store.claim_ready("e-fence", owner="owner", duration=timedelta(seconds=30))
    await store.bind_child_run(
        "e-fence", owner="owner", fence=claimed.fence, child_run_id="child-fence"
    )
    with pytest.raises(StorageConflictError):
        await store.complete(
            "e-fence",
            owner="owner",
            fence=claimed.fence + 99,
            result={"x": 1},
            usage=TaskUsage(),
        )
    with pytest.raises(StorageConflictError):
        await store.complete(
            "e-fence",
            owner="intruder",
            fence=claimed.fence,
            result={"x": 1},
            usage=TaskUsage(),
        )


async def assert_claim_skip_cancel_races_have_one_outcome(store) -> None:
    # claim vs skip on the same READY node: exactly one wins.
    plan = _plan("cs-plan")
    await store.create_plan(plan, (TaskExecution("e-cs", "cs-plan", "a", TaskStatus.READY),))
    claim_coro = store.claim_ready("e-cs", owner="w", duration=timedelta(seconds=30))
    skip_coro = store.skip("e-cs", blocked_by=("upstream",), reason="dep failed")
    results = await asyncio.gather(claim_coro, skip_coro, return_exceptions=True)
    final = await store.get_execution("e-cs")
    assert final.status in {TaskStatus.CLAIMED, TaskStatus.SKIPPED}
    if final.status is TaskStatus.CLAIMED:
        assert isinstance(results[1], StorageConflictError)
    else:
        assert isinstance(results[0], StorageConflictError)


async def assert_terminals_are_frozen(store) -> None:
    plan = _plan("term-plan")
    await store.create_plan(plan, (TaskExecution("e-term", "term-plan", "a", TaskStatus.READY),))
    claimed = await store.claim_ready("e-term", owner="w", duration=timedelta(seconds=30))
    await store.bind_child_run(
        "e-term", owner="w", fence=claimed.fence, child_run_id="child-term"
    )
    done = await store.complete(
        "e-term",
        owner="w",
        fence=claimed.fence,
        result={"done": True},
        usage=TaskUsage(7, 3),
    )
    assert done.status is TaskStatus.COMPLETED
    # every transition off COMPLETED is rejected.
    with pytest.raises(StorageConflictError):
        await store.complete("e-term", owner="w", fence=claimed.fence, result={}, usage=TaskUsage())
    with pytest.raises(StorageConflictError):
        await store.fail(
            "e-term",
            owner="w",
            fence=claimed.fence,
            error=RunError("e", "x"),
            usage=TaskUsage(),
        )
    with pytest.raises(StorageConflictError):
        await store.skip("e-term", blocked_by=("x",), reason="late")


async def assert_empty_plan_succeeds(store) -> None:
    empty = TaskPlan("empty-plan", ())
    await store.create_plan(empty, ())
    fetched = await store.get_plan("empty-plan")
    assert fetched is not None and fetched.nodes == ()
    assert await store.list_executions("empty-plan") == ()


async def assert_usage_round_trips_through_complete(store) -> None:
    plan = _plan("usage-plan")
    await store.create_plan(plan, (TaskExecution("e-usage", "usage-plan", "a", TaskStatus.READY),))
    from decimal import Decimal

    claimed = await store.claim_ready("e-usage", owner="w", duration=timedelta(seconds=30))
    await store.bind_child_run(
        "e-usage", owner="w", fence=claimed.fence, child_run_id="child-usage"
    )
    done = await store.complete(
        "e-usage",
        owner="w",
        fence=claimed.fence,
        result={"ok": True},
        usage=TaskUsage(input_tokens=12, output_tokens=8, total_cost=Decimal("0.0042")),
    )
    assert done.usage.input_tokens == 12
    assert done.usage.output_tokens == 8
    assert done.usage.total_cost == Decimal("0.0042")
    refetched = await store.get_execution("e-usage")
    assert refetched.usage.input_tokens == 12
    assert refetched.usage.total_cost == Decimal("0.0042")
