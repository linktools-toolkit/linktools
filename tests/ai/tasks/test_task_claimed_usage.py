#!/usr/bin/env python3
"""Monotonic usage persistence for an in-flight task claim."""

from datetime import timedelta
from decimal import Decimal

import pytest

from linktools.ai.errors import (
    StorageConflictError,
    UsageObservationConflictError,
    UsageRegressionError,
)
from linktools.ai.tasks.models import (
    TaskExecution,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)
from linktools.ai.tasks.persistence.local import LocalTaskBackend


def _plan(name: str) -> TaskPlan:
    return TaskPlan(name, (TaskNode("a", TaskGraphNodePayload("agent-a", "prompt")),))


async def _assert_usage_contract(store) -> None:
    plan = _plan("claimed-usage")
    await store.create_plan(
        plan,
        (TaskExecution("execution", plan.id, "a", TaskStatus.READY),),
    )
    claimed = await store.claim_ready(
        "execution", owner="worker", duration=timedelta(minutes=1)
    )
    first = await store.record_claimed_usage(
        claimed.id,
        owner="worker",
        fence=claimed.fence,
        snapshot_revision=1,
        usage=TaskUsage(input_tokens=8, output_tokens=1),
    )
    assert first.usage.input_tokens == 8
    with pytest.raises(UsageObservationConflictError):
        await store.record_claimed_usage(
            claimed.id,
            owner="worker",
            fence=claimed.fence,
            snapshot_revision=1,
            usage=TaskUsage(input_tokens=7, output_tokens=1),
        )
    recovered = await store.record_claimed_usage(
        claimed.id,
        owner="worker",
        fence=claimed.fence,
        snapshot_revision=2,
        usage=TaskUsage(input_tokens=8, output_tokens=1, total_cost=Decimal("0.2")),
    )
    assert recovered.usage.total_cost == Decimal("0.2")
    unknown = await store.record_claimed_usage(
        claimed.id,
        owner="worker",
        fence=claimed.fence,
        snapshot_revision=3,
        usage=TaskUsage(input_tokens=9, output_tokens=2),
    )
    assert unknown.usage.total_cost is None
    assert unknown.active_run_id is None
    assert unknown.attempt == 1
    await store.bind_child_run(
        claimed.id,
        owner="worker",
        fence=claimed.fence,
        child_run_id="child",
    )

    with pytest.raises(UsageRegressionError):
        await store.complete(
            claimed.id,
            owner="worker",
            fence=claimed.fence,
            snapshot_revision=4,
            result={},
            usage=TaskUsage(input_tokens=8, output_tokens=2),
        )


@pytest.mark.asyncio
async def test_local_claimed_usage_is_monotonic():
    await _assert_usage_contract(LocalTaskBackend())


@pytest.mark.asyncio
async def test_sql_claimed_usage_is_monotonic(tmp_path):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.tasks.persistence.sqlalchemy import SqlAlchemyTaskBackend

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}")
    store = SqlAlchemyTaskBackend(
        async_sessionmaker(engine, expire_on_commit=False)
    )
    await store.initialize_storage(engine)
    try:
        await _assert_usage_contract(store)
    finally:
        await engine.dispose()
