#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAS tests for local and real SQL task-store implementations."""

import asyncio
from datetime import timedelta

import pytest

from linktools.ai.errors import StorageConflictError
from linktools.ai.tasks.models import TaskExecution, TaskGraphNodePayload, TaskNode, TaskPlan, TaskStatus
from linktools.ai.tasks.persistence.local import LocalTaskBackend


def _plan(plan_id: str) -> TaskPlan:
    return TaskPlan(plan_id, (TaskNode("a", TaskGraphNodePayload("agent-a", "prompt")),))


async def _assert_claim_and_takeover_races(store) -> None:
    plan = _plan("race")
    execution = TaskExecution("execution", plan.id, "a", TaskStatus.READY)
    await store.create_plan(plan, (execution,))
    results = await asyncio.gather(
        store.claim_ready(execution.id, owner="worker-a", duration=timedelta(seconds=30)),
        store.claim_ready(execution.id, owner="worker-b", duration=timedelta(seconds=30)),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1

    reclaim_plan = _plan("reclaim")
    reclaim_execution = TaskExecution("reclaim-execution", reclaim_plan.id, "a", TaskStatus.READY)
    await store.create_plan(reclaim_plan, (reclaim_execution,))
    expired = await store.claim_ready(
        reclaim_execution.id,
        owner="old-worker",
        duration=timedelta(seconds=-1),
    )
    takeover_results = await asyncio.gather(
        store.take_over_expired_claim_for_reconcile(
            expired.id,
            owner="recovery-a",
            now=expired.lease.expires_at,
            duration=timedelta(seconds=30),
        ),
        store.take_over_expired_claim_for_reconcile(
            expired.id,
            owner="recovery-b",
            now=expired.lease.expires_at,
            duration=timedelta(seconds=30),
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in takeover_results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in takeover_results) == 1
    current = await store.get_execution(expired.id)
    assert current.owner in {"recovery-a", "recovery-b"}
    assert current.fence == expired.fence + 1


@pytest.mark.asyncio
async def test_local_claim_and_reconcile_takeover_are_single_winner():
    await _assert_claim_and_takeover_races(LocalTaskBackend())


@pytest.mark.asyncio
async def test_sql_claim_and_reconcile_takeover_are_single_winner(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.tasks.persistence.sqlalchemy import SqlAlchemyTaskBackend

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'task-race.db'}")
    store = SqlAlchemyTaskBackend(async_sessionmaker(engine, expire_on_commit=False))
    await store.initialize_storage(engine)
    try:
        await _assert_claim_and_takeover_races(store)
    finally:
        await engine.dispose()
