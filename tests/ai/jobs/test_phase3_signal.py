#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 / §7.3 WaitSignal timeout: bounded waits persist a deadline,
reconcile_due expires them (retry or terminal cancel), and a matching signal
wakes the task. Parametrized over the file and sqlalchemy backends."""

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.filesystem.task import FilesystemTaskStore
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.task import SqlAlchemyTaskStore
from linktools.ai.jobs.models import (
    ActorChain,
    ActorRef,
    JobRecord,
    JobStatus,
    RetryPolicy,
    SideEffectPolicy,
    TaskBudget,
    TaskPrincipal,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskWaitCondition,
)
from linktools.ai.jobs.protocols import TaskSuccess, WaitSignal


class FakeClock:
    def __init__(self, start):
        self._t = start

    def now(self):
        return self._t

    def advance(self, seconds):
        self._t = self._t + timedelta(seconds=seconds)

    async def sleep(self, seconds):
        self.advance(seconds)


def _run(coro):
    return asyncio.run(coro)


async def _make_store(backend, tmp_path):
    clock = FakeClock(datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc))
    if backend == "file":
        return FilesystemTaskStore(tmp_path, clock=clock)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/sig.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyTaskStore(session_factory=factory, clock=clock)


@pytest.fixture(params=["file", "sqlite"])
def task_store(request, tmp_path):
    return asyncio.run(_make_store(request.param, tmp_path))


def _job(clock, *, retry=None):
    return JobRecord(
        id="j1",
        status=JobStatus.PENDING,
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        budget=TaskBudget(),
        root_task_id="t1",
        input_artifact_id=None,
        output_artifact_id=None,
        version=1,
        created_at=clock.now(),
        started_at=None,
        finished_at=None,
    )


def _task(clock, *, retry=None):
    return TaskRecord(
        id="t1",
        job_id="j1",
        parent_task_id=None,
        key="k",
        handler="h",
        status=TaskStatus.PENDING,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=(),
        retry_policy=retry or RetryPolicy(max_attempts=3),
        side_effect_policy=SideEffectPolicy(),
        attempt_count=0,
        available_at=clock.now(),
        lease_owner=None,
        lease_expires_at=None,
        fencing_token=0,
        active_attempt_id=None,
        timeout_seconds=None,
        resource_snapshots=(),
        version=1,
        created_at=clock.now(),
        updated_at=clock.now(),
    )


async def _to_waiting(store, clock, *, timeout=10.0, retry=None):
    await store.create_job(_job(clock), _task(clock, retry=retry))
    claimed = await store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
    await store.commit_success(
        claimed.claim,
        TaskSuccess(commands=(WaitSignal(name="ev", correlation_key="k", timeout_seconds=timeout),)),
    )
    return claimed.claim


def test_wait_signal_with_timeout_persists_deadline(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await _to_waiting(store=task_store, clock=clock, timeout=10.0)
        task = await task_store.get_task("t1")
        assert task.status == TaskStatus.WAITING
        assert task.wait_conditions == (TaskWaitCondition(name="ev", correlation_key="k"),)
        assert task.wait_deadline_at == clock.now() + timedelta(seconds=10.0)

    _run(run())


def test_wait_fields_round_trip(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await _to_waiting(store=task_store, clock=clock, timeout=10.0)
        task = await task_store.get_task("t1")
        assert task.wait_conditions == (TaskWaitCondition(name="ev", correlation_key="k"),)
        assert task.wait_deadline_at is not None

    _run(run())


def test_signal_timeout_retries_when_allowed(task_store) -> None:
    """A bounded wait whose deadline passes is re-queued (WAITING -> READY, paced
    by available_at) when TIMEOUT is retryable and attempts remain; the wait
    state is cleared."""
    clock = task_store._clock

    async def run() -> None:
        await _to_waiting(store=task_store, clock=clock, timeout=10.0, retry=RetryPolicy(max_attempts=3))
        clock.advance(20.0)  # past the 10s deadline
        await task_store.reconcile_due(now=clock.now(), limit=10)
        task = await task_store.get_task("t1")
        assert task.status == TaskStatus.READY
        assert task.wait_conditions == ()
        assert task.wait_deadline_at is None

    _run(run())


def test_signal_timeout_finalizes_when_not_retryable(task_store) -> None:
    """When the retry is exhausted (attempt_count == max_attempts), a timed-out
    wait is finalized (CANCELLED) -- the task is not parked forever."""
    clock = task_store._clock

    async def run() -> None:
        # max_attempts=1: the claim that produced the WAITING state already used
        # the single attempt, so the signal timeout cannot retry.
        await _to_waiting(store=task_store, clock=clock, timeout=10.0, retry=RetryPolicy(max_attempts=1))
        clock.advance(20.0)
        await task_store.reconcile_due(now=clock.now(), limit=10)
        task = await task_store.get_task("t1")
        assert task.status == TaskStatus.CANCELLED
        assert task.wait_conditions == ()
        assert task.wait_deadline_at is None

    _run(run())


def test_signal_before_deadline_wakes_task(task_store) -> None:
    """A matching signal that arrives BEFORE the deadline wakes the task and
    clears the wait state; a later deadline sweep does nothing."""
    clock = task_store._clock

    async def run() -> None:
        await _to_waiting(store=task_store, clock=clock, timeout=10.0)
        sig = TaskSignalRecord(
            id="s1",
            job_id="j1",
            name="ev",
            correlation_key="k",
            payload_artifact_id=None,
            created_at=clock.now(),
            consumed_by_task_id=None,
        )
        await task_store.submit_signal(sig)
        woken = await task_store.get_task("t1")
        assert woken.status == TaskStatus.READY
        assert woken.wait_conditions == ()
        # A later deadline sweep must not re-touch the woken task.
        clock.advance(20.0)
        handled = await task_store.reconcile_due(now=clock.now(), limit=10)
        assert handled == ()

    _run(run())


def test_unbounded_wait_is_not_expired(task_store) -> None:
    """A WaitSignal without a timeout has no deadline and is NOT finalized by
    reconcile_due -- it stays WAITING until a signal arrives."""
    clock = task_store._clock

    async def run() -> None:
        await _to_waiting(store=task_store, clock=clock, timeout=None)
        clock.advance(1000.0)
        handled = await task_store.reconcile_due(now=clock.now(), limit=10)
        assert handled == ()
        task = await task_store.get_task("t1")
        assert task.status == TaskStatus.WAITING

    _run(run())
