#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/test_swarm.py — SqlAlchemySwarmStore contract.
Uses the `def test_x(): asyncio.run(_run())` style (sync test wrapper driving its
own event loop) so no pytest-asyncio mode config is needed."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import (
    InvalidSwarmTransitionError,
    SwarmConflictError,
    SwarmRunNotFoundError,
    SwarmTaskNotFoundError,
)
from linktools.ai.run.models import RunErrorInfo, RunResult
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.swarm import SqlAlchemySwarmStore
from linktools.ai.swarm_runtime.models import (
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _run(
    swarm_run_id: str = "swarm-1",
    status: SwarmStatus = SwarmStatus.PENDING,
    version: int = 1,
    round: int = 0,
) -> SwarmRun:
    now = _now()
    return SwarmRun(
        id=swarm_run_id,
        run_id="run-1",
        round=round,
        status=status,
        version=version,
        token_usage=TokenUsage(input_tokens=10, output_tokens=20, total_cost=Decimal("1.25")),
        cost=Decimal("1.25"),
        created_at=now,
        updated_at=now,
        metadata={"k": "v"},
    )


def _task(
    task_id: str = "task-1",
    swarm_run_id: str = "swarm-1",
    parent_task_id: "str | None" = None,
    status: SwarmTaskStatus = SwarmTaskStatus.PENDING,
    dependencies: "tuple[str, ...]" = (),
    assigned_agent_id: "str | None" = None,
    attempts: int = 0,
    version: int = 1,
) -> SwarmTask:
    now = _now()
    return SwarmTask(
        id=task_id,
        swarm_run_id=swarm_run_id,
        parent_task_id=parent_task_id,
        assigned_agent_id=assigned_agent_id,
        description="do thing",
        status=status,
        dependencies=dependencies,
        input=TaskInput(prompt="hi", metadata={"t": 1}),
        result=None,
        error=None,
        attempts=attempts,
        version=version,
        claimed_at=None,
        lease_expires_at=None,
        created_at=now,
        updated_at=now,
    )


@asynccontextmanager
async def _store_ctx(tmp_path):
    """Build a SqlAlchemySwarmStore against an in-file SQLite DB. The engine is
    disposed on exit so aiosqlite's background worker threads shut down before
    the per-test event loop closes (otherwise they call into a dead loop)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/swarm.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        yield SqlAlchemySwarmStore(session_factory=session_factory)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# 1. create_run -> get_run round-trip
# ---------------------------------------------------------------------------


def test_create_run_then_get_run_roundtrip(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            created = await store.create_run(_run())
            fetched = await store.get_run("swarm-1")
            assert fetched is not None
            assert fetched.id == created.id
            assert fetched.status == SwarmStatus.PENDING
            assert fetched.token_usage.input_tokens == 10
            assert fetched.token_usage.output_tokens == 20
            assert fetched.cost == Decimal("1.25")
            assert fetched.metadata == {"k": "v"}
            # datetime reattached with UTC tzinfo on read (aiosqlite strips it).
            assert fetched.created_at.tzinfo is not None

    asyncio.run(_run_case())


def test_get_run_missing_returns_none(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            assert await store.get_run("nope") is None

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 2. update_run: version advance, conflict, invalid transition, not-found
# ---------------------------------------------------------------------------


def test_update_run_advances_version_and_applies_fields(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            updated = await store.update_run(
                "swarm-1",
                expected_version=1,
                status=SwarmStatus.RUNNING,
                round=2,
                token_usage=TokenUsage(input_tokens=100, output_tokens=200, total_cost=Decimal("2")),
                cost=Decimal("9.99"),
                metadata={"new": "k"},
            )
            assert updated.version == 2
            assert updated.status == SwarmStatus.RUNNING
            assert updated.round == 2
            assert updated.token_usage.input_tokens == 100
            assert updated.token_usage.output_tokens == 200
            assert updated.cost == Decimal("9.99")
            assert updated.metadata == {"new": "k"}

    asyncio.run(_run_case())


def test_update_run_wrong_expected_version_raises_conflict(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            with pytest.raises(SwarmConflictError):
                await store.update_run("swarm-1", expected_version=99, status=SwarmStatus.RUNNING)

    asyncio.run(_run_case())


def test_update_run_invalid_transition_raises(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            # PENDING -> SUCCEEDED is not in ALLOWED_SWARM_TRANSITIONS.
            await store.create_run(_run())
            with pytest.raises(InvalidSwarmTransitionError):
                await store.update_run("swarm-1", expected_version=1, status=SwarmStatus.SUCCEEDED)

    asyncio.run(_run_case())


def test_update_run_missing_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(SwarmRunNotFoundError):
                await store.update_run("nope", expected_version=1, status=SwarmStatus.RUNNING)

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 3. create_task + list_tasks (with status filter)
# ---------------------------------------------------------------------------


def test_create_task_then_list(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-a"))
            await store.create_task(_task(task_id="t-b"))
            tasks = await store.list_tasks("swarm-1")
            ids = {t.id for t in tasks}
            assert ids == {"t-a", "t-b"}

    asyncio.run(_run_case())


def test_list_tasks_status_filter(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-pending", status=SwarmTaskStatus.PENDING))
            await store.create_task(
                _task(task_id="t-claimed", status=SwarmTaskStatus.CLAIMED, assigned_agent_id="agent-7"),
            )
            pending = await store.list_tasks("swarm-1", status=SwarmTaskStatus.PENDING)
            claimed = await store.list_tasks("swarm-1", status=SwarmTaskStatus.CLAIMED)
            assert {t.id for t in pending} == {"t-pending"}
            assert {t.id for t in claimed} == {"t-claimed"}

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 4. claim_task: PENDING -> CLAIMED, dependencies, empty -> None, lease
# ---------------------------------------------------------------------------


def test_claim_task_assigns_and_stamps(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-9")
            assert claimed is not None
            assert claimed.id == "t-1"
            assert claimed.status == SwarmTaskStatus.CLAIMED
            assert claimed.assigned_agent_id == "agent-9"
            assert claimed.claimed_at is not None
            assert claimed.version == 2

    asyncio.run(_run_case())


def test_claim_task_respects_dependencies(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            # t-dep not yet succeeded -> t-blocked must NOT be claimed.
            await store.create_task(_task(task_id="t-dep", status=SwarmTaskStatus.PENDING))
            await store.create_task(_task(task_id="t-blocked", dependencies=("t-dep",)))
            # First claim should pick t-dep (no deps), not t-blocked.
            first = await store.claim_task("swarm-1", "agent-1")
            assert first is not None
            assert first.id == "t-dep"
            # Now nothing claimable (t-blocked's only dep is CLAIMED, not SUCCEEDED).
            second = await store.claim_task("swarm-1", "agent-2")
            assert second is None

    asyncio.run(_run_case())


def test_claim_task_returns_none_when_empty(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            assert await store.claim_task("swarm-1", "agent-1") is None

    asyncio.run(_run_case())


def test_claim_task_lease_stamps_expiry(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-9", lease_seconds=30)
            assert claimed is not None
            assert claimed.lease_expires_at is not None
            assert claimed.lease_expires_at - claimed.claimed_at >= timedelta(seconds=29)

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 5. Atomic claim race: two sequential claims on ONE pending task
# ---------------------------------------------------------------------------


def test_atomic_claim_race_first_wins_second_gets_none(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-only"))
            # First claim flips status to 'claimed' via UPDATE...WHERE status='pending'.
            first = await store.claim_task("swarm-1", "agent-a")
            assert first is not None
            assert first.id == "t-only"
            # Second claim: the WHERE status='pending' clause no longer matches, so
            # the UPDATE hits 0 rows and claim_task returns None.
            second = await store.claim_task("swarm-1", "agent-b")
            assert second is None

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 6. complete_task + fail_task
# ---------------------------------------------------------------------------


def test_complete_task_stores_result(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            result = RunResult(output={"done": True}, token_usage={"input_tokens": 1}, metadata={"m": "n"})
            completed = await store.complete_task("t-1", result)
            assert completed.status == SwarmTaskStatus.SUCCEEDED
            assert completed.result.output == {"done": True}
            assert completed.result.metadata == {"m": "n"}
            assert completed.version == 2

    asyncio.run(_run_case())


def test_fail_task_stores_error_and_increments_attempts(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1", attempts=0))
            err = RunErrorInfo(error_type="ValueError", message="boom", detail={"x": 1})
            failed = await store.fail_task("t-1", err)
            assert failed.status == SwarmTaskStatus.FAILED
            assert failed.error.error_type == "ValueError"
            assert failed.error.message == "boom"
            assert failed.attempts == 1
            assert failed.version == 2

    asyncio.run(_run_case())


def test_complete_task_missing_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(SwarmTaskNotFoundError):
                await store.complete_task("nope", RunResult(output=None))

    asyncio.run(_run_case())


def test_fail_task_missing_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(SwarmTaskNotFoundError):
                await store.fail_task("nope", RunErrorInfo(error_type="X", message="y"))

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 7. reclaim_expired_tasks: expired lease -> back to PENDING
# ---------------------------------------------------------------------------


def test_reclaim_expired_tasks_flips_back_to_pending(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            # Claim with lease_seconds=0 so the lease is already expired by the
            # time reclaim runs (now > lease_expires_at immediately).
            await store.create_task(_task(task_id="t-stale"))
            claimed = await store.claim_task("swarm-1", "agent-x", lease_seconds=0)
            assert claimed is not None
            assert claimed.status == SwarmTaskStatus.CLAIMED

            reclaimed = await store.reclaim_expired_tasks("swarm-1")
            assert len(reclaimed) == 1
            assert reclaimed[0].id == "t-stale"
            assert reclaimed[0].status == SwarmTaskStatus.PENDING
            assert reclaimed[0].assigned_agent_id is None
            assert reclaimed[0].claimed_at is None
            assert reclaimed[0].lease_expires_at is None
            # After reclaim, the task should be claimable again.
            reclaimer = await store.claim_task("swarm-1", "agent-y")
            assert reclaimer is not None
            assert reclaimer.id == "t-stale"

    asyncio.run(_run_case())


def test_reclaim_expired_tasks_returns_empty_when_none_stale(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-fresh"))
            # No claimed tasks with expired leases.
            assert await store.reclaim_expired_tasks("swarm-1") == ()

    asyncio.run(_run_case())
