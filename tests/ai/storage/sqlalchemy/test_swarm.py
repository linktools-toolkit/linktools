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
from linktools.ai.swarm.models import (
    AttemptStatus,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskAttempt,
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
            claimed = await store.claim_task("swarm-1", "agent-a")
            result = RunResult(output={"done": True}, token_usage={"input_tokens": 1}, metadata={"m": "n"})
            completed = await store.complete_task("t-1", result, expected_version=claimed.version)
            assert completed.status == SwarmTaskStatus.SUCCEEDED
            assert completed.result.output == {"done": True}
            assert completed.result.metadata == {"m": "n"}
            assert completed.version == claimed.version + 1

    asyncio.run(_run_case())


def test_fail_task_stores_error_and_increments_attempts(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1", attempts=0))
            claimed = await store.claim_task("swarm-1", "agent-a")
            err = RunErrorInfo(error_type="ValueError", message="boom", detail={"x": 1})
            failed = await store.fail_task("t-1", err, expected_version=claimed.version)
            assert failed.status == SwarmTaskStatus.FAILED
            assert failed.error.error_type == "ValueError"
            assert failed.error.message == "boom"
            assert failed.attempts == 1
            assert failed.version == claimed.version + 1

    asyncio.run(_run_case())


def test_complete_task_missing_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(SwarmTaskNotFoundError):
                await store.complete_task("nope", RunResult(output=None), expected_version=1)

    asyncio.run(_run_case())


def test_fail_task_missing_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(SwarmTaskNotFoundError):
                await store.fail_task("nope", RunErrorInfo(error_type="X", message="y"), expected_version=1)

    asyncio.run(_run_case())


def test_complete_task_with_fencing_token_requires_claimed_status(tmp_path):
    """G9: complete_task's fencing-token path (expected_version supplied) must
    also require status='claimed' -- a task already completed/failed by a
    racing writer must not be silently re-completed just because the version
    it was claimed under still matches (it won't, since a real transition
    also bumps the version, but the status check is the defense-in-depth
    the spec calls for)."""
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-a")
            # First completion succeeds under the fencing token.
            await store.complete_task(
                "t-1", RunResult(output="first"), expected_version=claimed.version,
            )
            # A stale caller retrying with the SAME (now-consumed) fencing
            # token must be rejected -- the task is no longer CLAIMED.
            with pytest.raises(SwarmConflictError):
                await store.complete_task(
                    "t-1", RunResult(output="stale-retry"), expected_version=claimed.version,
                )

    asyncio.run(_run_case())


def test_fail_task_with_fencing_token_requires_claimed_status(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-a")
            await store.complete_task(
                "t-1", RunResult(output="first"), expected_version=claimed.version,
            )
            with pytest.raises(SwarmConflictError):
                await store.fail_task(
                    "t-1", RunErrorInfo(error_type="X", message="y"),
                    expected_version=claimed.version,
                )

    asyncio.run(_run_case())


def test_complete_task_requires_expected_version_argument(tmp_path):
    """scenario (actionable-fix-contract): expected_version is now mandatory
    -- there is no more expected_version=None legacy bypass."""
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            with pytest.raises(TypeError):
                await store.complete_task("t-1", RunResult(output="x"))  # missing kwarg

    asyncio.run(_run_case())


def test_fail_task_requires_expected_version_argument(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            with pytest.raises(TypeError):
                await store.fail_task("t-1", RunErrorInfo(error_type="X", message="y"))  # missing kwarg

    asyncio.run(_run_case())


def test_complete_task_conflicts_on_wrong_active_run_id(tmp_path):
    """scenario (contract): a worker driving a since-superseded child Run (a
    stale active_run_id) must not complete the task even if it somehow still
    holds a matching version/status."""
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-a")
            await store.set_active_run("t-1", "run-real", expected_version=claimed.version)
            with pytest.raises(SwarmConflictError):
                await store.complete_task(
                    "t-1", RunResult(output="x"),
                    expected_version=claimed.version + 1,  # post-set_active_run version
                    active_run_id="run-stale",
                )

    asyncio.run(_run_case())


def test_fail_task_conflicts_on_wrong_active_run_id(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-a")
            await store.set_active_run("t-1", "run-real", expected_version=claimed.version)
            with pytest.raises(SwarmConflictError):
                await store.fail_task(
                    "t-1", RunErrorInfo(error_type="X", message="y"),
                    expected_version=claimed.version + 1,
                    active_run_id="run-stale",
                )

    asyncio.run(_run_case())


def test_complete_and_fail_concurrent_only_one_wins(tmp_path):
    """scenario (contract): complete_task and fail_task racing on the SAME
    CLAIMED task must both target the same fencing token -- only one can
    win; the loser observes a conflict, never a corrupted mixed state."""
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-a")

            results = await asyncio.gather(
                store.complete_task("t-1", RunResult(output="ok"), expected_version=claimed.version),
                store.fail_task(
                    "t-1", RunErrorInfo(error_type="X", message="y"),
                    expected_version=claimed.version,
                ),
                return_exceptions=True,
            )
            successes = [r for r in results if not isinstance(r, BaseException)]
            failures = [r for r in results if isinstance(r, BaseException)]
            assert len(successes) == 1, f"expected exactly one winner, got {results}"
            assert len(failures) == 1
            assert isinstance(failures[0], SwarmConflictError)
            final = await store.get_run("swarm-1")
            assert final is not None  # sanity: run row untouched by task race

    asyncio.run(_run_case())


def test_reclaim_expired_tasks_is_atomic_under_concurrent_races(tmp_path):
    """G9: reclaim_expired_tasks must not re-reclaim a task whose lease a
    concurrent renew_lease just pushed into the future -- the bulk UPDATE's
    WHERE clause re-evaluates lease_expires_at at UPDATE time, not at an
    earlier SELECT time."""
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-a", lease_seconds=0)
            # Renew the lease far into the future BEFORE reclaim runs.
            renewed = await store.renew_lease(
                "t-1", expected_version=claimed.version, lease_seconds=3600,
            )
            reclaimed = await store.reclaim_expired_tasks("swarm-1")
            assert reclaimed == (), "a freshly-renewed lease must not be reclaimed"
            still_claimed = await store.list_tasks("swarm-1", status=SwarmTaskStatus.CLAIMED)
            assert len(still_claimed) == 1
            assert still_claimed[0].version == renewed.version

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


def test_reclaim_expired_tasks_skips_non_expired_lease(tmp_path):
    """A CLAIMED task whose lease is still live is NOT reclaimed -- contract
    guard against blindly re-running side-effecting tasks."""
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-fresh"))
            claimed = await store.claim_task("swarm-1", "agent-x", lease_seconds=300)
            assert claimed is not None
            # Lease is 5 min in the future -> reclaim skips it.
            assert await store.reclaim_expired_tasks("swarm-1") == ()

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 8. SwarmTaskAttempt: record -> list round-trip (design note contract)
# ---------------------------------------------------------------------------


def _attempt(
    attempt_id: str = "att-1",
    task_id: str = "task-1",
    run_id: str = "run-1",
    agent_id: str = "agent-1",
    attempt: int = 1,
    status: AttemptStatus = AttemptStatus.RUNNING,
    started_at: "datetime | None" = None,
    finished_at: "datetime | None" = None,
    error: "RunErrorInfo | None" = None,
) -> SwarmTaskAttempt:
    return SwarmTaskAttempt(
        id=attempt_id,
        task_id=task_id,
        run_id=run_id,
        agent_id=agent_id,
        attempt=attempt,
        status=status,
        started_at=started_at or _now(),
        finished_at=finished_at,
        error=error,
    )


def test_record_attempt_then_list_roundtrip(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            started = _now()
            recorded = await store.record_attempt(_attempt(
                attempt_id="att-1", task_id="task-1", run_id="run-1",
                attempt=1, status=AttemptStatus.RUNNING, started_at=started,
            ))
            assert recorded.id == "att-1"
            listed = await store.list_attempts("task-1")
            assert len(listed) == 1
            assert listed[0].task_id == "task-1"
            assert listed[0].run_id == "run-1"
            assert listed[0].agent_id == "agent-1"
            assert listed[0].attempt == 1
            assert listed[0].status is AttemptStatus.RUNNING
            assert listed[0].started_at == started
            assert listed[0].finished_at is None

    asyncio.run(_run_case())


def test_record_attempt_upsert_updates_existing_row(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.record_attempt(_attempt(
                attempt_id="att-1", status=AttemptStatus.RUNNING,
            ))
            finished = _now()
            await store.record_attempt(_attempt(
                attempt_id="att-1", status=AttemptStatus.SUCCEEDED,
                finished_at=finished,
            ))
            listed = await store.list_attempts("task-1")
            assert len(listed) == 1
            assert listed[0].status is AttemptStatus.SUCCEEDED
            assert listed[0].finished_at == finished

    asyncio.run(_run_case())


def test_list_attempts_filters_by_task_id_and_orders_by_attempt(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.record_attempt(_attempt(
                attempt_id="att-1a", task_id="task-1", attempt=1,
            ))
            await store.record_attempt(_attempt(
                attempt_id="att-1b", task_id="task-1", attempt=2,
            ))
            await store.record_attempt(_attempt(
                attempt_id="att-2a", task_id="task-2", attempt=1,
            ))
            listed = await store.list_attempts("task-1")
            assert [a.attempt for a in listed] == [1, 2]
            assert all(a.task_id == "task-1" for a in listed)

    asyncio.run(_run_case())


def test_record_attempt_round_trips_error_field(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            err = RunErrorInfo(error_type="ValueError", message="boom", detail={"k": "v"})
            await store.record_attempt(_attempt(
                attempt_id="att-1", status=AttemptStatus.FAILED,
                finished_at=_now(), error=err,
            ))
            listed = await store.list_attempts("task-1")
            assert listed[0].error == err

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 9. renew_lease (design note contract)
# ---------------------------------------------------------------------------


def test_renew_lease_extends_lease_expires_at(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-1", lease_seconds=10.0)
            assert claimed is not None
            original_lease = claimed.lease_expires_at
            assert original_lease is not None
            renewed = await store.renew_lease(
                "t-1", expected_version=claimed.version, lease_seconds=60.0,
            )
            assert renewed.lease_expires_at is not None
            assert renewed.lease_expires_at > original_lease
            assert renewed.version == claimed.version + 1

    asyncio.run(_run_case())


def test_renew_lease_wrong_version_raises_conflict(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            claimed = await store.claim_task("swarm-1", "agent-1", lease_seconds=10.0)
            assert claimed is not None
            with pytest.raises(SwarmConflictError):
                await store.renew_lease(
                    "t-1", expected_version=claimed.version + 1, lease_seconds=60.0,
                )

    asyncio.run(_run_case())


def test_renew_lease_non_claimed_raises_invalid_transition(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            await store.create_run(_run())
            await store.create_task(_task(task_id="t-1"))
            with pytest.raises(InvalidSwarmTransitionError):
                await store.renew_lease(
                    "t-1", expected_version=1, lease_seconds=60.0,
                )

    asyncio.run(_run_case())
