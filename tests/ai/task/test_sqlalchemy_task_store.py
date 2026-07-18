#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyTaskStore contract (plan section 28 phase-4 acceptance).

Exercises the same reliable-task invariants as the FileTaskStore suite, over an
in-memory SQLite backend. The CAS ``UPDATE ... WHERE status='ready'`` + rowcount
is the atomic claim that stops two workers taking one task; the fencing
``UPDATE ... WHERE status='claimed' AND lease_owner AND ... AND fencing_token``
stops a stale worker overwriting a new result.
"""

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.task import SqlAlchemyTaskStore
from linktools.ai.task.models import (
    ActorChain,
    ActorRef,
    JobRecord,
    JobStatus,
    RetryPolicy,
    ScopeSet,
    SideEffectPolicy,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
    TaskRecord,
    TaskStatus,
    TaskWaitCondition,
)
from linktools.ai.task.protocols import TaskFailure, TaskSuccess
from linktools.ai.task.store import TaskClaimLostError


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self._t = start

    def now(self) -> datetime:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t = self._t + timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def _job(clock, *, job_id="j1", root_task_id="t1") -> JobRecord:
    return JobRecord(
        id=job_id,
        status=JobStatus.PENDING,
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        budget=TaskBudget(),
        root_task_id=root_task_id,
        input_artifact_id=None,
        output_artifact_id=None,
        version=1,
        created_at=clock.now(),
        started_at=None,
        finished_at=None,
    )


def _task(clock, *, task_id="t1", job_id="j1") -> TaskRecord:
    return TaskRecord(
        id=task_id,
        job_id=job_id,
        parent_task_id=None,
        key="k",
        handler="runtime",
        status=TaskStatus.PENDING,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=(),
        retry_policy=RetryPolicy(max_attempts=2),
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


@pytest.fixture
def task_store(tmp_path):
    clock = FakeClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")

    asyncio.run(_create_tables(engine))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyTaskStore(session_factory=factory, clock=clock)


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _run(coro):
    return asyncio.run(coro)


def test_create_claim_complete(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is not None
        assert claimed.task.status == TaskStatus.CLAIMED
        assert claimed.task.fencing_token == 1
        done = await task_store.commit_success(claimed.claim, TaskSuccess())
        assert done.status == TaskStatus.SUCCEEDED
        assert (await task_store.get_job("j1")).status == JobStatus.SUCCEEDED

    _run(run())


def test_two_workers_cannot_claim_same_task(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        first = await task_store.claim(
            worker_id="w1", now=clock.now(), lease_seconds=30
        )
        # A second worker claims BEFORE the first completes -- the CAS WHERE
        # status='ready' finds 0 rows (the task is now 'claimed').
        second = await task_store.claim(
            worker_id="w2", now=clock.now(), lease_seconds=30
        )
        assert first is not None
        assert second is None

    _run(run())


def test_stale_fencing_rejected_after_reclaim(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        first = await task_store.claim(
            worker_id="w1", now=clock.now(), lease_seconds=30
        )
        clock.advance(60)
        await task_store.recover_expired(now=clock.now(), limit=10)
        reclaimed = await task_store.claim(
            worker_id="w2", now=clock.now(), lease_seconds=30
        )
        assert reclaimed.task.fencing_token == first.task.fencing_token + 1
        with pytest.raises(TaskClaimLostError):
            await task_store.commit_success(first.claim, TaskSuccess())
        await task_store.commit_success(reclaimed.claim, TaskSuccess())

    _run(run())


def test_transient_failure_retries_then_succeeds(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        first = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        failed = await task_store.commit_failure(
            first.claim,
            TaskFailure(
                kind=TaskFailureKind.TRANSIENT, error_type="NetError", message="boom"
            ),
        )
        assert failed.status == TaskStatus.RETRY_WAIT
        clock.advance(60)
        retry = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert retry is not None and retry.task.attempt_count == 2
        await task_store.commit_success(retry.claim, TaskSuccess())
        assert (await task_store.get_task("t1")).status == TaskStatus.SUCCEEDED

    _run(run())


def test_cancel(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        assert (await task_store.request_cancel("j1")).status == JobStatus.CANCELLED
        assert (await task_store.get_task("t1")).status == TaskStatus.CANCELLED

    _run(run())


def test_recover_expired(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        clock.advance(60)
        recovered = await task_store.recover_expired(now=clock.now(), limit=10)
        assert len(recovered) == 1
        assert recovered[0].status == TaskStatus.READY
        attempts = await task_store.list_attempts("t1")
        assert attempts[0].status.value == "superseded"
        # Re-claimable with bumped fencing.
        reclaimed = await task_store.claim(
            worker_id="w2", now=clock.now(), lease_seconds=30
        )
        assert reclaimed is not None
        assert reclaimed.task.fencing_token == claimed.task.fencing_token + 1

    _run(run())


def test_attempt_audit_complete(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        # Two attempts (transient failure + retry success).
        c1 = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        await task_store.commit_failure(
            c1.claim,
            TaskFailure(kind=TaskFailureKind.TRANSIENT, error_type="E", message="m"),
        )
        clock.advance(60)
        c2 = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        await task_store.commit_success(c2.claim, TaskSuccess())
        attempts = await task_store.list_attempts("t1")
        assert len(attempts) == 2
        assert attempts[0].status.value == "failed"
        assert attempts[1].status.value == "succeeded"
        transitions = await task_store.list_transitions("j1")
        # created + claimed + failed/retry-wait + retry-due + claimed + succeeded
        assert len(transitions) >= 4

    _run(run())


def test_non_utc_timestamps_round_trip(task_store) -> None:
    # C1 regression: a non-UTC created_at must store/round-trip to the same UTC
    # instant (not the wall-clock value). 08:00-05:00 == 13:00 UTC.
    from datetime import timezone as tz

    clock = task_store._clock

    async def run() -> None:
        offset = tz(timedelta(hours=-5))
        wall = datetime(2026, 7, 16, 8, 0, tzinfo=offset)
        job = _job(clock)
        job = JobRecord(
            id=job.id,
            status=job.status,
            principal=job.principal,
            actor_chain=job.actor_chain,
            budget=job.budget,
            root_task_id=job.root_task_id,
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=wall,
            started_at=None,
            finished_at=None,
        )
        await task_store.create_job(job, _task(clock))
        got = await task_store.get_job("j1")
        assert got is not None
        assert got.created_at == wall  # same UTC instant, not the wall value
        assert got.created_at.utcoffset() == timedelta(0)  # normalized to UTC

    _run(run())


def test_cancel_moves_in_flight_claimed_task_to_cancelling(task_store) -> None:
    """Mirror of the file-store cancel test: an in-flight (CLAIMED) task moves to
    CANCELLING; the job stays CANCELLING; recovery finalizes both to CANCELLED."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is not None
        job = await task_store.request_cancel("j1", reason="user")
        assert job.status == JobStatus.CANCELLING
        assert (await task_store.get_task("t1")).status == TaskStatus.CANCELLING
        clock.advance(60)  # lease expires
        await task_store.recover_expired(now=clock.now(), limit=10)
        assert (await task_store.get_task("t1")).status == TaskStatus.CANCELLED
        assert (await task_store.get_job("j1")).status == JobStatus.CANCELLED

    _run(run())


def test_create_task_command_creates_child(task_store) -> None:
    from linktools.ai.task.protocols import CreateTask

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(commands=(CreateTask(key="child", handler="evidence"),)),
        )
        tasks = await task_store.list_tasks("j1")
        children = [t for t in tasks if t.parent_task_id == "t1"]
        assert len(children) == 1
        assert children[0].status == TaskStatus.READY

    _run(run())


def test_wait_signal_command_transitions_to_waiting(task_store) -> None:
    from linktools.ai.task.protocols import TaskSuccess, WaitSignal

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        result = await task_store.commit_success(
            claimed.claim,
            TaskSuccess(commands=(WaitSignal(name="review", correlation_key="c1"),)),
        )
        assert result.status == TaskStatus.WAITING
        # The wait condition is persisted as first-class state so submit_signal
        # can match it; an unbounded wait has no deadline.
        assert result.wait_conditions == (
            TaskWaitCondition(name="review", correlation_key="c1"),
        )
        assert result.wait_deadline_at is None

    _run(run())


def test_signal_wakes_matching_waiting_task(task_store) -> None:
    from linktools.ai.task.models import TaskSignalRecord
    from linktools.ai.task.protocols import TaskSuccess, WaitSignal

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(
                commands=(WaitSignal(name="approval", correlation_key="case-1"),)
            ),
        )
        assert (await task_store.get_task("t1")).status == TaskStatus.WAITING
        # Submit a matching signal -> task wakes to READY.
        await task_store.submit_signal(
            TaskSignalRecord(
                id="sig-1",
                job_id="j1",
                name="approval",
                correlation_key="case-1",
                payload_artifact_id=None,
                created_at=clock.now(),
                consumed_by_task_id=None,
            )
        )
        assert (await task_store.get_task("t1")).status == TaskStatus.READY
        # A non-matching signal does not wake another waiter.
        await task_store.submit_signal(
            TaskSignalRecord(
                id="sig-2",
                job_id="j1",
                name="wrong",
                correlation_key="nope",
                payload_artifact_id=None,
                created_at=clock.now(),
                consumed_by_task_id=None,
            )
        )

    _run(run())


def test_dependency_resolution_promotes_pending_to_ready(task_store) -> None:
    from linktools.ai.task.protocols import CreateTask, TaskSuccess

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        # Root succeeds and creates a child that depends on the root.
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(
                commands=(
                    CreateTask(key="child", handler="echo", dependencies=("t1",)),
                )
            ),
        )
        tasks = await task_store.list_tasks("j1")
        child = [t for t in tasks if t.id != "t1"][0]
        # Root (t1) is SUCCEEDED -> dependency satisfied -> child should be READY.
        assert child.status == TaskStatus.READY

    _run(run())


def test_missing_dependency_does_not_crash(task_store) -> None:
    """C1 regression: _resolve_dependencies_sql must tolerate a missing dep ref."""
    from linktools.ai.task.protocols import CreateTask, TaskSuccess

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        # A handler returns a child referencing a nonexistent dependency.
        # This must NOT crash; the child stays PENDING.
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(
                commands=(
                    CreateTask(
                        key="bad-child",
                        handler="echo",
                        dependencies=("nonexistent-task",),
                    ),
                )
            ),
        )
        tasks = await task_store.list_tasks("j1")
        child = [t for t in tasks if t.id != "t1"][0]
        assert child.status == TaskStatus.PENDING

    _run(run())


def test_commit_success_rejects_too_many_commands(task_store) -> None:
    """Wired command-count limit (section 17.5): an outcome carrying more than
    MAX_COMMANDS commands is rejected before any write."""
    from linktools.ai.task.protocols import CreateTask, TaskSuccess
    from linktools.ai.task.validation import MAX_COMMANDS

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        too_many = tuple(
            CreateTask(key=f"k{i}", handler="h") for i in range(MAX_COMMANDS + 1)
        )
        with pytest.raises(ValueError, match="commands"):
            await task_store.commit_success(
                claimed.claim, TaskSuccess(commands=too_many)
            )

    _run(run())


def test_commit_success_rejects_oversized_output_artifact(task_store) -> None:
    """Wired output-payload limit (section 17.5): an output artifact larger than
    the ceiling is rejected at commit."""
    from linktools.ai.artifact.models import ArtifactRef
    from linktools.ai.task.protocols import TaskSuccess
    from linktools.ai.task.validation import MAX_OUTPUT_PAYLOAD_BYTES

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        big = ArtifactRef(
            id="a1",
            sha256="x" * 64,
            media_type="text/plain",
            size=MAX_OUTPUT_PAYLOAD_BYTES + 1,
        )
        with pytest.raises(ValueError, match="exceeds"):
            await task_store.commit_success(
                claimed.claim, TaskSuccess(output_artifact=big)
            )

    _run(run())


def test_submit_signal_rejects_oversized_metadata(task_store) -> None:
    """Wired signal-metadata limit (section 17.5): inline signal JSON above the
    metadata ceiling is rejected before the signal is persisted."""
    from linktools.ai.task.models import TaskSignalRecord

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        signal = TaskSignalRecord(
            id="s1",
            job_id="j1",
            name="n",
            correlation_key="c",
            payload_artifact_id=None,
            created_at=clock.now(),
            consumed_by_task_id=None,
            metadata={"big": "x" * (256 * 1024)},
        )
        with pytest.raises(ValueError, match="exceeds"):
            await task_store.submit_signal(signal)

    _run(run())


def test_child_task_delegated_scopes_narrow_and_actor_appends(task_store) -> None:
    """A child task's delegated scopes are the intersection of the parent's
    scopes and the command's requested scopes (never a union -- an ungranted
    scope is dropped, not added), and its actor chain appends the handler that
    created it."""
    from linktools.ai.task.protocols import CreateTask, TaskSuccess

    clock = task_store._clock

    async def run() -> None:
        job = JobRecord(
            id="j1",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(
                actors=(ActorRef("user", "alice"),),
                delegated_scopes=ScopeSet.of("read", "write", "exec"),
            ),
            budget=TaskBudget(),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=clock.now(),
            started_at=None,
            finished_at=None,
        )
        await task_store.create_job(job, dataclasses.replace(
            _task(clock), delegated_scopes=ScopeSet.allow_all()))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(
                commands=(
                    CreateTask(
                        key="child",
                        handler="echo",
                        delegated_scopes=("read", "delete"),  # 'delete' ungranted
                    ),
                )
            ),
        )
        tasks = await task_store.list_tasks("j1")
        child = [t for t in tasks if t.id != "t1"][0]
        # Intersection only: 'read' kept, 'delete' dropped -- no widening.
        assert child.delegated_scopes == ScopeSet.of("read")
        # Actor chain appended the creating handler as a new Actor.
        assert child.actor_chain is not None
        assert child.actor_chain.actors[-1].kind == "TaskHandler"
        assert child.actor_chain.actors[-1].id == "echo"
        # The originating user actor is still present (chain appended, not replaced).
        assert child.actor_chain.actors[0].kind == "user"

    _run(run())


def test_max_depth_exceeded_fails_commit(task_store) -> None:
    """Plan 5.1.6: a child beyond max_depth must fail the whole commit (raise
    TaskBudgetExceededError) -- it is never silently dropped. The over-depth
    child is not created AND the parent is not marked successful."""
    from linktools.ai.task.protocols import CreateTask, TaskSuccess
    from linktools.ai.task.store import TaskBudgetExceededError

    clock = task_store._clock

    async def run() -> None:
        job = JobRecord(
            id="j1",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(max_depth=0),  # only the root (depth 0) allowed
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=clock.now(),
            started_at=None,
            finished_at=None,
        )
        await task_store.create_job(job, _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        with pytest.raises(TaskBudgetExceededError):
            await task_store.commit_success(
                claimed.claim,
                TaskSuccess(commands=(CreateTask(key="child", handler="echo"),)),
            )
        tasks = await task_store.list_tasks("j1")
        # Only the root exists; the over-depth child was not created.
        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        # The parent was NOT marked successful; it stays CLAIMED.
        parent = await task_store.get_task("t1")
        assert parent.status == TaskStatus.CLAIMED

    _run(run())


def test_job_runtime_budget_does_not_leave_ready_task(task_store) -> None:
    """Plan 5.1.7: once a job exceeds its runtime cap, a claim finalizes the
    job (tasks CANCELLED, job FAILED) instead of leaving a READY zombie."""
    clock = task_store._clock

    async def run() -> None:
        started = clock.now()
        job = JobRecord(
            id="j1",
            status=JobStatus.RUNNING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(max_runtime_seconds=1.0),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=started,
            started_at=started,
            finished_at=None,
        )
        await task_store.create_job(job, _task(clock))
        clock.advance(2.0)  # past the 1s runtime cap
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is None
        job_now = await task_store.get_job("j1")
        assert job_now.status == JobStatus.FAILED
        tasks = await task_store.list_tasks("j1")
        assert all(
            t.status not in (TaskStatus.READY, TaskStatus.RETRY_WAIT) for t in tasks
        )

    _run(run())


def test_job_attempt_budget_does_not_leave_ready_task(task_store) -> None:
    """Plan 5.1.7: once a job's aggregate attempt cap is met, a later claim
    finalizes the job (READY child CANCELLED, job FAILED), no zombie."""
    from linktools.ai.task.protocols import CreateTask

    clock = task_store._clock
    started = clock.now()

    async def run() -> None:
        job = JobRecord(
            id="j1",
            status=JobStatus.RUNNING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(max_attempts=1),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=started,
            started_at=started,
            finished_at=None,
        )
        await task_store.create_job(job, _task(clock))
        first = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert first is not None
        await task_store.commit_success(
            first.claim,
            TaskSuccess(commands=(CreateTask(key="c1", handler="h"),)),
        )
        second = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert second is None
        job_now = await task_store.get_job("j1")
        assert job_now.status == JobStatus.FAILED
        tasks = await task_store.list_tasks("j1")
        assert all(
            t.status not in (TaskStatus.READY, TaskStatus.RETRY_WAIT) for t in tasks
        )

    _run(run())


def test_bind_run_fences_on_worker_id(task_store) -> None:
    """bind_run applies the full 4-field guard: a wrong worker_id is rejected
    even if status/attempt/fencing match, so a stale worker cannot bind a run
    to a claim another worker now owns."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w1", now=clock.now(), lease_seconds=30
        )
        # Wrong worker_id -> rejected.
        with pytest.raises(TaskClaimLostError):
            await task_store.bind_run(
                task_id="t1",
                attempt_id=claimed.claim.attempt_id,
                fencing_token=claimed.claim.fencing_token,
                worker_id="impostor",
                run_id="run-1",
            )
        # Correct worker_id -> bound.
        attempt = await task_store.bind_run(
            task_id="t1",
            attempt_id=claimed.claim.attempt_id,
            fencing_token=claimed.claim.fencing_token,
            worker_id="w1",
            run_id="run-1",
        )
        assert attempt.run_id == "run-1"

    _run(run())


def test_permanent_failure_does_not_retry(task_store) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        failed = await task_store.commit_failure(
            claimed.claim,
            TaskFailure(
                kind=TaskFailureKind.PERMANENT, error_type="BadInput", message="nope"
            ),
        )
        assert failed.status == TaskStatus.FAILED
        # A failed task is not claimable.
        assert (
            await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
            is None
        )

    _run(run())


def test_create_task_rejects_duplicate_key_within_job(task_store) -> None:
    """Per-task key uniqueness (section 13.3): the SQL store mirrors the file
    store's explicit pre-check so a duplicate key raises a clean ValueError
    rather than an IntegrityError that would strand the parent in CLAIMED."""
    from linktools.ai.task.protocols import CreateTask, TaskSuccess

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        with pytest.raises(ValueError, match="duplicate task key"):
            await task_store.commit_success(
                claimed.claim,
                TaskSuccess(
                    commands=(
                        CreateTask(key="dup", handler="h"),
                        CreateTask(key="dup", handler="h"),
                    )
                ),
            )

    _run(run())
