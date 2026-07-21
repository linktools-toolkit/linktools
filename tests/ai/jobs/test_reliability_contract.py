#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-backend reliability contract (fix-plan sections 4.5, 9.1).

The reliable-task state-machine and command-scope invariants must hold for BOTH
backends. This module parametrizes the same behavioral tests over the file and
sqlalchemy TaskStores so any divergence is a bug. It covers the plan's Phase 1
fixes: budget judgment (5.1), terminal-job claim gating (5.2) and cross-job
command scope (5.3).

A fake clock drives lease/retry timing so no real sleep is needed.
"""

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.filesystem.job import FilesystemJobStore
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.job import SqlAlchemyJobStore
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
    TaskStatus,
)
from linktools.ai.jobs.protocols import (
    CancelJob,
    CancelTask,
    CompleteJob,
    CreateTask,
    TaskSuccess,
)
from linktools.ai.jobs.store import (
    InvalidTaskCommandError,
    RunnableBindingError,
    TaskBudgetExceededError,
    TaskClaimLostError,
)


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self._t = start

    def now(self) -> datetime:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t = self._t + timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _make_store(backend: str, tmp_path):
    clock = FakeClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    if backend == "file":
        return FilesystemJobStore(tmp_path, clock=clock)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/rel.db")
    asyncio.run(_create_tables(engine))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyJobStore(session_factory=factory, clock=clock)


@pytest.fixture(params=["file", "sqlite"])
def task_store(request, tmp_path):
    return _make_store(request.param, tmp_path)


def _run(coro):
    return asyncio.run(coro)


def _job(
    clock,
    *,
    job_id="j1",
    root_task_id="t1",
    budget=None,
) -> JobRecord:
    return JobRecord(
        id=job_id,
        status=JobStatus.PENDING,
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        budget=budget or TaskBudget(),
        root_task_id=root_task_id,
        input_artifact_id=None,
        output_artifact_id=None,
        version=1,
        created_at=clock.now(),
        started_at=None,
        finished_at=None,
    )


def _task(
    clock,
    *,
    task_id="t1",
    job_id="j1",
    key="k",
    handler="runtime",
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        job_id=job_id,
        parent_task_id=None,
        key=key,
        handler=handler,
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


async def _force_job_status(store, job_id: str, status: JobStatus) -> None:
    """White-box: place a job in a terminal status while leaving its tasks as
    they are, so the claim whitelist (5.2.2) can be exercised against an
    inconsistent-but-possible leftover state. Defense-in-depth only -- the store
    API itself cannot reach this state once 5.2.3 holds."""
    if isinstance(store, FilesystemJobStore):
        job = await store.get_job(job_id)
        store._write(store._job_path(job_id), dataclasses.replace(job, status=status))
    else:
        from sqlalchemy import update as sa_update

        from linktools.ai.storage.sqlalchemy.models import TaskJobRow

        async with store._session_factory() as session:
            await session.execute(
                sa_update(TaskJobRow)
                .where(TaskJobRow.id == job_id)
                .values(status=status.value)
            )
            await session.commit()


async def _inject_ready_sibling(store, job_id: str, task_id: str, clock) -> None:
    """White-box: write a READY sibling task into an existing job so a later
    claim has a candidate. Used to drive budget finalization while another task
    in the job is still CLAIMED (a state the single-threaded store API cannot
    reach on its own, since creating a child requires committing its parent)."""
    from linktools.ai.jobs.models import AttemptStatus  # noqa: F401

    record = _task(clock, task_id=task_id, job_id=job_id, key=task_id)
    record = dataclasses.replace(record, status=TaskStatus.READY)
    if isinstance(store, FilesystemJobStore):
        store._write(store._task_path(job_id, task_id), record)
    else:
        import json

        from linktools.ai.storage.sqlalchemy.models import TaskRow
        from linktools.ai.storage.sqlalchemy.job import _store_dt, _task_envelope

        async with store._session_factory() as session:
            session.add(
                TaskRow(
                    id=task_id,
                    job_id=job_id,
                    parent_task_id=None,
                    key=task_id,
                    handler="runtime",
                    status=TaskStatus.READY.value,
                    input_artifact_id=None,
                    output_artifact_id=None,
                    attempt_count=0,
                    available_at=_store_dt(clock.now()),
                    lease_owner=None,
                    lease_expires_at=None,
                    fencing_token=0,
                    active_attempt_id=None,
                    timeout_seconds=None,
                    version=1,
                    created_at=_store_dt(clock.now()),
                    updated_at=_store_dt(clock.now()),
                    data_json=_task_envelope(record),
                )
            )
            await session.commit()


# ---------------------------------------------------------------- 5.1 budget --


def test_store_rejects_invalid_job_budget(task_store) -> None:
    """The store itself re-validates the job budget (not just the runtime), so a
    caller that bypasses JobRuntime still cannot persist a job whose budget
    would brick its root task (e.g. max_tasks=0)."""
    clock = task_store._clock

    async def run() -> None:
        job = _job(clock, budget=TaskBudget(max_tasks=0))
        with pytest.raises(ValueError):
            await task_store.create_job(job, _task(clock))

    _run(run())


def test_max_tasks_one_allows_root_task(task_store) -> None:
    """Plan 5.1.1 / 5.1.4: with max_tasks=1 the root task (which makes the count
    1) must still be claimable. The old Claim check (count >= max_tasks)
    permanently blocked the root of a max_tasks=1 job."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(
            _job(clock, budget=TaskBudget(max_tasks=1)), _task(clock)
        )
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is not None
        assert claimed.task.id == "t1"

    _run(run())


def test_task_at_budget_limit_is_still_claimable(task_store) -> None:
    """Plan 5.1: an existing task at the budget limit remains claimable; the cap
    only restricts CREATION of further tasks."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(
            _job(clock, budget=TaskBudget(max_tasks=2)), _task(clock)
        )
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        # Root succeeds and creates one child -> task count is now 2 == cap.
        await task_store.commit_success(
            root.claim,
            TaskSuccess(commands=(CreateTask(key="c1", handler="h"),)),
        )
        # The child exists at the cap; it must still be claimable.
        child = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert child is not None
        assert child.task.key == "c1"

    _run(run())


def test_child_creation_rejected_when_max_tasks_exhausted(task_store) -> None:
    """Plan 5.1.5: creating a child beyond max_tasks raises (checked at the
    count, not at claim), and no child is created."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(
            _job(clock, budget=TaskBudget(max_tasks=1)), _task(clock)
        )
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        with pytest.raises(TaskBudgetExceededError):
            await task_store.commit_success(
                root.claim,
                TaskSuccess(commands=(CreateTask(key="c1", handler="h"),)),
            )
        tasks = await task_store.list_tasks("j1")
        assert [t.key for t in tasks] == ["k"]

    _run(run())


def test_multiple_child_commands_are_budget_checked_atomically(task_store) -> None:
    """Plan 5.1.5: when a single success creates several children, the budget
    check is atomic -- either all are created or none. Requesting more children
    than the cap allows rejects the whole batch; zero partial children."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(
            _job(clock, budget=TaskBudget(max_tasks=3)), _task(clock)
        )
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        # Root (1) + 3 requested = 4 > max_tasks(3) -> reject all three.
        with pytest.raises(TaskBudgetExceededError):
            await task_store.commit_success(
                root.claim,
                TaskSuccess(
                    commands=(
                        CreateTask(key="c1", handler="h"),
                        CreateTask(key="c2", handler="h"),
                        CreateTask(key="c3", handler="h"),
                    )
                ),
            )
        tasks = await task_store.list_tasks("j1")
        # Only the root exists -- no partial creation.
        assert [t.key for t in tasks] == ["k"]

    _run(run())


# --------------------------------------------------- 5.2 terminal-job gating --


def test_complete_job_rejected_with_live_sibling(task_store) -> None:
    """Plan 5.2.3: CompleteJob is rejected (InvalidTaskCommandError) while a
    sibling task is still live, so a task cannot finish the job out from under
    running work."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        # Root succeeds, creating two READY children (siblings of each other).
        await task_store.commit_success(
            root.claim,
            TaskSuccess(
                commands=(
                    CreateTask(key="c1", handler="h"),
                    CreateTask(key="c2", handler="h"),
                )
            ),
        )
        c1 = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert c1.task.key == "c1"
        # c2 is still READY (a live sibling) -> CompleteJob on c1 is rejected.
        with pytest.raises(InvalidTaskCommandError):
            await task_store.commit_success(
                c1.claim, TaskSuccess(commands=(CompleteJob(),))
            )
        job = await task_store.get_job("j1")
        assert job.status not in (JobStatus.SUCCEEDED,)

    _run(run())


def test_complete_job_allowed_when_current_is_only_live_task(task_store) -> None:
    """Plan 5.2.3: CompleteJob succeeds when the committing task is the only
    non-terminal task; the job then lands SUCCEEDED."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        # Root succeeds with one child; the child is then the only live task.
        await task_store.commit_success(
            root.claim,
            TaskSuccess(commands=(CreateTask(key="c1", handler="h"),)),
        )
        child = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        await task_store.commit_success(
            child.claim, TaskSuccess(commands=(CompleteJob(),))
        )
        job = await task_store.get_job("j1")
        assert job.status == JobStatus.SUCCEEDED

    _run(run())


@pytest.mark.parametrize(
    "terminal", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]
)
def test_terminal_job_cannot_claim_task(task_store, terminal) -> None:
    """Plan 5.2.2: a task whose job is terminal (SUCCEEDED/FAILED) is never
    claimed, even if a leftover READY task somehow exists. Defense-in-depth for
    the claim whitelist -- the API cannot reach this state once 5.2.3 holds."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        # Force the job terminal while its READY root still exists.
        await _force_job_status(task_store, "j1", terminal)
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is None

    _run(run())


# ----------------------------------------------------- 5.3 command scope -----


def test_cancel_job_command_has_no_job_id_field() -> None:
    """Plan 5.3.2: a handler CancelJob cannot even express another job -- the
    command carries only a reason, never a job_id."""
    fields = {f.name for f in dataclasses.fields(CancelJob)}
    assert "job_id" not in fields
    assert "reason" in fields


def test_handler_cancel_job_only_targets_current_job(task_store) -> None:
    """Plan 5.3.2: a CancelJob command affects only the job whose task produced
    it; a concurrent job is untouched and remains claimable."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock, job_id="j1", root_task_id="t1"), _task(clock, task_id="t1", job_id="j1"))
        await task_store.create_job(_job(clock, job_id="j2", root_task_id="t2"), _task(clock, task_id="t2", job_id="j2", key="k2"))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert root.task.id == "t1"
        await task_store.commit_success(
            root.claim, TaskSuccess(commands=(CancelJob(),))
        )
        # j1 is cancelled; j2 is untouched and its root is still claimable.
        j1 = await task_store.get_job("j1")
        assert j1.status in (JobStatus.CANCELLING, JobStatus.CANCELLED)
        other = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert other is not None
        assert other.task.id == "t2"
        j2 = await task_store.get_job("j2")
        assert j2.status not in (JobStatus.CANCELLED, JobStatus.CANCELLING)

    _run(run())


def test_cancel_task_same_job_records_legal_transition_edges(task_store) -> None:
    """A same-job CancelTask records only legal state-machine edges
    (pre -> CANCELLING -> CANCELLED), never a forbidden direct pre -> CANCELLED.
    Guards the audit-log invariant across both backends."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        await task_store.commit_success(
            root.claim,
            TaskSuccess(
                commands=(
                    CreateTask(key="c1", handler="h"),
                    CreateTask(key="c2", handler="h"),
                )
            ),
        )
        tasks = await task_store.list_tasks("j1")
        c2_id = next(t.id for t in tasks if t.key == "c2")
        c1 = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert c1.task.key == "c1"
        await task_store.commit_success(
            c1.claim, TaskSuccess(commands=(CancelTask(task_id=c2_id),))
        )
        transitions = await task_store.list_transitions("j1")
        edges = [
            (t.from_status, t.to_status)
            for t in transitions
            if t.task_id == c2_id and t.from_status is not None
        ]
        # c2 was READY; the cancel must walk the legal two-step path.
        assert ("ready", "cancelling") in edges
        assert ("cancelling", "cancelled") in edges
        assert ("ready", "cancelled") not in edges

    _run(run())


def test_handler_cannot_cancel_foreign_task(task_store) -> None:
    """Plan 5.3.3: CancelTask is scoped to the current job. Naming a task that
    belongs to another job raises InvalidTaskCommandError and never touches the
    foreign task."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock, job_id="j1", root_task_id="t1"), _task(clock, task_id="t1", job_id="j1"))
        await task_store.create_job(_job(clock, job_id="j2", root_task_id="t2"), _task(clock, task_id="t2", job_id="j2", key="k2"))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert root.task.id == "t1"
        # Try to cancel j2's task from within j1's commit -> rejected.
        with pytest.raises(InvalidTaskCommandError):
            await task_store.commit_success(
                root.claim,
                TaskSuccess(commands=(CancelTask(task_id="t2"),)),
            )
        # The foreign task is untouched (still claimable).
        other = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert other is not None
        assert other.task.id == "t2"
        assert other.task.status not in (
            TaskStatus.CANCELLING,
            TaskStatus.CANCELLED,
        )

    _run(run())


def test_cross_tenant_cancel_task_is_rejected(task_store) -> None:
    """Plan 5.3.4 / 5.3.5: tenant isolation follows from job scoping. A task in
    a job owned by tenant B cannot be canceled from within tenant A's job."""
    clock = task_store._clock

    async def run() -> None:
        job_a = JobRecord(
            id="ja",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="tenant-a", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(),
            root_task_id="ta",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=clock.now(),
            started_at=None,
            finished_at=None,
        )
        job_b = dataclasses.replace(
            job_a,
            id="jb",
            principal=TaskPrincipal(tenant_id="tenant-b", user_id="bob"),
            root_task_id="tb",
        )
        await task_store.create_job(job_a, _task(clock, task_id="ta", job_id="ja"))
        await task_store.create_job(job_b, _task(clock, task_id="tb", job_id="jb", key="kb"))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert root.task.id == "ta"
        with pytest.raises(InvalidTaskCommandError):
            await task_store.commit_success(
                root.claim,
                TaskSuccess(commands=(CancelTask(task_id="tb"),)),
            )

    _run(run())


def test_two_workers_claim_same_task_only_one_wins(task_store) -> None:
    """Plan §13 (concurrency): two workers claiming concurrently against one
    READY task cannot both win -- exactly one ClaimedTask comes back."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        results = await asyncio.gather(
            task_store.claim(worker_id="w1", now=clock.now(), lease_seconds=30),
            task_store.claim(worker_id="w2", now=clock.now(), lease_seconds=30),
        )
        claimed = [r for r in results if r is not None]
        assert len(claimed) == 1

    _run(run())


def test_complete_job_combined_with_create_task_is_rejected_atomically(
    task_store,
) -> None:
    """Plan 5.1.5 (all-or-none) + 5.2.3: a contradictory batch that both creates a
    child and completes the job is rejected BEFORE any child is written, so no
    partial child survives the failed commit."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        with pytest.raises(InvalidTaskCommandError):
            await task_store.commit_success(
                root.claim,
                TaskSuccess(
                    commands=(
                        CreateTask(key="c1", handler="h"),
                        CompleteJob(),
                    )
                ),
            )
        tasks = await task_store.list_tasks("j1")
        # No partial child created -- only the root remains.
        assert [t.key for t in tasks] == ["k"]

    _run(run())


def test_budget_exhausted_does_not_orphan_running_attempt(task_store) -> None:
    """Plan §13 ('no Attempt RUNNING permanently残留'): when aggregate-budget
    finalization cancels a CLAIMED task mid-flight, its active RUNNING attempt
    is closed (CANCELLED), not orphaned. A READY sibling is injected so a claim
    has a candidate to trigger finalization while the root is still CLAIMED."""
    from linktools.ai.jobs.models import AttemptStatus

    clock = task_store._clock

    async def run() -> None:
        job = JobRecord(
            id="j1",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(max_runtime_seconds=1.0),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=clock.now(),
            started_at=None,
            finished_at=None,
        )
        await task_store.create_job(job, _task(clock))
        # Claim the root -> it is CLAIMED with a RUNNING attempt; job RUNNING.
        root = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert root is not None
        root_attempt = root.claim.attempt_id
        # Inject a READY sibling so a later claim has a finalization candidate.
        await _inject_ready_sibling(task_store, "j1", "t2", clock)
        clock.advance(2.0)  # past the 1s runtime cap
        claimed = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        # The candidate was finalized (not claimed), and the root's attempt was
        # closed -- no RUNNING attempt left behind in the job.
        assert claimed is None
        attempts = await task_store.list_attempts("t1")
        root_att = [a for a in attempts if a.id == root_attempt]
        assert root_att and root_att[0].status == AttemptStatus.CANCELLED

    _run(run())


# ----------------------------------------------------- Phase 5 concurrency --


def test_bind_runnable_pins_and_rejects_drift(task_store) -> None:
    """The first runnable resolution is pinned on the task; an idempotent re-bind
    with the same resolution succeeds, but a drift (a mapping change returned a
    different fingerprint) is rejected -- so a retry can never silently re-run a
    different agent. Holds on both backends."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        c = claimed.claim
        bound = await task_store.bind_runnable(
            task_id=c.task_id, attempt_id=c.attempt_id, worker_id=c.worker_id,
            fencing_token=c.fencing_token, runnable_id="a", revision="r1", fingerprint="fp1",
        )
        assert bound.resolved_runnable_id == "a"
        assert bound.resolved_runnable_fingerprint == "fp1"
        # Idempotent re-bind (a retry that re-resolved the same spec).
        again = await task_store.bind_runnable(
            task_id=c.task_id, attempt_id=c.attempt_id, worker_id=c.worker_id,
            fencing_token=c.fencing_token, runnable_id="a", revision="r1", fingerprint="fp1",
        )
        assert again.resolved_runnable_fingerprint == "fp1"
        # A drift is rejected.
        with pytest.raises(RunnableBindingError):
            await task_store.bind_runnable(
                task_id=c.task_id, attempt_id=c.attempt_id, worker_id=c.worker_id,
                fencing_token=c.fencing_token, runnable_id="a", revision="r1", fingerprint="fp2",
            )

    _run(run())


def test_bind_runnable_fences_on_stale_worker(task_store) -> None:
    """bind_runnable is fenced like every post-claim write: a wrong worker or a
    stale fencing token is rejected with TaskClaimLostError (both backends)."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        c = claimed.claim
        with pytest.raises(TaskClaimLostError):
            await task_store.bind_runnable(
                task_id=c.task_id, attempt_id=c.attempt_id, worker_id="impostor",
                fencing_token=c.fencing_token, runnable_id="a", revision=None, fingerprint="fp",
            )
        with pytest.raises(TaskClaimLostError):
            await task_store.bind_runnable(
                task_id=c.task_id, attempt_id=c.attempt_id, worker_id=c.worker_id,
                fencing_token=c.fencing_token + 1, runnable_id="a", revision=None, fingerprint="fp",
            )

    _run(run())


def test_stale_worker_commit_after_reclaim_is_rejected(task_store) -> None:
    """Plan §9.2 fencing invariant: worker A's lease expires, the task is
    recovered and reclaimed by worker B; A's later commit is rejected with
    TaskClaimLostError -- a stale worker can never overwrite the new owner's
    result. Holds on both backends."""
    from datetime import timedelta

    from linktools.ai.jobs.protocols import TaskSuccess

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        a = await task_store.claim(worker_id="A", now=clock.now(), lease_seconds=30)
        assert a is not None
        # Lease expires; recovery resets the task, B reclaims it.
        clock.advance(60.0)
        await task_store.recover_expired(now=clock.now(), limit=10)
        b = await task_store.claim(worker_id="B", now=clock.now(), lease_seconds=30)
        assert b is not None
        assert b.claim.worker_id == "B"
        # A's stale commit must be rejected (fencing).
        with pytest.raises(TaskClaimLostError):
            await task_store.commit_success(a.claim, TaskSuccess())
        # B still holds the task and can commit.
        await task_store.commit_success(b.claim, TaskSuccess())
        done = await task_store.get_task("t1")
        assert done.status == TaskStatus.SUCCEEDED

    _run(run())
