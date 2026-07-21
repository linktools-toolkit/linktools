#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemJobStore contract -- the reliable-task invariants over the file backend
(plan section 28 phase-3 acceptance, section 30.4 key invariants).

A fake clock drives lease/retry timing so no real sleep is needed (plan 30.5).
These tests are the contract every JobStore backend must satisfy; the
SQLAlchemy backend (later phase) is exercised by parameterizing ``task_store``.
"""

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from linktools.ai.storage.filesystem.job import FilesystemJobStore
from linktools.ai.jobs.models import (
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
from linktools.ai.jobs.protocols import TaskFailure, TaskSuccess
from linktools.ai.jobs.store import TaskClaimLostError


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


def _task(clock, *, task_id="t1", job_id="j1", handler="runtime") -> TaskRecord:
    return TaskRecord(
        id=task_id,
        job_id=job_id,
        parent_task_id=None,
        key="k",
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


@pytest.fixture
def task_store(tmp_path: Path) -> FilesystemJobStore:
    clock = FakeClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    return FilesystemJobStore(tmp_path, clock=clock)


def _run(coro):
    return asyncio.run(coro)


def test_create_claim_complete_completes_job(task_store: FilesystemJobStore) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is not None
        assert claimed.task.status == TaskStatus.CLAIMED
        assert claimed.task.fencing_token == 1
        assert claimed.task.attempt_count == 1
        done = await task_store.commit_success(claimed.claim, TaskSuccess())
        assert done.status == TaskStatus.SUCCEEDED
        # Root task completing succeeds the job.
        assert (await task_store.get_job("j1")).status == JobStatus.SUCCEEDED

    _run(run())


def test_stale_fencing_token_is_rejected_after_reclaim(
    task_store: FilesystemJobStore,
) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        first = await task_store.claim(
            worker_id="w1", now=clock.now(), lease_seconds=30
        )
        clock.advance(60)  # lease expires
        await task_store.recover_expired(now=clock.now(), limit=10)
        reclaimed = await task_store.claim(
            worker_id="w2", now=clock.now(), lease_seconds=30
        )
        assert reclaimed.task.fencing_token == first.task.fencing_token + 1
        # w1's stale commit is rejected; it must not overwrite w2's result.
        with pytest.raises(TaskClaimLostError):
            await task_store.commit_success(first.claim, TaskSuccess())
        await task_store.commit_success(reclaimed.claim, TaskSuccess())

    _run(run())


def test_completed_task_is_not_reclaimed(task_store: FilesystemJobStore) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(claimed.claim, TaskSuccess())
        # No task left to claim.
        assert (
            await task_store.claim(worker_id="other", now=clock.now(), lease_seconds=30)
            is None
        )

    _run(run())


def test_json_commit_journal_recovers_without_reexecuting_commands(
    task_store: FilesystemJobStore, monkeypatch
) -> None:
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess
    clock = task_store._clock
    original = task_store._journal.mark_step
    failed = {"once": False}

    def interrupt(path, step):
        if step == "COMMANDS_APPLIED" and not failed["once"]:
            failed["once"] = True
            raise OSError("simulated crash")
        original(path, step)

    async def run():
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        monkeypatch.setattr(task_store._journal, "mark_step", interrupt)
        with pytest.raises(OSError, match="simulated crash"):
            await task_store.commit_success(claimed.claim, TaskSuccess(commands=(
                CreateTask(key="child", handler="echo"),)))
        monkeypatch.setattr(task_store._journal, "mark_step", original)
        await task_store.recover_incomplete_commits()
        tasks = await task_store.list_tasks("j1")
        assert len([task for task in tasks if task.key == "child"]) == 1
        assert (await task_store.get_task("t1")).status is TaskStatus.SUCCEEDED
        assert not list(task_store._journal.root.glob("*.json"))

    _run(run())


def test_transient_failure_retries_then_succeeds(task_store: FilesystemJobStore) -> None:
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
        clock.advance(60)  # past the retry delay
        retry = await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        assert retry is not None
        assert (
            retry.task.attempt_count == 2
        )  # a fresh attempt, not a mutation of the old one
        await task_store.commit_success(retry.claim, TaskSuccess())
        assert (await task_store.get_task("t1")).status == TaskStatus.SUCCEEDED

    _run(run())


def test_permanent_failure_does_not_retry(task_store: FilesystemJobStore) -> None:
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


def test_cancel_marks_job_and_non_active_tasks_cancelled(
    task_store: FilesystemJobStore,
) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        job = await task_store.request_cancel("j1", reason="user")
        assert job.status == JobStatus.CANCELLED
        assert (await task_store.get_task("t1")).status == TaskStatus.CANCELLED

    _run(run())


def test_cancel_moves_in_flight_claimed_task_to_cancelling(
    task_store: FilesystemJobStore,
) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is not None
        # An in-flight (CLAIMED) task moves to CANCELLING so the owning worker
        # observes it on its next poll and lands CANCELLED when it commits. The
        # job stays CANCELLING (not outright CANCELLED) while a task is still
        # in-flight.
        job = await task_store.request_cancel("j1", reason="user")
        assert job.status == JobStatus.CANCELLING
        assert (await task_store.get_task("t1")).status == TaskStatus.CANCELLING
        # A CANCELLING task whose lease expired is finalized CANCELLED by
        # recovery, which then lets the job converge to CANCELLED.
        clock.advance(60)
        await task_store.recover_expired(now=clock.now(), limit=10)
        assert (await task_store.get_task("t1")).status == TaskStatus.CANCELLED
        assert (await task_store.get_job("j1")).status == JobStatus.CANCELLED

    _run(run())


def test_two_workers_concurrent_claim_same_task_only_one_wins(
    task_store: FilesystemJobStore,
) -> None:
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        # Two workers race to claim the SAME ready task concurrently. The
        # backend's lock/CAS must ensure exactly one wins.
        results = await asyncio.gather(
            task_store.claim(
                worker_id="w1", now=clock.now(), lease_seconds=30
            ),
            task_store.claim(
                worker_id="w2", now=clock.now(), lease_seconds=30
            ),
        )
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0].claim.worker_id in {"w1", "w2"}

    _run(run())


def test_concurrent_cancel_and_complete_stays_consistent(
    task_store: FilesystemJobStore,
) -> None:
    """§30.2: a request_cancel racing with commit_success must leave a
    consistent state (no crash, no illegal transition) -- fencing + the
    CANCELLING commit-guard decide the outcome deterministically."""
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        assert claimed is not None

        async def _commit() -> None:
            try:
                await task_store.commit_success(claimed.claim, TaskSuccess())
            except TaskClaimLostError:
                pass  # cancel won the race -- acceptable

        await asyncio.gather(_commit(), task_store.request_cancel("j1"))
        # Whatever won, the task is terminal and the job reached a terminal /
        # CANCELLING state with no illegal transition raised.
        task = await task_store.get_task("t1")
        assert task.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.CANCELLED,
            TaskStatus.CANCELLING,
        )

    _run(run())


def test_recover_expired_resets_claimed_and_supersedes_attempt(
    task_store: FilesystemJobStore,
) -> None:
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
        # The dead attempt is SUPERSEDED.
        attempts = await task_store.list_attempts("t1")
        assert attempts[0].status.value == "superseded"
        # The task is reclaimable by a new worker with a bumped fencing token.
        reclaimed = await task_store.claim(
            worker_id="w2", now=clock.now(), lease_seconds=30
        )
        assert reclaimed is not None
        assert reclaimed.task.fencing_token == claimed.task.fencing_token + 1

    _run(run())


def test_persistence_survives_reopen(tmp_path: Path) -> None:
    # A new store over the same root sees the previously written job (recovery).
    clock = FakeClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    task_store = FilesystemJobStore(tmp_path, clock=clock)

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))

    _run(run())

    reopened = FilesystemJobStore(
        tmp_path, clock=FakeClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    )

    async def read() -> None:
        job = await reopened.get_job("j1")
        assert job is not None and job.id == "j1"
        task = await reopened.get_task("t1")
        assert task is not None and task.status == TaskStatus.READY

    _run(read())


def test_recovery_tolerates_missing_attempt_file(task_store: FilesystemJobStore) -> None:
    # Crash window: claim wrote the task but the attempt file is gone. Recovery
    # must not abort -- it resets the task and moves on.
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        # Simulate the crash: remove the attempt file, leaving a dangling
        # active_attempt_id on the CLAIMED task.
        job_id = task_store._find_task_owner_job("t1")[0]
        task_store._attempt_path(job_id, claimed.claim.attempt_id).unlink()
        clock.advance(60)
        recovered = await task_store.recover_expired(now=clock.now(), limit=10)
        assert len(recovered) == 1
        assert recovered[0].status == TaskStatus.READY

    _run(run())


def test_recovery_converges_job_left_running_after_commit_crash(
    task_store: FilesystemJobStore,
) -> None:
    # Crash window: the root task was committed SUCCEEDED but the job-completion
    # write did not land, leaving the job RUNNING. Recovery re-converges it.
    import dataclasses

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(claimed.claim, TaskSuccess())
        assert (await task_store.get_job("j1")).status == JobStatus.SUCCEEDED
        # Rewind the job to RUNNING to simulate the post-commit crash window.
        job_id = task_store._find_task_owner_job("t1")[0]
        job = await task_store.get_job("j1")
        task_store._write(
            task_store._job_path(job_id),
            dataclasses.replace(job, status=JobStatus.RUNNING, finished_at=None),
        )
        await task_store.recover_expired(now=clock.now(), limit=10)
        assert (await task_store.get_job("j1")).status == JobStatus.SUCCEEDED

    _run(run())


def test_recovery_fails_task_when_attempts_exhausted(task_store: FilesystemJobStore) -> None:
    # A task whose attempt_count has reached max_attempts is FAILED on lease
    # recovery, not reset to READY (section 21.3 / 20.1 -- no infinite retry).
    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))  # max_attempts=2
        await task_store.claim(worker_id="w", now=clock.now(), lease_seconds=30)
        clock.advance(60)
        first = await task_store.recover_expired(now=clock.now(), limit=10)
        assert first[0].status == TaskStatus.READY  # attempt 1 < 2 -> retryable
        await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )  # attempt 2
        clock.advance(60)
        second = await task_store.recover_expired(now=clock.now(), limit=10)
        assert second[0].status == TaskStatus.FAILED  # attempt 2 >= 2 -> exhausted

    _run(run())


def test_create_task_command_creates_child(task_store: FilesystemJobStore) -> None:
    clock = task_store._clock

    async def run() -> None:
        from linktools.ai.jobs.protocols import CreateTask, TaskSuccess

        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(commands=(CreateTask(key="collect", handler="evidence"),)),
        )
        tasks = await task_store.list_tasks("j1")
        assert len(tasks) == 2  # root + child
        child = [t for t in tasks if t.id != "t1"][0]
        assert child.status == TaskStatus.READY
        assert child.parent_task_id == "t1"
        assert child.handler == "evidence"

    _run(run())


def test_wait_signal_command_transitions_to_waiting(task_store: FilesystemJobStore) -> None:
    from linktools.ai.jobs.protocols import TaskSuccess, WaitSignal

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


def test_signal_wakes_matching_waiting_task(task_store: FilesystemJobStore) -> None:
    from linktools.ai.jobs.models import TaskSignalRecord
    from linktools.ai.jobs.protocols import TaskSuccess, WaitSignal

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
        # Submit a matching signal → task wakes to READY.
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


def test_dependency_resolution_promotes_pending_to_ready(
    task_store: FilesystemJobStore,
) -> None:
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess

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
        # Root (t1) is SUCCEEDED → dependency satisfied → child should be READY.
        assert child.status == TaskStatus.READY

    _run(run())


def test_missing_dependency_does_not_crash(task_store: FilesystemJobStore) -> None:
    """C1 regression: _resolve_dependencies must tolerate a missing dep ref."""
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess

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


def test_commit_success_rejects_too_many_commands(
    task_store: FilesystemJobStore,
) -> None:
    """Wired command-count limit (section 17.5): an outcome carrying more than
    MAX_COMMANDS commands is rejected before any write."""
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess
    from linktools.ai.jobs.validation import MAX_COMMANDS

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


def test_commit_success_rejects_oversized_output_artifact(
    task_store: FilesystemJobStore,
) -> None:
    """Wired output-payload limit (section 17.5): an output artifact larger than
    the ceiling is rejected at commit."""
    from linktools.ai.artifact.models import ArtifactRef
    from linktools.ai.jobs.protocols import TaskSuccess
    from linktools.ai.jobs.validation import MAX_OUTPUT_PAYLOAD_BYTES

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


def test_submit_signal_rejects_oversized_metadata(
    task_store: FilesystemJobStore,
) -> None:
    """Wired signal-metadata limit (section 17.5): inline signal JSON above the
    metadata ceiling is rejected before the signal is persisted."""
    from linktools.ai.jobs.models import TaskSignalRecord

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


def test_child_task_delegated_scopes_narrow_and_actor_appends(
    task_store: FilesystemJobStore,
) -> None:
    """A child task's delegated scopes are the intersection of the parent's
    scopes and the command's requested scopes (never a union -- an ungranted
    scope is dropped, not added), and its actor chain appends the handler that
    created it."""
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess

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
        await task_store.create_job(
            job, dataclasses.replace(_task(clock), delegated_scopes=ScopeSet.allow_all())
        )
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


def test_max_depth_exceeded_fails_commit(task_store: FilesystemJobStore) -> None:
    """Plan 5.1.6: a child beyond max_depth must fail the whole commit (raise
    TaskBudgetExceededError) -- it is never silently dropped. The over-depth
    child is not created AND the parent is not marked successful."""
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess
    from linktools.ai.jobs.store import TaskBudgetExceededError

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
        # The parent was NOT marked successful (commit failed before the terminal
        # write); it stays CLAIMED for recovery / retry.
        parent = await task_store.get_task("t1")
        assert parent.status == TaskStatus.CLAIMED

    _run(run())


def test_job_runtime_budget_does_not_leave_ready_task(
    task_store: FilesystemJobStore,
) -> None:
    """Plan 5.1.7: once a job exceeds its runtime cap, a claim does not skip the
    READY candidate and leave it as a zombie -- it finalizes the job (tasks
    CANCELLED, job FAILED), so no task is permanently unclaimable-but-READY."""
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
        # Nothing claimable: the candidate was finalized, not deferred.
        assert claimed is None
        job_now = await task_store.get_job("j1")
        assert job_now.status == JobStatus.FAILED
        tasks = await task_store.list_tasks("j1")
        # No task left in a ready/retry-wait zombie state.
        assert all(
            t.status not in (TaskStatus.READY, TaskStatus.RETRY_WAIT) for t in tasks
        )

    _run(run())


def test_job_attempt_budget_does_not_leave_ready_task(
    task_store: FilesystemJobStore,
) -> None:
    """Plan 5.1.7: once a job's aggregate attempt cap is met, a later claim
    finalizes the job (the still-READY child is CANCELLED, job FAILED) rather
    than leaving it permanently claimable-but-unclaimable."""
    from linktools.ai.jobs.protocols import CreateTask

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
        assert first is not None  # job's attempt total is now 1 == cap
        # The first task succeeds and creates a READY child.
        await task_store.commit_success(
            first.claim,
            TaskSuccess(commands=(CreateTask(key="c1", handler="h"),)),
        )
        # The child is READY, but the job's aggregate-attempt cap is met.
        second = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        # Finalized, not deferred.
        assert second is None
        job_now = await task_store.get_job("j1")
        assert job_now.status == JobStatus.FAILED
        tasks = await task_store.list_tasks("j1")
        assert all(
            t.status not in (TaskStatus.READY, TaskStatus.RETRY_WAIT) for t in tasks
        )

    _run(run())


def test_bind_run_fences_on_worker_id(task_store: FilesystemJobStore) -> None:
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


def test_concurrent_signals_for_same_waiting_task_wake_it_once(
    task_store: FilesystemJobStore,
) -> None:
    """§30.2: two signals for the same WAITING task submitted concurrently must
    wake it exactly once -- the second finds it no longer WAITING."""
    from linktools.ai.jobs.models import TaskSignalRecord
    from linktools.ai.jobs.protocols import WaitSignal

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(commands=(WaitSignal(name="review", correlation_key="c"),)),
        )
        assert (await task_store.get_task("t1")).status == TaskStatus.WAITING
        s1 = TaskSignalRecord(
            id="s1", job_id="j1", name="review", correlation_key="c",
            payload_artifact_id=None, created_at=clock.now(), consumed_by_task_id=None,
        )
        s2 = TaskSignalRecord(
            id="s2", job_id="j1", name="review", correlation_key="c",
            payload_artifact_id=None, created_at=clock.now(), consumed_by_task_id=None,
        )
        await asyncio.gather(
            task_store.submit_signal(s1), task_store.submit_signal(s2)
        )
        task = await task_store.get_task("t1")
        assert task.status == TaskStatus.READY  # woken exactly once

    _run(run())


def test_commit_success_write_fault_leaves_parent_claimed_not_half_committed(
    task_store: FilesystemJobStore,
) -> None:
    """§12.5 + §30.3: a write fault mid commit_success (here, while creating the
    child task) must NOT leave the parent SUCCEEDED with its child missing. The
    parent flip is the LAST write, so the task stays CLAIMED and is recoverable."""
    from linktools.ai.jobs.models import TaskRecord as _TR
    from linktools.ai.jobs.protocols import CreateTask

    clock = task_store._clock
    original_write = task_store._write

    def flaky_write(path, record):
        # Fail when writing a CHILD task record (parent_task_id set).
        if isinstance(record, _TR) and record.parent_task_id is not None:
            raise OSError("disk full mid child-create")
        return original_write(path, record)

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        task_store._write = flaky_write  # inject the fault
        with pytest.raises(OSError):
            await task_store.commit_success(
                claimed.claim,
                TaskSuccess(commands=(CreateTask(key="c1", handler="h"),)),
            )
        task_store._write = original_write
        # The parent was never flipped to SUCCEEDED (its write comes last), so
        # it is still CLAIMED and recoverable -- not half-committed.
        task = await task_store.get_task("t1")
        assert task.status == TaskStatus.CLAIMED

    _run(run())


def test_recover_expired_reconciles_unconsumed_signal_to_waiting_task(
    task_store: FilesystemJobStore,
) -> None:
    """Crash-window reconciliation: if submit_signal's signal file landed on disk
    but its matching WAITING task was never woken (process died between save and
    wake), recover_expired must re-match them so the task is not stuck WAITING."""
    from linktools.ai.jobs.models import TaskSignalRecord
    from linktools.ai.jobs.protocols import TaskSuccess, WaitSignal

    clock = task_store._clock

    async def run() -> None:
        await task_store.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w", now=clock.now(), lease_seconds=30
        )
        await task_store.commit_success(
            claimed.claim,
            TaskSuccess(
                commands=(WaitSignal(name="approval", correlation_key="case-9"),)
            ),
        )
        assert (await task_store.get_task("t1")).status == TaskStatus.WAITING
        # Simulate the crash window: write the signal file directly (bypassing
        # submit_signal's wake) so it is saved-but-unconsumed.
        orphan = TaskSignalRecord(
            id="sig-orphan",
            job_id="j1",
            name="approval",
            correlation_key="case-9",
            payload_artifact_id=None,
            created_at=clock.now(),
            consumed_by_task_id=None,
        )
        task_store._write(task_store._signal_path("j1", "sig-orphan"), orphan)
        # Recovery reconciles the unconsumed signal -> wakes the WAITING task.
        await task_store.recover_expired(now=clock.now(), limit=10)
        assert (await task_store.get_task("t1")).status == TaskStatus.READY
        # And the signal is now marked consumed by that task.
        consumed = task_store._read(task_store._signal_path("j1", "sig-orphan"))
        assert consumed["consumed_by_task_id"] == "t1"

    _run(run())


def test_create_task_rejects_duplicate_key_within_job(
    task_store: FilesystemJobStore,
) -> None:
    """Per-task key uniqueness within a job (section 13.3, the UNIQUE(job_id,key)
    invariant): a second child created with an already-used key is rejected
    rather than silently overwriting or stranding the parent."""
    from linktools.ai.jobs.protocols import CreateTask, TaskSuccess

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
        # The first child was created; the second raised. The parent flip is the
        # last write, so the parent is still CLAIMED (not half-committed).
        assert (await task_store.get_task("t1")).status == TaskStatus.CLAIMED

    _run(run())
