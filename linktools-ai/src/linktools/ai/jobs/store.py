#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JobStore contract.

The store is a *domain-semantic* interface, not generic CRUD. Every method is
one of the reliable-task operations -- claim, renew lease, bind a run, commit
success or failure, request cancel, submit a signal, recover expired leases --
and each performs its own fencing check and atomic state move. A caller never
hands the store a bare ``task_id`` to mutate; after a claim it carries the full
:class:`TaskClaim` (task + attempt + worker + fencing token), and the store
rejects any write whose claim no longer matches the stored task.

``TaskSuccess`` / ``TaskFailure`` (the handler outcomes) are the payloads of
``commit_success`` / ``commit_failure``; committing applies the outcome's
commands atomically with the task's terminal transition.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from .models import (
    JobRecord,
    TaskAttemptRecord,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskTransitionRecord,
)
from .protocols import TaskFailure, TaskSuccess


class JobStoreError(Exception):
    """Base for JobStore failures (claim lost, not found, conflict)."""


class JobNotFoundError(JobStoreError):
    pass


class TaskNotFoundError(JobStoreError):
    pass


class TaskClaimLostError(JobStoreError):
    """The fencing check failed -- this worker no longer owns the task."""


class TaskBudgetExceededError(JobStoreError):
    """A Job budget (max_tasks / max_depth / aggregate attempts / runtime) was
    exceeded. Raised atomically when a task-creation command would push the job
    past its cap, so the whole commit fails and no partial children land."""


class InvalidTaskCommandError(JobStoreError):
    """A handler command is invalid in the current state -- e.g. CompleteJob
    while sibling tasks are still live, or a task waiting on more than one
    signal. Raised at commit so the task transitions to a terminal failure
    rather than silently dropping the command."""


class UnsupportedTaskSchemaError(JobStoreError):
    """A persisted task/job envelope used a schema version this code does not
    know how to read. Raised on read so a future-version row is never silently
    misinterpreted (e.g. security fields restored to broader permission)."""


class RunnableBindingError(JobStoreError):
    """A task's pinned runnable resolution does not match the freshly resolved
    one. Raised by ``bind_runnable`` so a retry can never silently re-run a
    different agent after a mapping change; the handler maps it to a permanent
    failure."""


class TaskRunTimeoutError(TimeoutError):
    """Raised by ``run_one_task`` when its wait budget elapses before the job
    reaches a terminal state. The job is cancelled (so in-flight work stops)
    rather than returning a still-running task the caller would mistake for
    finished."""

    def __init__(self, job_id: str, task_id: str, timeout_seconds: float) -> None:
        super().__init__(
            f"run_one_task for task {task_id} (job {job_id}) did not finish "
            f"within {timeout_seconds}s"
        )
        self.job_id = job_id
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds

class TaskCancellationDidNotConvergeError(RuntimeError):
    def __init__(self, job_id, task_id, job_status, task_status, cancel_grace_seconds):
        super().__init__(f"cancellation did not converge for task {task_id} (job {job_id})")
        self.job_id, self.task_id = job_id, task_id
        self.job_status, self.task_status = job_status, task_status
        self.cancel_grace_seconds = cancel_grace_seconds


@dataclass(frozen=True, slots=True)
class TaskClaim:
    task_id: str
    attempt_id: str
    worker_id: str
    fencing_token: int


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    claim: TaskClaim
    job: JobRecord
    task: TaskRecord
    attempt: TaskAttemptRecord


def claim_matches_task(claim: TaskClaim, task: TaskRecord) -> bool:
    """Pure ownership check: does ``task`` still belong to ``claim``? Every field
    that fencing protects -- status (CLAIMED or CANCELLING), lease_owner (the
    worker), active_attempt_id, and fencing_token -- must match. A worker whose
    lease expired and was reclaimed fails on lease_owner / fencing_token and is
    told it no longer owns the task. The worker's re-read check uses this; the
    store guards inline the same predicate (with extra not-found handling) at
    their fenced write sites."""
    return (
        task.status in {TaskStatus.CLAIMED, TaskStatus.CANCELLING}
        and task.lease_owner == claim.worker_id
        and task.active_attempt_id == claim.attempt_id
        and task.fencing_token == claim.fencing_token
    )


@runtime_checkable
class JobStore(Protocol):
    async def create_job(
        self,
        job: JobRecord,
        root_task: TaskRecord,
    ) -> JobRecord: ...

    async def get_job(self, job_id: str) -> "JobRecord | None": ...

    async def get_task(self, task_id: str) -> "TaskRecord | None": ...

    async def list_tasks(
        self,
        job_id: str,
        *,
        status: "TaskStatus | None" = None,
    ) -> "tuple[TaskRecord, ...]": ...

    async def claim(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        handlers: "tuple[str, ...] | None" = None,
    ) -> "ClaimedTask | None": ...

    async def renew_lease(
        self,
        *,
        task_id: str,
        attempt_id: str,
        worker_id: str,
        fencing_token: int,
        now: datetime,
        lease_seconds: float,
    ) -> TaskRecord: ...

    async def bind_run(
        self,
        *,
        task_id: str,
        attempt_id: str,
        fencing_token: int,
        worker_id: str,
        run_id: str,
    ) -> TaskAttemptRecord: ...

    async def bind_runnable(
        self,
        *,
        task_id: str,
        attempt_id: str,
        fencing_token: int,
        worker_id: str,
        runnable_id: str,
        revision: "str | None",
        fingerprint: str,
    ) -> TaskRecord: ...

    async def commit_success(
        self,
        claim: TaskClaim,
        outcome: TaskSuccess,
    ) -> TaskRecord: ...

    async def commit_failure(
        self,
        claim: TaskClaim,
        outcome: TaskFailure,
    ) -> TaskRecord: ...

    async def request_cancel(
        self,
        job_id: str,
        *,
        reason: "str | None" = None,
    ) -> JobRecord: ...

    async def submit_signal(
        self,
        signal: TaskSignalRecord,
    ) -> TaskSignalRecord: ...

    async def recover_expired(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> "tuple[TaskRecord, ...]": ...

    async def reconcile_due(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> "tuple[TaskRecord, ...]": ...

    async def list_orphan_run_ids(self, *, limit: int = 500) -> "tuple[str, ...]": ...

    async def list_attempts(
        self,
        task_id: str,
    ) -> "tuple[TaskAttemptRecord, ...]": ...

    async def list_transitions(
        self,
        job_id: str,
    ) -> "tuple[TaskTransitionRecord, ...]": ...


__all__: "list[str]" = [
    "TaskClaim",
    "ClaimedTask",
    "JobStore",
    "JobStoreError",
    "JobNotFoundError",
    "TaskNotFoundError",
    "TaskClaimLostError",
    "TaskBudgetExceededError",
    "InvalidTaskCommandError",
    "UnsupportedTaskSchemaError",
    "RunnableBindingError",
    "TaskRunTimeoutError",
    "TaskCancellationDidNotConvergeError",
    "claim_matches_task",
]
