#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskStore contract.

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


class TaskStoreError(Exception):
    """Base for TaskStore failures (claim lost, not found, conflict)."""


class JobNotFoundError(TaskStoreError):
    pass


class TaskNotFoundError(TaskStoreError):
    pass


class TaskClaimLostError(TaskStoreError):
    """The fencing check failed -- this worker no longer owns the task."""


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


@runtime_checkable
class TaskStore(Protocol):
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
    "TaskStore",
    "TaskStoreError",
    "JobNotFoundError",
    "TaskNotFoundError",
    "TaskClaimLostError",
]
