#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyTaskStore: DB-backed TaskStore.

Mirrors the other SQLAlchemy stores: a ``session_factory`` constructor and
read-check-mutate-commit transactions. The reliable-task semantics live in SQL:

* ``claim`` promotes due RETRY_WAIT tasks, then issues an atomic
  ``UPDATE ... SET status='claimed' WHERE id=? AND status='ready'``; the WHERE
  clause + ``rowcount`` is the race-decider so two workers never claim the same
  task (``FOR UPDATE SKIP LOCKED`` is a no-op on SQLite, real on Postgres).
* ``commit_success`` / ``commit_failure`` / ``renew_lease`` / ``bind_run``
  re-check ``status='claimed' AND lease_owner AND active_attempt_id AND
  fencing_token`` in the UPDATE WHERE -- a stale worker (lease expired and
  reclaimed) updates 0 rows and raises :class:`TaskClaimLostError`.
* ``recover_expired`` resets CLAIMED tasks whose lease expired (SUPERSEDE the
  attempt, READY or FAILED per side-effect / attempts-exhausted), and converges
  job state so a crash between a task's terminal write and the job-completion
  write does not leave the job stuck.

Complex policy/context fields are stored as a JSON envelope (``data_json``)
alongside the indexed query columns. Command application lands in a later
phase; for now a completed root task completes the job.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ...task.models import (
    AttemptStatus,
    JobRecord,
    JobStatus,
    JOB_TRANSITIONS,
    JOB_TERMINAL,
    ResourceSnapshotRef,
    RetryPolicy,
    SideEffectMode,
    SideEffectPolicy,
    TaskAttemptRecord,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
    ActorChain,
    TaskRecord,
    TaskStatus,
    TaskTransitionRecord,
    assert_job_transition,
    assert_task_transition,
    from_jsonable,
    to_jsonable,
)
from ...task.protocols import (
    CancelJob,
    CancelTask,
    Clock,
    CompleteJob,
    CreateTask,
    SystemClock,
    TaskFailure,
    TaskSuccess,
    WaitSignal,
)
from ...task.store import (
    ClaimedTask,
    JobNotFoundError,
    TaskClaim,
    TaskClaimLostError,
    TaskNotFoundError,
)
from .models import (
    TaskAttemptRow,
    TaskJobRow,
    TaskRow,
    TaskSignalRow,
    TaskTransitionRow,
)


def _as_utc(value: "datetime | None") -> "datetime | None":
    # aiosqlite round-trips datetimes as naive; re-stamp them UTC so comparisons
    # against tz-aware clock values hold.
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _store_dt(value: "datetime | None") -> "datetime | None":
    # Inverse of _as_utc for the write side: store/compare as naive UTC so the
    # ORM evaluator never has to compare a naive DB value against an aware one.
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _task_envelope(task: TaskRecord) -> str:
    return json.dumps(
        {
            "dependencies": list(task.dependencies),
            "retry_policy": to_jsonable(task.retry_policy),
            "side_effect_policy": to_jsonable(task.side_effect_policy),
            "resource_snapshots": to_jsonable(task.resource_snapshots),
            "metadata": dict(task.metadata),
        }
    )


def _row_to_task(row: TaskRow) -> TaskRecord:
    env = json.loads(row.data_json)
    return TaskRecord(
        id=row.id,
        job_id=row.job_id,
        parent_task_id=row.parent_task_id,
        key=row.key,
        handler=row.handler,
        status=TaskStatus(row.status),
        input_artifact_id=row.input_artifact_id,
        output_artifact_id=row.output_artifact_id,
        dependencies=tuple(env["dependencies"]),
        retry_policy=from_jsonable(RetryPolicy, env["retry_policy"]),
        side_effect_policy=from_jsonable(SideEffectPolicy, env["side_effect_policy"]),
        attempt_count=row.attempt_count,
        available_at=_as_utc(row.available_at),
        lease_owner=row.lease_owner,
        lease_expires_at=_as_utc(row.lease_expires_at),
        fencing_token=row.fencing_token,
        active_attempt_id=row.active_attempt_id,
        timeout_seconds=row.timeout_seconds,
        resource_snapshots=tuple(
            from_jsonable(ResourceSnapshotRef, s) for s in env["resource_snapshots"]
        ),
        version=row.version,
        created_at=_as_utc(row.created_at),
        updated_at=_as_utc(row.updated_at),
        depth=env.get("depth", 0),
        delegated_scopes=(
            tuple(env["delegated_scopes"])
            if env.get("delegated_scopes") is not None
            else None
        ),
        actor_chain=(
            from_jsonable(ActorChain, env["actor_chain"])
            if env.get("actor_chain") is not None
            else None
        ),
        metadata=env["metadata"],
    )


def _job_envelope(job: JobRecord) -> str:
    return json.dumps(
        {
            "principal": to_jsonable(job.principal),
            "actor_chain": to_jsonable(job.actor_chain),
            "budget": to_jsonable(job.budget),
            "metadata": dict(job.metadata),
        }
    )


def _row_to_job(row: TaskJobRow) -> JobRecord:
    env = json.loads(row.data_json)
    return JobRecord(
        id=row.id,
        status=JobStatus(row.status),
        principal=from_jsonable(TaskPrincipal, env["principal"]),
        actor_chain=from_jsonable(ActorChain, env["actor_chain"]),
        budget=from_jsonable(TaskBudget, env["budget"]),
        root_task_id=row.root_task_id,
        input_artifact_id=row.input_artifact_id,
        output_artifact_id=row.output_artifact_id,
        version=row.version,
        created_at=_as_utc(row.created_at),
        started_at=_as_utc(row.started_at),
        finished_at=_as_utc(row.finished_at),
        metadata=env["metadata"],
    )


def _row_to_attempt(row: TaskAttemptRow) -> TaskAttemptRecord:
    env = json.loads(row.data_json) if row.data_json else {}
    return TaskAttemptRecord(
        id=row.id,
        task_id=row.task_id,
        job_id=row.job_id,
        attempt=row.attempt,
        worker_id=row.worker_id,
        fencing_token=row.fencing_token,
        status=AttemptStatus(row.status),
        started_at=_as_utc(row.started_at),
        run_id=row.run_id,
        finished_at=_as_utc(row.finished_at),
        failure_kind=TaskFailureKind(row.failure_kind) if row.failure_kind else None,
        error_type=row.error_type,
        error_message=row.error_message,
        metadata=env.get("metadata", {}),
    )


class SqlAlchemyTaskStore:
    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        clock: "Clock | None" = None,
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock or SystemClock()
        self._session = session

    async def _in_session(self, action):
        if self._session is not None:
            result = await action(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            try:
                result = await action(session)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise

    def _transition(
        self, job_id, *, task_id, attempt_id, from_status, to_status, reason, now
    ) -> TaskTransitionRow:
        return TaskTransitionRow(
            job_id=job_id,
            task_id=task_id,
            attempt_id=attempt_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            occurred_at=now,
            data_json="{}",
        )

    # --------------------------------------------------------- public API --

    async def create_job(self, job: JobRecord, root_task: TaskRecord) -> JobRecord:
        async def do(session: AsyncSession) -> JobRecord:
            existing = await session.get(TaskJobRow, job.id)
            if existing is not None:
                raise FileExistsError(job.id)
            session.add(
                TaskJobRow(
                    id=job.id,
                    status=job.status.value,
                    tenant_id=job.principal.tenant_id,
                    root_task_id=job.root_task_id,
                    input_artifact_id=job.input_artifact_id,
                    output_artifact_id=job.output_artifact_id,
                    version=job.version,
                    created_at=_store_dt(job.created_at),
                    started_at=_store_dt(job.started_at),
                    finished_at=_store_dt(job.finished_at),
                    data_json=_job_envelope(job),
                )
            )
            root = root_task
            root = _with_status(root, TaskStatus.READY)
            session.add(_task_to_row(root))
            session.add(
                self._transition(
                    job.id,
                    task_id=root.id,
                    attempt_id=None,
                    from_status=None,
                    to_status=TaskStatus.READY.value,
                    reason="created",
                    now=_store_dt(job.created_at),
                )
            )
            return job

        return await self._in_session(do)

    async def get_job(self, job_id: str) -> "JobRecord | None":
        async def do(session: AsyncSession):
            row = await session.get(TaskJobRow, job_id)
            return _row_to_job(row) if row is not None else None

        return await self._in_session(do)

    async def get_task(self, task_id: str) -> "TaskRecord | None":
        async def do(session: AsyncSession):
            row = await session.get(TaskRow, task_id)
            return _row_to_task(row) if row is not None else None

        return await self._in_session(do)

    async def list_tasks(
        self, job_id: str, *, status: "TaskStatus | None" = None
    ) -> "tuple[TaskRecord, ...]":
        async def do(session: AsyncSession):
            stmt = select(TaskRow).where(TaskRow.job_id == job_id)
            if status is not None:
                stmt = stmt.where(TaskRow.status == status.value)
            stmt = stmt.order_by(TaskRow.created_at)
            rows = (await session.execute(stmt)).scalars().all()
            return tuple(_row_to_task(r) for r in rows)

        return await self._in_session(do)

    async def claim(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        handlers: "tuple[str, ...] | None" = None,
    ) -> "ClaimedTask | None":
        now = _store_dt(now)  # DB layer compares/stores naive UTC

        async def do(session: AsyncSession):
            # Promote due RETRY_WAIT tasks to READY first (own transition each).
            due = (
                (
                    await session.execute(
                        select(TaskRow)
                        .where(TaskRow.status == TaskStatus.RETRY_WAIT.value)
                        .where(TaskRow.available_at <= now)
                    )
                )
                .scalars()
                .all()
            )
            for t in due:
                promo = await session.execute(
                    update(TaskRow)
                    .where(TaskRow.id == t.id)
                    .where(TaskRow.status == TaskStatus.RETRY_WAIT.value)
                    .values(
                        status=TaskStatus.READY.value,
                        version=TaskRow.version + 1,
                        updated_at=now,
                    )
                )
                # Only audit the promotion if this CAS actually won the race.
                if promo.rowcount == 1:
                    session.add(
                        self._transition(
                            t.job_id,
                            task_id=t.id,
                            attempt_id=None,
                            from_status=TaskStatus.RETRY_WAIT.value,
                            to_status=TaskStatus.READY.value,
                            reason="retry_due",
                            now=now,
                        )
                    )
            # Pick the earliest READY candidate whose job is active. Filter
            # handlers in SQL so the (handler, status, available_at) index is
            # used and the limit window is not wasted on other handlers.
            cand_stmt = select(TaskRow).where(
                TaskRow.status == TaskStatus.READY.value,
                TaskRow.available_at <= now,
            )
            if handlers is not None:
                cand_stmt = cand_stmt.where(TaskRow.handler.in_(list(handlers)))
            cand_stmt = (
                cand_stmt.order_by(TaskRow.available_at, TaskRow.created_at)
                .limit(64)
                .with_for_update(skip_locked=True)
            )
            candidates = (await session.execute(cand_stmt)).scalars().all()
            for candidate in candidates:
                if handlers is not None and candidate.handler not in handlers:
                    continue
                job_row = await session.get(TaskJobRow, candidate.job_id)
                if job_row is None:
                    continue
                if job_row.status in (
                    JobStatus.CANCELLING.value,
                    JobStatus.CANCELLED.value,
                ):
                    continue
                # Budget enforcement: runtime cap, task cap, and total
                # attempt cap. started_at is a naive-UTC column.
                job_env = json.loads(job_row.data_json)
                budget = job_env.get("budget") or {}
                max_runtime = budget.get("max_runtime_seconds")
                if (
                    max_runtime is not None
                    and job_row.started_at is not None
                    and (now - job_row.started_at).total_seconds() > max_runtime
                ):
                    continue
                max_tasks = budget.get("max_tasks")
                if max_tasks is not None:
                    task_count = (
                        await session.execute(
                            select(func.count())
                            .select_from(TaskRow)
                            .where(TaskRow.job_id == candidate.job_id)
                        )
                    ).scalar_one()
                    if task_count >= max_tasks:
                        continue
                max_attempts = budget.get("max_attempts")
                if max_attempts is not None:
                    attempt_total = (
                        await session.execute(
                            select(
                                func.coalesce(func.sum(TaskRow.attempt_count), 0)
                            ).where(TaskRow.job_id == candidate.job_id)
                        )
                    ).scalar_one()
                    if attempt_total >= max_attempts:
                        continue
                attempt_number = candidate.attempt_count + 1
                attempt_id = f"{candidate.id}-att{attempt_number}"
                # Atomic CAS: READY -> CLAIMED. rowcount==1 means we won the race.
                result = await session.execute(
                    update(TaskRow)
                    .where(TaskRow.id == candidate.id)
                    .where(TaskRow.status == TaskStatus.READY.value)
                    .where(TaskRow.available_at <= now)
                    .values(
                        status=TaskStatus.CLAIMED.value,
                        lease_owner=worker_id,
                        lease_expires_at=now + timedelta(seconds=lease_seconds),
                        fencing_token=TaskRow.fencing_token + 1,
                        attempt_count=TaskRow.attempt_count + 1,
                        active_attempt_id=attempt_id,
                        version=TaskRow.version + 1,
                        updated_at=now,
                    )
                )
                if result.rowcount != 1:
                    continue  # lost the race; try the next candidate
                await session.refresh(candidate)
                task = _row_to_task(candidate)
                attempt = TaskAttemptRecord(
                    id=attempt_id,
                    task_id=task.id,
                    job_id=task.job_id,
                    attempt=attempt_number,
                    worker_id=worker_id,
                    fencing_token=task.fencing_token,
                    status=AttemptStatus.RUNNING,
                    started_at=_as_utc(now),  # returned value: tz-aware
                    run_id=None,
                    finished_at=None,
                    failure_kind=None,
                    error_type=None,
                    error_message=None,
                )
                session.add(_attempt_to_row(attempt))
                session.add(
                    self._transition(
                        task.job_id,
                        task_id=task.id,
                        attempt_id=attempt_id,
                        from_status=TaskStatus.READY.value,
                        to_status=TaskStatus.CLAIMED.value,
                        reason="claimed",
                        now=now,
                    )
                )
                if job_row.status == JobStatus.PENDING.value:
                    job_row.status = JobStatus.RUNNING.value
                    job_row.started_at = now
                    job_row.version += 1
                return ClaimedTask(
                    claim=TaskClaim(
                        task_id=task.id,
                        attempt_id=attempt_id,
                        worker_id=worker_id,
                        fencing_token=task.fencing_token,
                    ),
                    job=_row_to_job(job_row),
                    task=task,
                    attempt=attempt,
                )
            return None

        return await self._in_session(do)

    async def commit_success(
        self, claim: TaskClaim, outcome: TaskSuccess
    ) -> TaskRecord:
        from ...task.validation import (
            validate_child_tasks,
            validate_commands,
            validate_output_payload,
        )

        validate_commands(len(outcome.commands))
        validate_child_tasks(
            sum(1 for c in outcome.commands if isinstance(c, CreateTask))
        )
        if outcome.output_artifact is not None:
            validate_output_payload(outcome.output_artifact.size)
        now = _store_dt(self._clock.now())

        async def do(session: AsyncSession) -> TaskRecord:
            # Cancel precedence: if the task was moved to CANCELLING
            # while the handler ran, discard the outcome and land CANCELLED.
            if (
                await self._require_claimed(session, claim)
            ).status == TaskStatus.CANCELLING.value:
                return await self._commit_cancelled_sql(session, claim, now)
            has_wait = any(isinstance(c, WaitSignal) for c in outcome.commands)
            target = TaskStatus.WAITING if has_wait else TaskStatus.SUCCEEDED

            extra_values: dict = {
                "output_artifact_id": (
                    outcome.output_artifact.id
                    if outcome.output_artifact is not None
                    else None
                ),
            }

            if has_wait:
                # Persist WaitSignal conditions in task metadata so
                # submit_signal can match them.
                row = await self._require_claimed(session, claim)
                env = json.loads(row.data_json)
                wait_signals = [
                    c for c in outcome.commands if isinstance(c, WaitSignal)
                ]
                env["metadata"] = {
                    **env.get("metadata", {}),
                    "wait_on": [
                        {"name": ws.name, "correlation_key": ws.correlation_key}
                        for ws in wait_signals
                    ],
                }
                extra_values["data_json"] = json.dumps(env)

            updated = await self._fenced_update(
                session,
                claim,
                status=target.value,
                extra_values=extra_values,
                now=now,
            )
            await session.execute(
                update(TaskAttemptRow)
                .where(TaskAttemptRow.id == claim.attempt_id)
                .values(status=AttemptStatus.SUCCEEDED.value, finished_at=now)
            )
            session.add(
                self._transition(
                    updated.job_id,
                    task_id=updated.id,
                    attempt_id=claim.attempt_id,
                    from_status=TaskStatus.CLAIMED.value,
                    to_status=target.value,
                    reason="wait_signal" if has_wait else "succeeded",
                    now=now,
                )
            )

            # Apply commands atomically.
            job_record: "JobRecord | None" = None
            for cmd in outcome.commands:
                if isinstance(cmd, CreateTask):
                    if job_record is None:
                        jr = await session.get(TaskJobRow, updated.job_id)
                        if jr is None:
                            # Invariant: a task being committed belongs to an
                            # existing job; skip child creation if it vanished.
                            continue
                        job_record = _row_to_job(jr)
                    await self._apply_create_task_sql(
                        session,
                        updated.job_id,
                        updated,
                        cmd,
                        now,
                        job_record,
                    )
                elif isinstance(cmd, CompleteJob):
                    job_row = await session.get(TaskJobRow, updated.job_id)
                    if job_row and job_row.status == JobStatus.RUNNING.value:
                        job_row.status = JobStatus.SUCCEEDED.value
                        job_row.finished_at = now
                        job_row.version += 1
                elif isinstance(cmd, CancelTask):
                    ct = await session.get(TaskRow, cmd.task_id)
                    if ct and ct.status not in (
                        TaskStatus.SUCCEEDED.value,
                        TaskStatus.FAILED.value,
                        TaskStatus.CANCELLED.value,
                    ):
                        # Validate the transition through the state machine
                        # (current -> CANCELLING -> CANCELLED) so an illegal
                        # source state is rejected, not silently overwritten.
                        assert_task_transition(TaskStatus(ct.status), TaskStatus.CANCELLING)
                        assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
                        ct.status = TaskStatus.CANCELLED.value
                        ct.updated_at = now
                        ct.version += 1
                elif isinstance(cmd, CancelJob):
                    target_job = cmd.job_id or updated.job_id
                    await self._cancel_job_sql(session, target_job, now)

            # Resolve dependencies: PENDING → READY when all deps SUCCEEDED.
            await self._resolve_dependencies_sql(session, updated.job_id, now)
            await self._maybe_complete_job(session, updated.job_id, now)
            return _row_to_task(updated)

        return await self._in_session(do)

    async def commit_failure(
        self, claim: TaskClaim, outcome: TaskFailure
    ) -> TaskRecord:
        now = _store_dt(self._clock.now())

        async def do(session: AsyncSession) -> TaskRecord:
            task_row = await self._require_claimed(session, claim)
            if task_row.status == TaskStatus.CANCELLING.value:
                # Cancel precedence: land CANCELLED, not FAILED/RETRY.
                return await self._commit_cancelled_sql(session, claim, now)
            task = _row_to_task(task_row)
            retryable = outcome.retryable
            if retryable is None:
                retryable = outcome.kind in task.retry_policy.retryable_kinds
            non_idempotent = (
                task.side_effect_policy.mode == SideEffectMode.NON_IDEMPOTENT
            )
            can_retry = (
                retryable
                and not non_idempotent
                and task.attempt_count < task.retry_policy.max_attempts
            )
            await session.execute(
                update(TaskAttemptRow)
                .where(TaskAttemptRow.id == claim.attempt_id)
                .values(
                    status=AttemptStatus.FAILED.value,
                    finished_at=now,
                    failure_kind=outcome.kind.value,
                    error_type=outcome.error_type,
                    error_message=outcome.message,
                )
            )
            if can_retry:
                target = TaskStatus.RETRY_WAIT
                values = {
                    "status": target.value,
                    "available_at": now
                    + timedelta(
                        seconds=_retry_delay(task.retry_policy, task.attempt_count)
                    ),
                }
                reason = "retry"
            else:
                target = TaskStatus.FAILED
                values = {"status": target.value}
                reason = "failed"
            updated = await self._fenced_update(
                session, claim, extra_values=values, now=now
            )
            session.add(
                self._transition(
                    updated.job_id,
                    task_id=updated.id,
                    attempt_id=claim.attempt_id,
                    from_status=TaskStatus.CLAIMED.value,
                    to_status=target.value,
                    reason=reason,
                    now=now,
                )
            )
            await self._maybe_complete_job(session, updated.job_id, now)
            return _row_to_task(updated)

        return await self._in_session(do)

    async def _commit_cancelled_sql(
        self, session: AsyncSession, claim: TaskClaim, now: datetime
    ) -> TaskRecord:
        """Land a CANCELLING task as CANCELLED -- cancel takes precedence over
        the handler's outcome. The owning worker still holds the claim
        (verified by the caller via _require_claimed); no commands are applied."""
        row = await self._require_claimed(session, claim)
        from_status = row.status
        row.status = TaskStatus.CANCELLED.value
        row.lease_owner = None
        row.lease_expires_at = None
        row.active_attempt_id = None
        row.updated_at = now
        row.version += 1
        await session.execute(
            update(TaskAttemptRow)
            .where(TaskAttemptRow.id == claim.attempt_id)
            .where(TaskAttemptRow.status == AttemptStatus.RUNNING.value)
            .values(status=AttemptStatus.CANCELLED.value, finished_at=now)
        )
        session.add(
            self._transition(
                row.job_id,
                task_id=row.id,
                attempt_id=claim.attempt_id,
                from_status=from_status,
                to_status=TaskStatus.CANCELLED.value,
                reason="cancelled",
                now=now,
            )
        )
        await self._maybe_complete_job(session, row.job_id, now)
        return _row_to_task(row)

    async def request_cancel(
        self, job_id: str, *, reason: "str | None" = None
    ) -> JobRecord:
        now = _store_dt(self._clock.now())

        async def do(session: AsyncSession) -> JobRecord:
            job_row = await session.get(TaskJobRow, job_id)
            if job_row is None:
                raise JobNotFoundError(job_id)
            if job_row.status in (
                JobStatus.SUCCEEDED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ):
                # Already terminal: cancelling a finished job is a no-op (the
                # race with a concurrent commit must not raise).
                return _row_to_job(job_row)
            if job_row.status != JobStatus.CANCELLING.value:
                assert_job_transition(JobStatus(job_row.status), JobStatus.CANCELLING)
            job_row.status = JobStatus.CANCELLING.value
            job_row.version += 1
            tasks = (
                (await session.execute(select(TaskRow).where(TaskRow.job_id == job_id)))
                .scalars()
                .all()
            )
            for t in tasks:
                if t.status in (
                    TaskStatus.PENDING.value,
                    TaskStatus.READY.value,
                    TaskStatus.WAITING.value,
                    TaskStatus.RETRY_WAIT.value,
                ):
                    from_status = t.status
                    t.status = TaskStatus.CANCELLED.value
                    t.updated_at = now
                    t.version += 1
                    session.add(
                        self._transition(
                            job_id,
                            task_id=t.id,
                            attempt_id=None,
                            from_status=from_status,
                            to_status=TaskStatus.CANCELLED.value,
                            reason="cancelled",
                            now=now,
                        )
                    )
                elif t.status == TaskStatus.CLAIMED.value:
                    #  an in-flight task moves to CANCELLING so the owning
                    # worker observes it (heartbeat/watcher) and lands CANCELLED
                    # when its handler stops and commits.
                    from_status = t.status
                    t.status = TaskStatus.CANCELLING.value
                    t.updated_at = now
                    t.version += 1
                    session.add(
                        self._transition(
                            job_id,
                            task_id=t.id,
                            attempt_id=None,
                            from_status=from_status,
                            to_status=TaskStatus.CANCELLING.value,
                            reason="cancelling",
                            now=now,
                        )
                    )
            # The job cancels outright only when nothing is still in-flight:
            # neither CLAIMED nor mid-cancellation (CANCELLING).
            if not any(
                t.status in (TaskStatus.CLAIMED.value, TaskStatus.CANCELLING.value)
                for t in tasks
            ):
                assert_job_transition(JobStatus.CANCELLING, JobStatus.CANCELLED)
                job_row.status = JobStatus.CANCELLED.value
                job_row.finished_at = now
            return _row_to_job(job_row)

        return await self._in_session(do)

    async def submit_signal(self, signal) -> object:
        from ...task.validation import validate_metadata

        validate_metadata(dict(signal.metadata))

        async def do(session: AsyncSession):
            existing = await session.get(TaskSignalRow, signal.id)
            if existing is not None:
                return signal
            session.add(
                TaskSignalRow(
                    id=signal.id,
                    job_id=signal.job_id,
                    name=signal.name,
                    correlation_key=signal.correlation_key,
                    payload_artifact_id=signal.payload_artifact_id,
                    created_at=_store_dt(signal.created_at),
                    consumed_by_task_id=signal.consumed_by_task_id,
                    data_json=json.dumps(dict(signal.metadata)),
                )
            )
            waiting = (
                (
                    await session.execute(
                        select(TaskRow)
                        .where(TaskRow.job_id == signal.job_id)
                        .where(TaskRow.status == TaskStatus.WAITING.value)
                    )
                )
                .scalars()
                .all()
            )
            woken = None
            for t in waiting:
                env = json.loads(t.data_json)
                wait_on = env.get("metadata", {}).get("wait_on", [])
                # A WAITING task with no recorded wait conditions matches
                # nothing: only an explicit (name, correlation_key) wakes it.
                if any(
                    w["name"] == signal.name
                    and w["correlation_key"] == signal.correlation_key
                    for w in wait_on
                ):
                    woken = t
                    break
            if woken is not None:
                from_status = woken.status
                woken.status = TaskStatus.READY.value
                woken.available_at = _store_dt(signal.created_at)
                woken.updated_at = _store_dt(signal.created_at)
                woken.version += 1
                session.add(
                    self._transition(
                        signal.job_id,
                        task_id=woken.id,
                        attempt_id=None,
                        from_status=from_status,
                        to_status=TaskStatus.READY.value,
                        reason="signal",
                        now=_store_dt(signal.created_at),
                    )
                )
                # Record which task consumed this signal (the reverse link).
                await session.execute(
                    update(TaskSignalRow)
                    .where(TaskSignalRow.id == signal.id)
                    .values(consumed_by_task_id=woken.id)
                )
            # A woken task may move a WAITING job back to RUNNING,
            # or complete the job if all tasks are now terminal.
            await self._maybe_complete_job(session, signal.job_id, _store_dt(signal.created_at))
            return signal

        return await self._in_session(do)

    async def recover_expired(
        self, *, now: datetime, limit: int = 100
    ) -> "tuple[TaskRecord, ...]":
        now = _store_dt(now)

        async def do(session: AsyncSession):
            rows = (
                (
                    await session.execute(
                        select(TaskRow)
                        .where(
                            TaskRow.status.in_(
                                [
                                    TaskStatus.CLAIMED.value,
                                    TaskStatus.CANCELLING.value,
                                ]
                            )
                        )
                        .where(TaskRow.lease_expires_at < now)
                        .order_by(TaskRow.lease_expires_at)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            recovered: list = []
            for t in rows:
                if t.status == TaskStatus.CANCELLING.value:
                    #  a CANCELLING task whose lease expired will never be
                    # committed by its (now-gone) worker -- finalize CANCELLED.
                    if t.active_attempt_id:
                        await session.execute(
                            update(TaskAttemptRow)
                            .where(TaskAttemptRow.id == t.active_attempt_id)
                            .where(TaskAttemptRow.status == AttemptStatus.RUNNING.value)
                            .values(
                                status=AttemptStatus.CANCELLED.value, finished_at=now
                            )
                        )
                    from_status = t.status
                    attempt_id = t.active_attempt_id
                    t.status = TaskStatus.CANCELLED.value
                    t.lease_owner = None
                    t.lease_expires_at = None
                    t.active_attempt_id = None
                    t.updated_at = now
                    t.version += 1
                    session.add(
                        self._transition(
                            t.job_id,
                            task_id=t.id,
                            attempt_id=attempt_id,
                            from_status=from_status,
                            to_status=TaskStatus.CANCELLED.value,
                            reason="cancelled",
                            now=now,
                        )
                    )
                    recovered.append(_row_to_task(t))
                    continue
                task = _row_to_task(t)
                non_idempotent = (
                    task.side_effect_policy.mode == SideEffectMode.NON_IDEMPOTENT
                )
                exhausted = task.attempt_count >= task.retry_policy.max_attempts
                target = (
                    TaskStatus.FAILED
                    if (non_idempotent or exhausted)
                    else TaskStatus.READY
                )
                if t.active_attempt_id:
                    await session.execute(
                        update(TaskAttemptRow)
                        .where(TaskAttemptRow.id == t.active_attempt_id)
                        .where(TaskAttemptRow.status == AttemptStatus.RUNNING.value)
                        .values(status=AttemptStatus.SUPERSEDED.value, finished_at=now)
                    )
                t.status = target.value
                t.lease_owner = None
                t.lease_expires_at = None
                t.active_attempt_id = None
                t.available_at = now
                t.updated_at = now
                t.version += 1
                reason = (
                    "non_idempotent"
                    if non_idempotent
                    else "attempts_exhausted"
                    if exhausted
                    else "lease_expired"
                )
                session.add(
                    self._transition(
                        t.job_id,
                        task_id=t.id,
                        attempt_id=task.active_attempt_id,
                        from_status=TaskStatus.CLAIMED.value,
                        to_status=target.value,
                        reason=reason,
                        now=now,
                    )
                )
                recovered.append(_row_to_task(t))
            await self._reconcile_signals_sql(session, now)
            await self._maybe_complete_job_all(session, now)
            return tuple(recovered)

        return await self._in_session(do)

    async def list_attempts(self, task_id: str) -> "tuple[TaskAttemptRecord, ...]":
        async def do(session: AsyncSession):
            rows = (
                (
                    await session.execute(
                        select(TaskAttemptRow)
                        .where(TaskAttemptRow.task_id == task_id)
                        .order_by(TaskAttemptRow.attempt)
                    )
                )
                .scalars()
                .all()
            )
            return tuple(_row_to_attempt(r) for r in rows)

        return await self._in_session(do)

    async def list_transitions(self, job_id: str) -> "tuple[TaskTransitionRecord, ...]":
        async def do(session: AsyncSession):
            rows = (
                (
                    await session.execute(
                        select(TaskTransitionRow)
                        .where(TaskTransitionRow.job_id == job_id)
                        .order_by(TaskTransitionRow.id)
                    )
                )
                .scalars()
                .all()
            )
            return tuple(
                TaskTransitionRecord(
                    id=str(r.id),
                    job_id=r.job_id,
                    task_id=r.task_id,
                    attempt_id=r.attempt_id,
                    from_status=r.from_status,
                    to_status=r.to_status,
                    reason=r.reason,
                    occurred_at=_as_utc(r.occurred_at),
                    metadata=json.loads(r.data_json) if r.data_json else {},
                )
                for r in rows
            )

        return await self._in_session(do)

    async def renew_lease(
        self,
        *,
        task_id: str,
        attempt_id: str,
        worker_id: str,
        fencing_token: int,
        now: datetime,
        lease_seconds: float,
    ) -> TaskRecord:
        now = _store_dt(now)

        async def do(session: AsyncSession):
            result = await session.execute(
                update(TaskRow)
                .where(TaskRow.id == task_id)
                .where(TaskRow.status == TaskStatus.CLAIMED.value)
                .where(TaskRow.lease_owner == worker_id)
                .where(TaskRow.active_attempt_id == attempt_id)
                .where(TaskRow.fencing_token == fencing_token)
                .values(
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                    version=TaskRow.version + 1,
                )
            )
            if result.rowcount != 1:
                raise TaskClaimLostError(task_id)
            return _row_to_task(await session.get(TaskRow, task_id))

        return await self._in_session(do)

    async def bind_run(
        self,
        *,
        task_id: str,
        attempt_id: str,
        fencing_token: int,
        worker_id: str,
        run_id: str,
    ) -> TaskAttemptRecord:
        async def do(session: AsyncSession):
            # 4-field fencing: status + lease_owner + active_attempt_id +
            # fencing_token must all match the live claim.
            task_row = await session.get(TaskRow, task_id)
            if task_row is None:
                raise TaskNotFoundError(task_id)
            if (
                task_row.status != TaskStatus.CLAIMED.value
                or task_row.lease_owner != worker_id
                or task_row.active_attempt_id != attempt_id
                or task_row.fencing_token != fencing_token
            ):
                raise TaskClaimLostError(task_id)
            await session.execute(
                update(TaskAttemptRow)
                .where(TaskAttemptRow.id == attempt_id)
                .values(run_id=run_id)
            )
            return _row_to_attempt(await session.get(TaskAttemptRow, attempt_id))

        return await self._in_session(do)

    # ----------------------------------------------------------- helpers --

    async def _require_claimed(
        self, session: AsyncSession, claim: TaskClaim
    ) -> TaskRow:
        row = await session.get(TaskRow, claim.task_id)
        if row is None:
            raise TaskNotFoundError(claim.task_id)
        if (
            # The owning worker may still commit a task moved to CANCELLING while
            # its handler ran -- it lands CANCELLED via the commit guard.
            row.status not in (TaskStatus.CLAIMED.value, TaskStatus.CANCELLING.value)
            or row.lease_owner != claim.worker_id
            or row.active_attempt_id != claim.attempt_id
            or row.fencing_token != claim.fencing_token
        ):
            raise TaskClaimLostError(claim.task_id)
        return row

    async def _fenced_update(
        self,
        session: AsyncSession,
        claim: TaskClaim,
        *,
        status: "str | None" = None,
        extra_values: dict,
        now: datetime,
    ) -> TaskRow:
        values = dict(extra_values)
        values["lease_owner"] = None
        values["lease_expires_at"] = None
        values["active_attempt_id"] = None
        values["version"] = TaskRow.version + 1
        values["updated_at"] = now
        if status is not None:
            values["status"] = status
        result = await session.execute(
            update(TaskRow)
            .where(TaskRow.id == claim.task_id)
            .where(TaskRow.status == TaskStatus.CLAIMED.value)
            .where(TaskRow.lease_owner == claim.worker_id)
            .where(TaskRow.active_attempt_id == claim.attempt_id)
            .where(TaskRow.fencing_token == claim.fencing_token)
            .values(**values)
        )
        if result.rowcount != 1:
            raise TaskClaimLostError(claim.task_id)
        return await session.get(TaskRow, claim.task_id)

    async def _maybe_complete_job(
        self, session: AsyncSession, job_id: str, now: datetime
    ) -> None:
        job_row = await session.get(TaskJobRow, job_id)
        if job_row is None or job_row.status not in (
            JobStatus.RUNNING.value,
            JobStatus.WAITING.value,
            JobStatus.CANCELLING.value,
        ):
            return
        tasks = (
            (await session.execute(select(TaskRow).where(TaskRow.job_id == job_id)))
            .scalars()
            .all()
        )
        if not tasks:
            return
        statuses = {t.status for t in tasks}
        terminal = {
            TaskStatus.SUCCEEDED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        }
        active = statuses - terminal
        if active:
            #  a RUNNING job whose active tasks are all WAITING parks at
            # WAITING; a WAITING job whose task woke (now READY/CLAIMED) returns
            # to RUNNING. CANCELLING jobs stay CANCELLING until all-terminal.
            if (
                job_row.status == JobStatus.RUNNING.value
                and active <= {TaskStatus.WAITING.value}
            ):
                job_row.status = JobStatus.WAITING.value
                job_row.version += 1
            elif (
                job_row.status == JobStatus.WAITING.value
                and not active <= {TaskStatus.WAITING.value}
            ):
                job_row.status = JobStatus.RUNNING.value
                job_row.version += 1
            return
        if job_row.status == JobStatus.CANCELLING.value:
            target = JobStatus.CANCELLED
        elif statuses == {TaskStatus.SUCCEEDED.value}:
            target = JobStatus.SUCCEEDED
        elif TaskStatus.FAILED.value in statuses:
            target = JobStatus.FAILED
        else:
            target = JobStatus.CANCELLED
        steps = (
            [JobStatus.RUNNING, target]
            if job_row.status == JobStatus.WAITING.value
            else [target]
        )
        current = JobStatus(job_row.status)
        for nxt in steps:
            if nxt == current or nxt.value not in JOB_TRANSITIONS.get(
                current, frozenset()
            ):
                break
            job_row.status = nxt.value
            if nxt in JOB_TERMINAL:
                job_row.finished_at = now
                if job_row.started_at is None:
                    job_row.started_at = now
            current = nxt
        job_row.version += 1

    async def _reconcile_signals_sql(
        self, session: AsyncSession, now: datetime
    ) -> None:
        """Reconcile unconsumed signals against WAITING tasks: a crash between
        submit_signal's save and its task-wake can leave a saved signal whose
        matching WAITING task was never woken. Re-match them so no task is stuck
        WAITING. One task consumes one signal."""
        unconsumed = (
            (
                await session.execute(
                    select(TaskSignalRow).where(
                        TaskSignalRow.consumed_by_task_id.is_(None)
                    )
                )
            )
            .scalars()
            .all()
        )
        for sig in unconsumed:
            waiting = (
                (
                    await session.execute(
                        select(TaskRow)
                        .where(TaskRow.job_id == sig.job_id)
                        .where(TaskRow.status == TaskStatus.WAITING.value)
                    )
                )
                .scalars()
                .all()
            )
            for t in waiting:
                env = json.loads(t.data_json)
                wait_on = env.get("metadata", {}).get("wait_on", [])
                if not any(
                    w["name"] == sig.name and w["correlation_key"] == sig.correlation_key
                    for w in wait_on
                ):
                    continue
                from_status = t.status
                t.status = TaskStatus.READY.value
                t.available_at = now
                t.updated_at = now
                t.version += 1
                sig.consumed_by_task_id = t.id
                session.add(
                    self._transition(
                        sig.job_id,
                        task_id=t.id,
                        attempt_id=None,
                        from_status=from_status,
                        to_status=TaskStatus.READY.value,
                        reason="signal_reconcile",
                        now=now,
                    )
                )
                break  # one task consumes one signal

    async def _maybe_complete_job_all(
        self, session: AsyncSession, now: datetime
    ) -> None:
        job_ids = (await session.execute(select(TaskJobRow.id))).scalars().all()
        for job_id in job_ids:
            await self._maybe_complete_job(session, job_id, now)

    async def _apply_create_task_sql(
        self,
        session: AsyncSession,
        job_id: str,
        parent: TaskRow,
        cmd: CreateTask,
        now: datetime,
        job: JobRecord,
    ) -> None:
        import uuid as _uuid

        from ...task.models import narrow_child_principal
        from ...task.validation import validate_create_task, validate_task_policies

        #  enforce the same per-command input limits as the file store;
        #  a NON_IDEMPOTENT child must cap retries at 1.
        validate_create_task(cmd.handler, cmd.key, dict(cmd.metadata))
        validate_task_policies(cmd.retry_policy, cmd.side_effect_policy)
        # Mirror the file store's in-memory uniqueness check so a duplicate key
        # raises a clean ValueError instead of an IntegrityError that would burn
        # the worker's commit-retries and strand the parent in CLAIMED.
        existing = await session.execute(
            select(TaskRow)
            .where(TaskRow.job_id == job_id)
            .where(TaskRow.key == cmd.key)
        )
        if existing.scalars().first() is not None:
            raise ValueError(f"duplicate task key {cmd.key!r} in job {job_id}")
        parent_rec = _row_to_task(parent)
        child_depth = parent_rec.depth + 1
        # Budget guardrail (max_depth): a child beyond the depth cap is not
        # created -- the parent still succeeds, recursion stays bounded.
        if job.budget.max_depth is not None and child_depth > job.budget.max_depth:
            return
        child_scopes, child_chain = narrow_child_principal(
            parent_rec, cmd.delegated_scopes, cmd.handler, job.actor_chain
        )
        child_id = f"{parent.id}-{cmd.key}-{_uuid.uuid4().hex[:8]}"
        env = {
            "dependencies": list(cmd.dependencies),
            "retry_policy": to_jsonable(cmd.retry_policy),
            "side_effect_policy": to_jsonable(cmd.side_effect_policy),
            "resource_snapshots": to_jsonable(parent_rec.resource_snapshots),
            "depth": child_depth,
            "delegated_scopes": (
                list(child_scopes) if child_scopes is not None else None
            ),
            "actor_chain": to_jsonable(child_chain),
            "metadata": dict(cmd.metadata),
        }
        session.add(
            TaskRow(
                id=child_id,
                job_id=job_id,
                parent_task_id=parent.id,
                key=cmd.key,
                handler=cmd.handler,
                status=(
                    TaskStatus.PENDING.value
                    if cmd.dependencies
                    else TaskStatus.READY.value
                ),
                input_artifact_id=(
                    cmd.input_artifact.id if cmd.input_artifact else None
                ),
                output_artifact_id=None,
                attempt_count=0,
                available_at=now,
                lease_owner=None,
                lease_expires_at=None,
                fencing_token=0,
                active_attempt_id=None,
                timeout_seconds=cmd.timeout_seconds,
                version=1,
                created_at=now,
                updated_at=now,
                data_json=json.dumps(env),
            )
        )
        session.add(
            self._transition(
                job_id,
                task_id=child_id,
                attempt_id=None,
                from_status=None,
                to_status=(
                    TaskStatus.PENDING.value
                    if cmd.dependencies
                    else TaskStatus.READY.value
                ),
                reason="created",
                now=now,
            )
        )

    async def _resolve_dependencies_sql(
        self,
        session: AsyncSession,
        job_id: str,
        now: datetime,
    ) -> None:
        pending = (
            (
                await session.execute(
                    select(TaskRow)
                    .where(TaskRow.job_id == job_id)
                    .where(TaskRow.status == TaskStatus.PENDING.value)
                )
            )
            .scalars()
            .all()
        )
        for t in pending:
            env = json.loads(t.data_json)
            deps = env.get("dependencies", [])
            if not deps:
                continue
            deps_ok = True
            for dep_id in deps:
                dep = await session.get(TaskRow, dep_id)
                if dep is None or dep.status != TaskStatus.SUCCEEDED.value:
                    deps_ok = False
                    break
            if deps_ok:
                t.status = TaskStatus.READY.value
                t.updated_at = now
                t.version += 1
                session.add(
                    self._transition(
                        job_id,
                        task_id=t.id,
                        attempt_id=None,
                        from_status=TaskStatus.PENDING.value,
                        to_status=TaskStatus.READY.value,
                        reason="deps_satisfied",
                        now=now,
                    )
                )

    async def _cancel_job_sql(
        self,
        session: AsyncSession,
        job_id: str,
        now: datetime,
    ) -> None:
        job_row = await session.get(TaskJobRow, job_id)
        if job_row is None:
            return
        if job_row.status in (
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            # Already terminal: a CancelJob command must not resurrect it.
            return
        if job_row.status != JobStatus.CANCELLING.value:
            job_row.status = JobStatus.CANCELLING.value
            job_row.version += 1
        tasks = (
            (await session.execute(select(TaskRow).where(TaskRow.job_id == job_id)))
            .scalars()
            .all()
        )
        for t in tasks:
            if t.status in (
                TaskStatus.PENDING.value,
                TaskStatus.READY.value,
                TaskStatus.WAITING.value,
                TaskStatus.RETRY_WAIT.value,
            ):
                t.status = TaskStatus.CANCELLED.value
                t.updated_at = now
                t.version += 1
        if not any(t.status == TaskStatus.CLAIMED.value for t in tasks):
            job_row.status = JobStatus.CANCELLED.value
            job_row.finished_at = now


def _with_status(task: TaskRecord, status: TaskStatus) -> TaskRecord:
    from dataclasses import replace

    return replace(task, status=status)


def _task_to_row(task: TaskRecord) -> TaskRow:
    return TaskRow(
        id=task.id,
        job_id=task.job_id,
        parent_task_id=task.parent_task_id,
        key=task.key,
        handler=task.handler,
        status=task.status.value,
        input_artifact_id=task.input_artifact_id,
        output_artifact_id=task.output_artifact_id,
        attempt_count=task.attempt_count,
        available_at=_store_dt(task.available_at),
        lease_owner=task.lease_owner,
        lease_expires_at=_store_dt(task.lease_expires_at),
        fencing_token=task.fencing_token,
        active_attempt_id=task.active_attempt_id,
        timeout_seconds=task.timeout_seconds,
        version=task.version,
        created_at=_store_dt(task.created_at),
        updated_at=_store_dt(task.updated_at),
        data_json=_task_envelope(task),
    )


def _attempt_to_row(attempt: TaskAttemptRecord) -> TaskAttemptRow:
    return TaskAttemptRow(
        id=attempt.id,
        task_id=attempt.task_id,
        job_id=attempt.job_id,
        attempt=attempt.attempt,
        worker_id=attempt.worker_id,
        fencing_token=attempt.fencing_token,
        status=attempt.status.value,
        run_id=attempt.run_id,
        started_at=_store_dt(attempt.started_at),
        finished_at=_store_dt(attempt.finished_at),
        failure_kind=attempt.failure_kind.value if attempt.failure_kind else None,
        error_type=attempt.error_type,
        error_message=attempt.error_message,
        data_json=json.dumps({"metadata": dict(attempt.metadata)}),
    )


def _retry_delay(policy: RetryPolicy, attempt_number: int) -> float:
    import random

    if attempt_number <= 1:
        base = policy.initial_delay_seconds
    else:
        base = policy.initial_delay_seconds * (
            policy.multiplier ** (attempt_number - 1)
        )
    base = min(base, policy.max_delay_seconds)
    if policy.jitter_ratio > 0:
        base *= 1 + random.uniform(-policy.jitter_ratio, policy.jitter_ratio)
    return max(0.0, base)


__all__: "list[str]" = ["SqlAlchemyTaskStore"]
