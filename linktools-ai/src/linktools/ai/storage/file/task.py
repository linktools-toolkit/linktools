#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileTaskStore: single-process file backend for TaskStore.

One JSON file per record under ``root/jobs/{job_id}/{job,tasks,attempts,
transitions,signals}/...``. Each public async method delegates to a ``_*_sync``
method via ``asyncio.to_thread`` while holding an ``asyncio.Lock``; the lock
spans the ``to_thread`` call so the claim / fencing / transition invariants
hold within one process (a SQLAlchemy backend provides the same guarantees
across processes via CAS + transactions).

Fencing: every mutating write after a claim re-checks the stored task's
``status == CLAIMED``, ``lease_owner``, ``active_attempt_id`` and
``fencing_token`` against the claim, and rejects (raising
:class:`TaskClaimLostError`) if any differ -- so a worker whose lease expired
and was reclaimed cannot overwrite the new owner's result.
"""

import asyncio
import dataclasses
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ...task.models import (
    AttemptStatus,
    CLAIMABLE_JOB_STATUSES,
    JOB_TRANSITIONS,
    JOB_TERMINAL,
    JobRecord,
    JobStatus,
    SideEffectMode,
    TASK_TERMINAL,
    TaskAttemptRecord,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskTransitionRecord,
    TaskWaitCondition,
    TaskFailureKind,
    assert_attempt_transition,
    assert_job_transition,
    assert_task_transition,
    from_jsonable,
    narrow_child_principal,
    to_jsonable,
)
from ...security.principal import ScopeSet


class FileTaskCommitJournal:
    """Durable outcome journal used by :class:`FileTaskStore`.

    The store owns the state-machine replay; this small type centralizes the
    on-disk JSON step journal so recovery is explicit and inspectable.
    explicit journal boundary.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, attempt_id: str) -> Path:
        return self.root / f"{attempt_id}.json"

    def mark_step(self, path: Path, step: str) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["step"] = step
        _atomic_write(path, json.dumps(raw, sort_keys=True).encode("utf-8"))
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
    InvalidTaskCommandError,
    JobNotFoundError,
    RunnableBindingError,
    TaskBudgetExceededError,
    TaskClaim,
    TaskClaimLostError,
    TaskNotFoundError,
)
from ._util import _atomic_write, _validate_id_segment


class FileTaskStore:
    def __init__(self, root: Path, *, clock: "Clock | None" = None) -> None:
        self._root = Path(root)
        self._journal = FileTaskCommitJournal(self._root / "commit-journal")
        self._lock = asyncio.Lock()
        self._clock = clock or SystemClock()

    # ----------------------------------------------------------- paths --

    def _job_dir(self, job_id: str) -> Path:
        return self._root / "jobs" / _validate_id_segment(job_id, kind="job_id")

    def _job_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _task_path(self, job_id: str, task_id: str) -> Path:
        return (
            self._job_dir(job_id)
            / "tasks"
            / f"{_validate_id_segment(task_id, kind='task_id')}.json"
        )

    def _attempts_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "attempts"

    def _attempt_path(self, job_id: str, attempt_id: str) -> Path:
        return (
            self._attempts_dir(job_id)
            / f"{_validate_id_segment(attempt_id, kind='attempt_id')}.json"
        )

    def _transitions_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "transitions"

    def _signal_path(self, job_id: str, signal_id: str) -> Path:
        return (
            self._job_dir(job_id)
            / "signals"
            / f"{_validate_id_segment(signal_id, kind='signal_id')}.json"
        )

    # --------------------------------------------------------- file I/O --

    def _read(self, path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, record: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(to_jsonable(record)).encode("utf-8"))

    def _read_job(self, job_id: str) -> JobRecord:
        path = self._job_path(job_id)
        if not path.exists():
            raise JobNotFoundError(job_id)
        return from_jsonable(JobRecord, self._read(path))  # type: ignore[return-value]

    def _read_task(self, job_id: str, task_id: str) -> TaskRecord:
        path = self._task_path(job_id, task_id)
        if not path.exists():
            raise TaskNotFoundError(task_id)
        raw = self._read(path)
        if isinstance(raw, dict) and raw.get("delegated_scopes") is None:
            raw["delegated_scopes"] = to_jsonable(ScopeSet.empty())
            raw["metadata"] = {**raw.get("metadata", {}), "_legacy_missing_scopes": True}
        return from_jsonable(TaskRecord, raw)  # type: ignore[return-value]

    def _all_task_files(self):
        jobs = self._root / "jobs"
        if not jobs.exists():
            return
        for job_dir in sorted(jobs.iterdir()):
            tasks_dir = job_dir / "tasks"
            if not tasks_dir.exists():
                continue
            for task_file in sorted(tasks_dir.iterdir()):
                yield job_dir.name, task_file

    def _append_transition(
        self, job_id: str, *, task_id, attempt_id, from_status, to_status, reason, now
    ) -> None:
        seq = (
            len(list(self._transitions_dir(job_id).iterdir()))
            if self._transitions_dir(job_id).exists()
            else 0
        )
        tid = f"{seq:010d}"
        record = TaskTransitionRecord(
            id=tid,
            job_id=job_id,
            task_id=task_id,
            attempt_id=attempt_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            occurred_at=now,
        )
        self._write(self._transitions_dir(job_id) / f"{tid}.json", record)

    # --------------------------------------------------------- public API --
    # Each holds the process lock across a to_thread sync call.

    async def create_job(self, job: JobRecord, root_task: TaskRecord) -> JobRecord:
        async with self._lock:
            return await asyncio.to_thread(self._create_job_sync, job, root_task)

    async def get_job(self, job_id: str) -> "JobRecord | None":
        async with self._lock:
            return await asyncio.to_thread(self._get_job_sync, job_id)

    async def get_task(self, task_id: str) -> "TaskRecord | None":
        async with self._lock:
            return await asyncio.to_thread(self._get_task_sync, task_id)

    async def list_tasks(
        self, job_id: str, *, status: "TaskStatus | None" = None
    ) -> "tuple[TaskRecord, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._list_tasks_sync, job_id, status)

    async def claim(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        handlers: "tuple[str, ...] | None" = None,
    ) -> "ClaimedTask | None":
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_sync, worker_id, now, lease_seconds, handlers
            )

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
        async with self._lock:
            journal = self._journal.path(claim.attempt_id)
            await asyncio.to_thread(self._write_journal, journal, (claim, outcome, "success"))
            result = await asyncio.to_thread(self._commit_success_sync, claim, outcome)
            self._journal.mark_step(journal, "COMMITTED")
            journal.unlink(missing_ok=True)
            return result

    async def commit_failure(
        self, claim: TaskClaim, outcome: TaskFailure
    ) -> TaskRecord:
        async with self._lock:
            journal = self._journal.path(claim.attempt_id)
            await asyncio.to_thread(self._write_journal, journal, (claim, outcome, "failure"))
            result = await asyncio.to_thread(self._commit_failure_sync, claim, outcome)
            self._journal.mark_step(journal, "COMMITTED")
            journal.unlink(missing_ok=True)
            return result

    @staticmethod
    def _write_journal(path: Path, payload) -> None:
        claim, outcome, kind = payload
        tmp = path.with_suffix(".tmp")
        record = {"schema_version": 1, "step": "PREPARED", "kind": kind,
                  "claim": to_jsonable(claim), "outcome": to_jsonable(outcome)}
        _atomic_write(tmp, json.dumps(record, sort_keys=True).encode("utf-8"))
        tmp.replace(path)

    async def recover_incomplete_commits(self) -> None:
        """Replay durable outcomes before expired-lease recovery."""
        async with self._lock:
            for path in sorted(self._journal.root.glob("*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                claim = from_jsonable(TaskClaim, raw["claim"])
                kind = raw["kind"]
                outcome_type = TaskFailure if kind == "failure" else TaskSuccess
                outcome = from_jsonable(outcome_type, raw["outcome"])
                try:
                    if kind == "success":
                        await asyncio.to_thread(self._commit_success_sync, claim, outcome)
                    else:
                        await asyncio.to_thread(self._commit_failure_sync, claim, outcome)
                    self._journal.mark_step(path, "COMMITTED")
                    path.unlink(missing_ok=True)
                except TaskClaimLostError:
                    # Preserve the journal: a later lease owner may complete it.
                    continue

    async def request_cancel(
        self, job_id: str, *, reason: "str | None" = None
    ) -> JobRecord:
        async with self._lock:
            return await asyncio.to_thread(self._request_cancel_sync, job_id, reason)

    async def submit_signal(self, signal) -> object:
        from ...task.validation import validate_metadata

        validate_metadata(dict(signal.metadata))
        async with self._lock:
            return await asyncio.to_thread(self._submit_signal_sync, signal)

    async def recover_expired(
        self, *, now: datetime, limit: int = 100
    ) -> "tuple[TaskRecord, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._recover_expired_sync, now, limit)

    async def reconcile_due(
        self, *, now: datetime, limit: int = 100
    ) -> "tuple[TaskRecord, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._reconcile_due_sync, now, limit)

    async def list_orphan_run_ids(self, *, limit: int = 500) -> "tuple[str, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._list_orphan_run_ids_sync, limit)

    async def list_attempts(self, task_id: str) -> "tuple[TaskAttemptRecord, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._list_attempts_sync, task_id)

    async def list_transitions(self, job_id: str) -> "tuple[TaskTransitionRecord, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._list_transitions_sync, job_id)

    async def renew_lease(self, **kwargs) -> TaskRecord:
        async with self._lock:
            return await asyncio.to_thread(self._renew_lease_sync, **kwargs)

    async def bind_run(self, **kwargs) -> TaskAttemptRecord:
        async with self._lock:
            return await asyncio.to_thread(self._bind_run_sync, **kwargs)

    async def bind_runnable(self, **kwargs) -> TaskRecord:
        async with self._lock:
            return await asyncio.to_thread(self._bind_runnable_sync, **kwargs)

    # ----------------------------------------------------------- sync impl --

    def _create_job_sync(self, job: JobRecord, root_task: TaskRecord) -> JobRecord:
        from ...task.validation import validate_job_budget

        # The store re-validates the budget so the domain invariant holds even
        # for callers that bypass TaskRuntime.
        validate_job_budget(job.budget)
        if self._job_path(job.id).exists():
            raise FileExistsError(job.id)
        now = job.created_at
        self._write(self._job_path(job.id), job)
        # Root task is created READY (dependencies satisfied) so it is claimable.
        # Resolve its effective delegated scopes from the job's actor chain at
        # creation: a root with no explicit scopes inherits the job's scopes
        # (None = unrestricted). Persisting the resolved value means None at the
        # TaskRecord level has the single "unrestricted" meaning, never
        # "unresolved/inherit".
        from ...task.models import resolve_effective_scopes

        root = dataclasses.replace(
            root_task,
            job_id=job.id,
            status=TaskStatus.READY,
            delegated_scopes=resolve_effective_scopes(
                root_task.delegated_scopes, job.actor_chain.delegated_scopes
            ),
        )
        self._write(self._task_path(job.id, root.id), root)
        self._append_transition(
            job.id,
            task_id=root.id,
            attempt_id=None,
            from_status=None,
            to_status=TaskStatus.READY.value,
            reason="created",
            now=now,
        )
        return job

    def _get_job_sync(self, job_id: str) -> "JobRecord | None":
        try:
            return self._read_job(job_id)
        except JobNotFoundError:
            return None

    def _find_task_owner_job(self, task_id: str) -> "tuple[str, TaskRecord] | None":
        for job_id, task_file in self._all_task_files():
            if task_file.stem == task_id:
                return job_id, from_jsonable(TaskRecord, self._read(task_file))  # type: ignore[return-value]
        return None

    def _get_task_sync(self, task_id: str) -> "TaskRecord | None":
        found = self._find_task_owner_job(task_id)
        return found[1] if found else None

    def _list_tasks_sync(self, job_id: str, status) -> "tuple[TaskRecord, ...]":
        tasks_dir = self._job_dir(job_id) / "tasks"
        if not tasks_dir.exists():
            return ()
        out = []
        for f in sorted(tasks_dir.iterdir()):
            task = from_jsonable(TaskRecord, self._read(f))  # type: ignore[assignment]
            if status is None or task.status == status:
                out.append(task)
        return tuple(out)

    def _claim_sync(
        self, worker_id, now, lease_seconds, handlers
    ) -> "ClaimedTask | None":
        # Pick the earliest-due claimable task: READY, or RETRY_WAIT whose
        # available_at has passed (promoted to READY on claim).
        best = None
        for job_id, task_file in self._all_task_files():
            task: TaskRecord = from_jsonable(TaskRecord, self._read(task_file))  # type: ignore[assignment]
            if handlers is not None and task.handler not in handlers:
                continue
            claimable = task.status == TaskStatus.READY
            if task.status == TaskStatus.RETRY_WAIT and task.available_at <= now:
                claimable = True
            if not claimable or task.available_at > now:
                continue
            job = self._read_job(job_id)
            # Whitelist claimable job statuses: a terminal (SUCCEEDED/FAILED/
            # CANCELLED) or CANCELLING job must never produce a new attempt, and
            # a future JobStatus is never claimable by default.
            if job.status not in CLAIMABLE_JOB_STATUSES:
                continue
            # Aggregate budget exhaustion (attempts / runtime) is NOT a reason
            # to skip a candidate and leave it as a permanently-unclaimable
            # READY zombie -- it finalizes the whole job here and we move on.
            if self._job_budget_exhausted_sync(job_id, job, now):
                continue
            key = (task.available_at, task.created_at)
            if best is None or key < best[0]:
                best = (key, job_id, task)
        if best is None:
            return None
        _, job_id, task = best
        job = self._read_job(job_id)
        new_fencing = task.fencing_token + 1
        attempt_id = f"{task.id}-att{task.attempt_count + 1}"
        # Transition via READY if it was RETRY_WAIT. Record the promotion as its
        # own legal transition so the audit log never carries an illegal edge.
        was_retry = task.status == TaskStatus.RETRY_WAIT
        pre_status = task.status.value
        if was_retry:
            assert_task_transition(task.status, TaskStatus.READY)
            task = dataclasses.replace(task, status=TaskStatus.READY)
        assert_task_transition(task.status, TaskStatus.CLAIMED)
        claimed = dataclasses.replace(
            task,
            status=TaskStatus.CLAIMED,
            lease_owner=worker_id,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            fencing_token=new_fencing,
            attempt_count=task.attempt_count + 1,
            active_attempt_id=attempt_id,
            updated_at=now,
            version=task.version + 1,
        )
        attempt = TaskAttemptRecord(
            id=attempt_id,
            task_id=task.id,
            job_id=job_id,
            attempt=task.attempt_count + 1,
            worker_id=worker_id,
            fencing_token=new_fencing,
            status=AttemptStatus.RUNNING,
            started_at=now,
            run_id=None,
            finished_at=None,
            failure_kind=None,
            error_type=None,
            error_message=None,
        )
        # Write the attempt BEFORE the task: a crash between the two leaves an
        # orphan attempt (harmless) rather than a task whose active_attempt_id
        # points at a non-existent attempt (which would wedge recovery).
        self._write(self._attempt_path(job_id, attempt_id), attempt)
        self._write(self._task_path(job_id, task.id), claimed)
        if was_retry:
            self._append_transition(
                job_id,
                task_id=task.id,
                attempt_id=None,
                from_status=pre_status,
                to_status=TaskStatus.READY.value,
                reason="retry_due",
                now=now,
            )
        self._append_transition(
            job_id,
            task_id=task.id,
            attempt_id=attempt_id,
            from_status=TaskStatus.READY.value,
            to_status=TaskStatus.CLAIMED.value,
            reason="claimed",
            now=now,
        )
        # The first claim starts the job.
        if job.status == JobStatus.PENDING:
            assert_job_transition(job.status, JobStatus.RUNNING)
            job = dataclasses.replace(
                job, status=JobStatus.RUNNING, started_at=now, version=job.version + 1
            )
            self._write(self._job_path(job_id), job)
        return ClaimedTask(
            claim=TaskClaim(
                task_id=task.id,
                attempt_id=attempt_id,
                worker_id=worker_id,
                fencing_token=new_fencing,
            ),
            job=job,
            task=claimed,
            attempt=attempt,
        )

    def _guard(self, job_id: str, claim: TaskClaim) -> TaskRecord:
        task = self._read_task(job_id, claim.task_id)
        if (
            # The owning worker may still commit a task that was moved to
            # CANCELLING while its handler ran -- it lands CANCELLED.
            task.status not in (TaskStatus.CLAIMED, TaskStatus.CANCELLING)
            or task.lease_owner != claim.worker_id
            or task.active_attempt_id != claim.attempt_id
            or task.fencing_token != claim.fencing_token
        ):
            raise TaskClaimLostError(claim.task_id)
        return task

    def _job_budget_exhausted_sync(
        self, job_id: str, job: JobRecord, now: datetime
    ) -> bool:
        """If the job has exhausted its aggregate attempt or runtime budget,
        finalize every non-terminal task via the legal CANCELLING -> CANCELLED
        path and fail the job, so no READY task is left as a permanently-
        unclaimable zombie. A budget exhaustion is NOT a handler execution, so
        no new attempt is created. Returns True iff the job was finalized here.

        Tasks land CANCELLED (the only legal terminal edge from READY/RETRY_WAIT
        without a handler run); the job lands FAILED via CANCELLING -> FAILED
        and the transition ``reason`` records which budget was exceeded."""
        budget = job.budget
        tasks = self._list_tasks_sync(job_id, None)
        runtime_exceeded = (
            budget.max_runtime_seconds is not None
            and job.started_at is not None
            and (now - job.started_at).total_seconds() > budget.max_runtime_seconds
        )
        attempts_exceeded = (
            budget.max_attempts is not None
            and sum(t.attempt_count for t in tasks) >= budget.max_attempts
        )
        if not (runtime_exceeded or attempts_exceeded):
            return False
        reason = (
            "job_runtime_budget_exceeded"
            if runtime_exceeded
            else "job_attempt_budget_exceeded"
        )
        for t in tasks:
            if t.status in TASK_TERMINAL:
                continue
            pre = t.status
            att_id = t.active_attempt_id
            if pre != TaskStatus.CANCELLING:
                assert_task_transition(pre, TaskStatus.CANCELLING)
            assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
            cancelled = dataclasses.replace(
                t,
                status=TaskStatus.CANCELLED,
                lease_owner=None,
                lease_expires_at=None,
                active_attempt_id=None,
                updated_at=now,
                version=t.version + 1,
            )
            self._write(self._task_path(job_id, t.id), cancelled)
            # Close the now-orphaned active attempt: a CLAIMED task had a
            # RUNNING attempt that must not be left RUNNING forever.
            if att_id:
                apath = self._attempt_path(job_id, att_id)
                if apath.exists():
                    att = from_jsonable(
                        TaskAttemptRecord, self._read(apath)  # type: ignore[arg-type]
                    )
                    if att.status == AttemptStatus.RUNNING:
                        self._write(
                            apath,
                            dataclasses.replace(
                                att,
                                status=AttemptStatus.CANCELLED,
                                finished_at=now,
                            ),
                        )
            # Record one legal transition per actual step so every audit edge
            # is itself legal (never a direct READY -> CANCELLED, which the
            # task transition table forbids).
            if pre != TaskStatus.CANCELLING:
                self._append_transition(
                    job_id,
                    task_id=t.id,
                    attempt_id=att_id,
                    from_status=pre.value,
                    to_status=TaskStatus.CANCELLING.value,
                    reason=reason,
                    now=now,
                )
            self._append_transition(
                job_id,
                task_id=t.id,
                attempt_id=att_id,
                from_status=TaskStatus.CANCELLING.value,
                to_status=TaskStatus.CANCELLED.value,
                reason=reason,
                now=now,
            )
        # Job -> FAILED via legal edges only; record one transition per actual
        # step so every audit edge is itself legal (never a direct WAITING ->
        # FAILED, which the transition table forbids).
        pre_job = job.status
        if pre_job != JobStatus.CANCELLING:
            assert_job_transition(pre_job, JobStatus.CANCELLING)
        assert_job_transition(JobStatus.CANCELLING, JobStatus.FAILED)
        failed = dataclasses.replace(
            job,
            status=JobStatus.FAILED,
            finished_at=now,
            version=job.version + 1,
        )
        self._write(self._job_path(job_id), failed)
        if pre_job != JobStatus.CANCELLING:
            self._append_transition(
                job_id,
                task_id=None,
                attempt_id=None,
                from_status=pre_job.value,
                to_status=JobStatus.CANCELLING.value,
                reason=reason,
                now=now,
            )
        self._append_transition(
            job_id,
            task_id=None,
            attempt_id=None,
            from_status=JobStatus.CANCELLING.value,
            to_status=JobStatus.FAILED.value,
            reason=reason,
            now=now,
        )
        return True

    def _commit_success_sync(
        self, claim: TaskClaim, outcome: TaskSuccess
    ) -> TaskRecord:
        now = self._clock.now()
        owner = self._find_task_owner_job(claim.task_id)
        if owner is None:
            raise TaskNotFoundError(claim.task_id)
        job_id, _ = owner
        task = self._guard(job_id, claim)
        # Cancel precedence: if the task was moved to CANCELLING while
        # the handler ran, the handler's success is discarded and the task lands
        # CANCELLED (no commands applied).
        if task.status == TaskStatus.CANCELLING:
            return self._commit_cancelled_sync(job_id, claim, task, now)

        # Atomic budget pre-check for child creation: count every CreateTask
        # command ONCE against the job's live task total before any child is
        # written, so a budget breach fails the whole commit (all or none) and
        # never leaves the first two of three children created. Done before the
        # attempt/parent are flipped so the task is not marked successful on a
        # breach.
        create_commands = [
            c for c in outcome.commands if isinstance(c, CreateTask)
        ]
        if create_commands:
            job = self._read_job(job_id)
            self._assert_child_budget_sync(job_id, job, task, create_commands)

        # CompleteJob all-or-none gate: a handler that both creates children and
        # completes the job is contradictory -- the new
        # children would be live siblings. Reject this BEFORE any child is
        # written, so the commit is atomic (no partial creation followed by a
        # mid-loop CompleteJob failure that would wedge the task).
        if any(isinstance(c, CompleteJob) for c in outcome.commands):
            live_siblings = [
                t
                for t in self._list_tasks_sync(job_id, None)
                if t.id != task.id and t.status not in TASK_TERMINAL
            ]
            if create_commands or live_siblings:
                raise InvalidTaskCommandError(
                    "CompleteJob requires the committing task to be the only "
                    "non-terminal task; it cannot combine with CreateTask or "
                    "run alongside live siblings"
                )

        # Determine target: WaitSignal → WAITING (await external signal);
        # otherwise SUCCEEDED.
        has_wait = any(isinstance(c, WaitSignal) for c in outcome.commands)
        target = TaskStatus.WAITING if has_wait else TaskStatus.SUCCEEDED
        assert_task_transition(task.status, target)
        from_status = task.status.value
        new_task = dataclasses.replace(
            task,
            status=target,
            output_artifact_id=(
                outcome.output_artifact.id
                if outcome.output_artifact is not None
                else None
            ),
            lease_owner=None,
            lease_expires_at=None,
            active_attempt_id=None,
            updated_at=now,
            version=task.version + 1,
        )
        attempt = self._read_attempt(job_id, claim.attempt_id)
        assert_attempt_transition(attempt.status, AttemptStatus.SUCCEEDED)
        new_attempt = dataclasses.replace(
            attempt, status=AttemptStatus.SUCCEEDED, finished_at=now
        )
        # Persist WaitSignal conditions as first-class state (not metadata) so
        # submit_signal / reconcile_due can match them deterministically. v1
        # allows a single signal condition per task; a bounded wait carries its
        # deadline so reconcile_due can expire it.
        if has_wait:
            wait_signals = [c for c in outcome.commands if isinstance(c, WaitSignal)]
            if len(wait_signals) > 1:
                raise InvalidTaskCommandError(
                    "a task may wait for only one signal condition"
                )
            ws = wait_signals[0]
            deadline = (
                now + timedelta(seconds=ws.timeout_seconds)
                if ws.timeout_seconds is not None
                else None
            )
            new_task = dataclasses.replace(
                new_task,
                wait_conditions=(
                    TaskWaitCondition(name=ws.name, correlation_key=ws.correlation_key),
                ),
                wait_deadline_at=deadline,
            )
        #  the attempt + commands are persisted BEFORE the parent task is
        # flipped to its terminal status, so a crash never leaves the parent
        # succeeded while its child tasks are missing.
        self._write(self._attempt_path(job_id, claim.attempt_id), new_attempt)

        # Apply commands atomically.
        for cmd in outcome.commands:
            if isinstance(cmd, CreateTask):
                self._apply_create_task(job_id, task, cmd, now)
            elif isinstance(cmd, CompleteJob):
                self._complete_job_sync(job_id, claim.task_id, cmd, now)
            elif isinstance(cmd, CancelTask):
                self._cancel_task_in_job_sync(job_id, cmd, now)
            elif isinstance(cmd, CancelJob):
                # A handler CancelJob always targets the CURRENT job only --
                # the command carries no job_id, so cross-job cancellation is
                # not expressible here (use TaskRuntime.request_cancel).
                self._request_cancel_sync(job_id, cmd.reason)

        #  the parent task is flipped to its terminal status LAST, after
        # every child task / signal / dependency update has been persisted, so
        # the "parent succeeded but child not created" window never opens.
        self._write(self._task_path(job_id, task.id), new_task)
        self._append_transition(
            job_id,
            task_id=task.id,
            attempt_id=claim.attempt_id,
            from_status=from_status,
            to_status=target.value,
            reason="wait_signal" if has_wait else "succeeded",
            now=now,
        )
        # Resolve dependencies: PENDING → READY when all deps SUCCEEDED.
        self._resolve_dependencies(job_id, now)
        self._converge_jobs_sync(now)
        return new_task

    def _assert_child_budget_sync(
        self,
        job_id: str,
        job: JobRecord,
        parent: TaskRecord,
        create_commands: "list[CreateTask]",
    ) -> None:
        """Atomic child-budget gate: check the live task total + every CreateTask
        command at once, before any child is written,
        so a breach fails the whole commit (all-or-none) instead of leaving the
        first N children created. Depth and count breaches both raise."""
        max_depth = job.budget.max_depth
        child_depth = parent.depth + 1
        if max_depth is not None and child_depth > max_depth:
            raise TaskBudgetExceededError(
                f"task depth {child_depth} exceeds max_depth {max_depth}"
            )
        max_tasks = job.budget.max_tasks
        if max_tasks is not None:
            current = len(self._list_tasks_sync(job_id, None))
            if current + len(create_commands) > max_tasks:
                raise TaskBudgetExceededError(
                    f"job {job_id} task budget exhausted: "
                    f"{current}/{max_tasks}"
                )

    def _assert_job_can_complete_sync(
        self, current_task_id: str, tasks: "tuple[TaskRecord, ...]"
    ) -> None:
        """CompleteJob is only legal when the committing task is the sole
        non-terminal task in the job. A live sibling means the handler is
        trying to finish the job out from under running work -- reject it so the
        task stays claimed rather than landing SUCCEEDED prematurely."""
        live = [
            t
            for t in tasks
            if t.id != current_task_id and t.status not in TASK_TERMINAL
        ]
        if live:
            ids = ", ".join(t.id for t in live[:5])
            raise InvalidTaskCommandError(
                "CompleteJob requires all sibling tasks to be terminal; "
                f"non-terminal tasks: {ids}"
            )

    def _complete_job_sync(
        self, job_id: str, current_task_id: str, cmd: CompleteJob, now: datetime
    ) -> None:
        job = self._read_job(job_id)
        if job.status in JOB_TERMINAL:
            return  # already finished -- nothing CompleteJob can do
        tasks = self._list_tasks_sync(job_id, None)
        self._assert_job_can_complete_sync(current_task_id, tasks)
        output_artifact_id = cmd.output_artifact.id if cmd.output_artifact else None
        # Move to SUCCEEDED via legal edges only (a WAITING job two-steps
        # through RUNNING). finished_at + the recorded output land on the
        # terminal step.
        steps = (
            [JobStatus.RUNNING, JobStatus.SUCCEEDED]
            if job.status == JobStatus.WAITING
            else [JobStatus.SUCCEEDED]
        )
        current = job.status
        record = job
        for nxt in steps:
            if nxt == current:
                continue
            if nxt not in JOB_TRANSITIONS.get(current, frozenset()):
                break
            record = dataclasses.replace(
                record,
                status=nxt,
                finished_at=now if nxt in JOB_TERMINAL else record.finished_at,
                output_artifact_id=(
                    output_artifact_id
                    if nxt == JobStatus.SUCCEEDED
                    else record.output_artifact_id
                ),
                version=record.version + 1,
            )
            current = nxt
        if record.status != job.status:
            self._write(self._job_path(job_id), record)

    def _cancel_task_in_job_sync(
        self, job_id: str, cmd: CancelTask, now: datetime
    ) -> None:
        """CancelTask is scoped to the current job: a handler cannot name a task
        outside the job whose task produced this command. The file backend stores
        tasks under their job dir, so a foreign id is simply absent -- surfaced
        here as an explicit scope error rather than a silent no-op."""
        path = self._task_path(job_id, cmd.task_id)
        if not path.exists():
            raise InvalidTaskCommandError(
                f"CancelTask target {cmd.task_id!r} is not in job {job_id}"
            )
        ct = self._read_task(job_id, cmd.task_id)
        if ct.status in TASK_TERMINAL:
            return  # already terminal
        pre = ct.status
        assert_task_transition(pre, TaskStatus.CANCELLING)
        assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
        self._write(
            self._task_path(job_id, ct.id),
            dataclasses.replace(
                ct,
                status=TaskStatus.CANCELLED,
                updated_at=now,
                version=ct.version + 1,
            ),
        )
        # Record one legal transition per actual step (never a direct
        # READY -> CANCELLED), matching the budget-finalization audit shape.
        if pre != TaskStatus.CANCELLING:
            self._append_transition(
                job_id,
                task_id=ct.id,
                attempt_id=None,
                from_status=pre.value,
                to_status=TaskStatus.CANCELLING.value,
                reason="cancelled",
                now=now,
            )
        self._append_transition(
            job_id,
            task_id=ct.id,
            attempt_id=None,
            from_status=TaskStatus.CANCELLING.value,
            to_status=TaskStatus.CANCELLED.value,
            reason="cancelled",
            now=now,
        )

    def _apply_create_task(
        self, job_id: str, parent: TaskRecord, cmd: CreateTask, now: datetime
    ) -> None:
        from ...task.validation import validate_create_task, validate_task_policies

        validate_create_task(cmd.handler, cmd.key, dict(cmd.metadata))
        validate_task_policies(cmd.retry_policy, cmd.side_effect_policy)
        job = self._read_job(job_id)
        # Enforce key uniqueness within the job (matches the SQL UNIQUE(job_id,
        # key) index): a duplicate key is a handler bug and is rejected, not
        # silently duplicated.
        for existing in self._list_tasks_sync(job_id, None):
            if existing.key == cmd.key:
                raise ValueError(
                    f"duplicate task key {cmd.key!r} in job {job_id}"
                )
        child_depth = parent.depth + 1
        # Budget guardrail (max_depth): a child beyond the depth cap fails the
        # commit -- it is never silently dropped, because a silent drop lets the
        # parent succeed while its recursion quietly vanishes. (commit_success
        # checks this atomically up front; this is the defense-in-depth guard
        # for any direct caller.)
        if job.budget.max_depth is not None and child_depth > job.budget.max_depth:
            raise TaskBudgetExceededError(
                f"task depth {child_depth} exceeds max_depth {job.budget.max_depth}"
            )
        child_scopes, child_chain = narrow_child_principal(
            parent, cmd.delegated_scopes, cmd.handler, job.actor_chain
        )
        child_id = f"{parent.id}-{cmd.key}-{uuid.uuid4().hex[:8]}"
        child = TaskRecord(
            id=child_id,
            job_id=job_id,
            parent_task_id=parent.id,
            key=cmd.key,
            handler=cmd.handler,
            status=TaskStatus.PENDING if cmd.dependencies else TaskStatus.READY,
            input_artifact_id=cmd.input_artifact.id if cmd.input_artifact else None,
            output_artifact_id=None,
            dependencies=cmd.dependencies,
            retry_policy=cmd.retry_policy,
            side_effect_policy=cmd.side_effect_policy,
            attempt_count=0,
            available_at=now,
            lease_owner=None,
            lease_expires_at=None,
            fencing_token=0,
            active_attempt_id=None,
            timeout_seconds=cmd.timeout_seconds,
            resource_snapshots=parent.resource_snapshots,
            version=1,
            created_at=now,
            updated_at=now,
            depth=child_depth,
            delegated_scopes=child_scopes,
            actor_chain=child_chain,
            metadata=dict(cmd.metadata),
        )
        self._write(self._task_path(job_id, child_id), child)
        self._append_transition(
            job_id,
            task_id=child_id,
            attempt_id=None,
            from_status=None,
            to_status=child.status.value,
            reason="created",
            now=now,
        )

    def _resolve_dependencies(self, job_id: str, now: datetime) -> None:
        for t in self._list_tasks_sync(job_id, TaskStatus.PENDING):
            if not t.dependencies:
                continue
            deps_ok = True
            for dep_id in t.dependencies:
                try:
                    dep = self._read_task(job_id, dep_id)
                except TaskNotFoundError:
                    deps_ok = False
                    break
                if dep.status != TaskStatus.SUCCEEDED:
                    deps_ok = False
                    break
            if deps_ok:
                assert_task_transition(t.status, TaskStatus.READY)
                ready = dataclasses.replace(
                    t, status=TaskStatus.READY, updated_at=now, version=t.version + 1
                )
                self._write(self._task_path(job_id, t.id), ready)
                self._append_transition(
                    job_id,
                    task_id=t.id,
                    attempt_id=None,
                    from_status=t.status.value,
                    to_status=TaskStatus.READY.value,
                    reason="deps_satisfied",
                    now=now,
                )

    def _commit_failure_sync(
        self, claim: TaskClaim, outcome: TaskFailure
    ) -> TaskRecord:
        now = self._clock.now()
        owner = self._find_task_owner_job(claim.task_id)
        if owner is None:
            raise TaskNotFoundError(claim.task_id)
        job_id, _ = owner
        task = self._guard(job_id, claim)
        if task.status == TaskStatus.CANCELLING:
            # Cancel precedence: a CANCELLING task lands CANCELLED even
            # if the handler reported a failure.
            return self._commit_cancelled_sync(job_id, claim, task, now)
        attempt = self._read_attempt(job_id, claim.attempt_id)
        assert_attempt_transition(attempt.status, AttemptStatus.FAILED)
        new_attempt = dataclasses.replace(
            attempt,
            status=AttemptStatus.FAILED,
            finished_at=now,
            failure_kind=outcome.kind,
            error_type=outcome.error_type,
            error_message=outcome.message,
        )
        retryable = outcome.retryable
        if retryable is None:
            retryable = outcome.kind in task.retry_policy.retryable_kinds
        non_idempotent = task.side_effect_policy.mode == SideEffectMode.NON_IDEMPOTENT
        can_retry = (
            retryable
            and not non_idempotent
            and task.attempt_count < task.retry_policy.max_attempts
        )
        from_status = task.status.value
        if can_retry:
            delay = _retry_delay(task.retry_policy, task.attempt_count)
            assert_task_transition(task.status, TaskStatus.RETRY_WAIT)
            new_task = dataclasses.replace(
                task,
                status=TaskStatus.RETRY_WAIT,
                lease_owner=None,
                lease_expires_at=None,
                active_attempt_id=None,
                available_at=now + timedelta(seconds=delay),
                updated_at=now,
                version=task.version + 1,
            )
            to_status = TaskStatus.RETRY_WAIT.value
            reason = "retry"
        else:
            assert_task_transition(task.status, TaskStatus.FAILED)
            new_task = dataclasses.replace(
                task,
                status=TaskStatus.FAILED,
                lease_owner=None,
                lease_expires_at=None,
                active_attempt_id=None,
                updated_at=now,
                version=task.version + 1,
            )
            to_status = TaskStatus.FAILED.value
            reason = "failed"
        self._write(self._task_path(job_id, task.id), new_task)
        self._write(self._attempt_path(job_id, claim.attempt_id), new_attempt)
        self._append_transition(
            job_id,
            task_id=task.id,
            attempt_id=claim.attempt_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            now=now,
        )
        # A failed task may complete the job (all tasks terminal); converge.
        self._converge_jobs_sync(now)
        return new_task

    def _commit_cancelled_sync(
        self, job_id: str, claim: TaskClaim, task: TaskRecord, now: datetime
    ) -> TaskRecord:
        """Land a task that was CANCELLING when its handler stopped. Cancel takes
        precedence over the handler's outcome: the task goes CANCELLED,
        the attempt CANCELLED, and no commands are applied."""
        attempt = self._read_attempt(job_id, claim.attempt_id)
        assert_attempt_transition(attempt.status, AttemptStatus.CANCELLED)
        new_attempt = dataclasses.replace(
            attempt, status=AttemptStatus.CANCELLED, finished_at=now
        )
        assert_task_transition(task.status, TaskStatus.CANCELLED)
        new_task = dataclasses.replace(
            task,
            status=TaskStatus.CANCELLED,
            lease_owner=None,
            lease_expires_at=None,
            active_attempt_id=None,
            updated_at=now,
            version=task.version + 1,
        )
        self._write(self._attempt_path(job_id, claim.attempt_id), new_attempt)
        self._write(self._task_path(job_id, task.id), new_task)
        self._append_transition(
            job_id,
            task_id=task.id,
            attempt_id=claim.attempt_id,
            from_status=task.status.value,
            to_status=TaskStatus.CANCELLED.value,
            reason="cancelled",
            now=now,
        )
        self._converge_jobs_sync(now)
        return new_task

    def _finalize_cancelling_recover_sync(
        self, job_id: str, task: TaskRecord, now: datetime
    ) -> TaskRecord:
        """Recovery path for a CANCELLING task whose lease expired: its worker is
        gone and will never commit the cancel, so finalize the task as CANCELLED
        and close its stale attempt. Idempotent under re-recovery."""
        assert_task_transition(task.status, TaskStatus.CANCELLED)
        reset = dataclasses.replace(
            task,
            status=TaskStatus.CANCELLED,
            lease_owner=None,
            lease_expires_at=None,
            active_attempt_id=None,
            updated_at=now,
            version=task.version + 1,
        )
        self._write(self._task_path(job_id, task.id), reset)
        if task.active_attempt_id:
            apath = self._attempt_path(job_id, task.active_attempt_id)
            if apath.exists():
                att = from_jsonable(
                    TaskAttemptRecord, self._read(apath)  # type: ignore[arg-type]
                )
                if att.status == AttemptStatus.RUNNING:
                    self._write(
                        apath,
                        dataclasses.replace(
                            att, status=AttemptStatus.CANCELLED, finished_at=now
                        ),
                    )
        self._append_transition(
            job_id,
            task_id=task.id,
            attempt_id=task.active_attempt_id,
            from_status=task.status.value,
            to_status=TaskStatus.CANCELLED.value,
            reason="cancelled",
            now=now,
        )
        return reset

    def _request_cancel_sync(self, job_id: str, reason) -> JobRecord:
        now = self._clock.now()
        job = self._read_job(job_id)
        if job.status in (
            JobStatus.CANCELLED,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
        ):
            # Already terminal: cancelling a finished job is a no-op (the race
            # with a concurrent commit_success/commit_failure must not raise).
            return job
        if job.status not in (JobStatus.CANCELLING,):
            assert_job_transition(job.status, JobStatus.CANCELLING)
        job = dataclasses.replace(
            job,
            status=JobStatus.CANCELLING,
            finished_at=job.finished_at,
            version=job.version + 1,
        )
        self._write(self._job_path(job_id), job)
        for task in self._list_tasks_sync(job_id, None):
            if task.status in (
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.WAITING,
                TaskStatus.RETRY_WAIT,
            ):
                # Non-active tasks cancel via CANCELLING; both steps are
                # validated, the task lands CANCELLED.
                assert_task_transition(task.status, TaskStatus.CANCELLING)
                assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
                cancelled = dataclasses.replace(
                    task, status=TaskStatus.CANCELLED, updated_at=now
                )
                self._write(self._task_path(job_id, task.id), cancelled)
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.CANCELLED.value,
                    reason="cancelled",
                    now=now,
                )
            elif task.status == TaskStatus.CLAIMED:
                #  an in-flight task moves to CANCELLING so the owning
                # worker observes it on its next heartbeat poll and stops the
                # handler; the task lands CANCELLED when the handler commits.
                assert_task_transition(task.status, TaskStatus.CANCELLING)
                cancelling = dataclasses.replace(
                    task, status=TaskStatus.CANCELLING, updated_at=now
                )
                self._write(self._task_path(job_id, task.id), cancelling)
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.CANCELLING.value,
                    reason="cancelling",
                    now=now,
                )
        # The job cancels outright only when nothing is still in-flight: neither
        # CLAIMED nor mid-cancellation (CANCELLING). A CANCELLING task still
        # needs its worker to stop and commit before the job can complete.
        if not any(
            t.status in (TaskStatus.CLAIMED, TaskStatus.CANCELLING)
            for t in self._list_tasks_sync(job_id, None)
        ):
            assert_job_transition(job.status, JobStatus.CANCELLED)
            job = dataclasses.replace(job, status=JobStatus.CANCELLED, finished_at=now)
            self._write(self._job_path(job_id), job)
        return job

    def _submit_signal_sync(self, signal) -> object:
        # Idempotent on signal.id; resolves one matching WAITING task.
        now = signal.created_at
        job_id = signal.job_id
        path = self._signal_path(job_id, signal.id)
        if path.exists():
            return from_jsonable(type(signal), self._read(path))  # type: ignore[return-value]
        self._write(path, signal)
        for task in self._list_tasks_sync(job_id, TaskStatus.WAITING):
            # Match against the task's structured wait_conditions (not metadata):
            # only an explicit (name, correlation_key) it asked for wakes it.
            matched = any(
                c.name == signal.name and c.correlation_key == signal.correlation_key
                for c in task.wait_conditions
            )
            if not matched:
                continue
            assert_task_transition(task.status, TaskStatus.READY)
            woken = dataclasses.replace(
                task,
                status=TaskStatus.READY,
                available_at=now,
                wait_conditions=(),
                wait_deadline_at=None,
                updated_at=now,
                version=task.version + 1,
            )
            self._write(self._task_path(job_id, task.id), woken)
            self._append_transition(
                job_id,
                task_id=task.id,
                attempt_id=None,
                from_status=task.status.value,
                to_status=TaskStatus.READY.value,
                reason="signal",
                now=now,
            )
            # Record which task consumed this signal (the reverse link).
            signal = dataclasses.replace(signal, consumed_by_task_id=woken.id)
            self._write(path, signal)
            break
        # A woken task may move a WAITING job back to RUNNING.
        self._converge_jobs_sync(now)
        return signal

    def _reconcile_signals_sync(self, now: datetime) -> None:
        """Reconcile unconsumed signals against WAITING tasks. A crash between
        submit_signal's save and its task-wake can leave a saved signal whose
        matching WAITING task was never woken (or a signal that arrived before
        the task waited). Re-match them so no task is stuck WAITING."""
        for job_id, _task_file in self._all_task_files():
            signals_dir = self._job_dir(job_id) / "signals"
            if not signals_dir.exists():
                continue
            for sig_file in sorted(signals_dir.iterdir()):
                try:
                    signal = from_jsonable(TaskSignalRecord, self._read(sig_file))  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001 - skip unreadable signal files
                    continue
                if signal.consumed_by_task_id is not None:
                    continue  # already consumed
                for task in self._list_tasks_sync(job_id, TaskStatus.WAITING):
                    if not any(
                        c.name == signal.name
                        and c.correlation_key == signal.correlation_key
                        for c in task.wait_conditions
                    ):
                        continue
                    assert_task_transition(task.status, TaskStatus.READY)
                    woken = dataclasses.replace(
                        task,
                        status=TaskStatus.READY,
                        available_at=now,
                        wait_conditions=(),
                        wait_deadline_at=None,
                        updated_at=now,
                        version=task.version + 1,
                    )
                    self._write(self._task_path(job_id, task.id), woken)
                    self._append_transition(
                        job_id,
                        task_id=task.id,
                        attempt_id=None,
                        from_status=task.status.value,
                        to_status=TaskStatus.READY.value,
                        reason="signal_reconcile",
                        now=now,
                    )
                    signal = dataclasses.replace(signal, consumed_by_task_id=woken.id)
                    self._write(sig_file, signal)
                    break  # one task consumes one signal
        self._converge_jobs_sync(now)

    def _recover_expired_sync(self, now, limit) -> "tuple[TaskRecord, ...]":
        recovered = []
        count = 0
        for job_id, task_file in self._all_task_files():
            if count >= limit:
                break
            task: TaskRecord = from_jsonable(TaskRecord, self._read(task_file))  # type: ignore[assignment]
            expired = bool(task.lease_expires_at and task.lease_expires_at < now)
            if task.status == TaskStatus.CANCELLING and expired:
                #  a CANCELLING task whose lease expired will never be
                # committed by its (now-gone) worker -- finalize as CANCELLED.
                recovered.append(
                    self._finalize_cancelling_recover_sync(job_id, task, now)
                )
                count += 1
                continue
            if not (task.status == TaskStatus.CLAIMED and expired):
                continue
            non_idempotent = (
                task.side_effect_policy.mode == SideEffectMode.NON_IDEMPOTENT
            )
            exhausted = task.attempt_count >= task.retry_policy.max_attempts
            target = (
                TaskStatus.FAILED if (non_idempotent or exhausted) else TaskStatus.READY
            )
            # Reset the TASK first. A crash after this line still leaves the task
            # out of CLAIMED, so a later recovery pass skips it (idempotent).
            assert_task_transition(task.status, target)
            reset = dataclasses.replace(
                task,
                status=target,
                lease_owner=None,
                lease_expires_at=None,
                active_attempt_id=None,
                available_at=now,
                updated_at=now,
                version=task.version + 1,
            )
            self._write(self._task_path(job_id, task.id), reset)
            # Close the stale attempt as SUPERSEDED, tolerating a missing or
            # already-superseded attempt (crash windows can leave either).
            if task.active_attempt_id:
                apath = self._attempt_path(job_id, task.active_attempt_id)
                if apath.exists():
                    att = from_jsonable(TaskAttemptRecord, self._read(apath))  # type: ignore[assignment]
                    if att.status == AttemptStatus.RUNNING:
                        self._write(
                            apath,
                            dataclasses.replace(
                                att, status=AttemptStatus.SUPERSEDED, finished_at=now
                            ),
                        )
            if non_idempotent:
                reason = "non_idempotent"
            elif exhausted:
                reason = "attempts_exhausted"
            else:
                reason = "lease_expired"
            self._append_transition(
                job_id,
                task_id=task.id,
                attempt_id=task.active_attempt_id,
                from_status=TaskStatus.CLAIMED.value,
                to_status=target.value,
                reason=reason,
                now=now,
            )
            recovered.append(reset)
            count += 1
        # Reconcile unconsumed signals (crash between save and wake) so no task
        # is left stuck WAITING, then converge job state.
        self._reconcile_signals_sync(now)
        self._converge_jobs_sync(now)
        return tuple(recovered)

    def _reconcile_due_sync(self, now, limit) -> "tuple[TaskRecord, ...]":
        """Move WAITING tasks past their signal deadline to a retry or a
        terminal cancel. A deadline is not a handler run, so no new attempt is
        created. Retryable (TIMEOUT in the retry policy AND attempts remain):
        WAITING -> READY, paced by available_at. Otherwise WAITING ->
        CANCELLING -> CANCELLED (the only legal terminal edge from WAITING
        without a handler run; FAILED is unreachable without expanding the
        state machine)."""
        handled = []
        count = 0
        for job_id, task_file in self._all_task_files():
            if count >= limit:
                break
            task: TaskRecord = from_jsonable(TaskRecord, self._read(task_file))  # type: ignore[assignment]
            if task.status != TaskStatus.WAITING:
                continue
            if task.wait_deadline_at is None or task.wait_deadline_at > now:
                continue
            retryable = (
                TaskFailureKind.TIMEOUT in task.retry_policy.retryable_kinds
                and task.attempt_count < task.retry_policy.max_attempts
            )
            if retryable:
                delay = _retry_delay(task.retry_policy, task.attempt_count)
                assert_task_transition(task.status, TaskStatus.READY)
                reset = dataclasses.replace(
                    task,
                    status=TaskStatus.READY,
                    available_at=now + timedelta(seconds=delay),
                    wait_conditions=(),
                    wait_deadline_at=None,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._write(self._task_path(job_id, task.id), reset)
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.READY.value,
                    reason="signal_timeout_retry",
                    now=now,
                )
            else:
                assert_task_transition(task.status, TaskStatus.CANCELLING)
                assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
                reset = dataclasses.replace(
                    task,
                    status=TaskStatus.CANCELLED,
                    lease_owner=None,
                    lease_expires_at=None,
                    active_attempt_id=None,
                    wait_conditions=(),
                    wait_deadline_at=None,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._write(self._task_path(job_id, task.id), reset)
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.CANCELLING.value,
                    reason="signal_timeout",
                    now=now,
                )
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=TaskStatus.CANCELLING.value,
                    to_status=TaskStatus.CANCELLED.value,
                    reason="signal_timeout",
                    now=now,
                )
            handled.append(reset)
            count += 1
        self._converge_jobs_sync(now)
        return tuple(handled)

    def _converge_jobs_sync(self, now: datetime) -> None:
        jobs_dir = self._root / "jobs"
        if not jobs_dir.exists():
            return
        for job_dir in sorted(jobs_dir.iterdir()):
            job_id = job_dir.name
            jpath = self._job_path(job_id)
            if not jpath.exists():
                continue
            job = from_jsonable(JobRecord, self._read(jpath))  # type: ignore[assignment]
            if job.status in (
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                continue
            tasks = self._list_tasks_sync(job_id, None)
            if not tasks:
                continue
            statuses = {t.status for t in tasks}
            terminal = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            active = statuses - terminal
            if active:
                #  the job still has non-terminal tasks. A RUNNING job with
                # every active task WAITING (nothing READY/CLAIMED) parks at
                # WAITING; a WAITING job whose task woke (now READY/CLAIMED) goes
                # back to RUNNING. Terminal convergence is not pursued here.
                if job.status == JobStatus.RUNNING and active <= {TaskStatus.WAITING}:
                    self._write(
                        jpath, dataclasses.replace(job, status=JobStatus.WAITING)
                    )
                elif (
                    job.status == JobStatus.WAITING
                    and not active <= {TaskStatus.WAITING}
                ):
                    self._write(
                        jpath, dataclasses.replace(job, status=JobStatus.RUNNING)
                    )
                continue  # still has active tasks
            # Pick the terminal target from the task outcomes.
            if job.status == JobStatus.CANCELLING:
                target = JobStatus.CANCELLED
            elif statuses == {TaskStatus.SUCCEEDED}:
                target = JobStatus.SUCCEEDED
            elif TaskStatus.FAILED in statuses:
                target = JobStatus.FAILED
            else:
                target = JobStatus.CANCELLED  # all terminal, none failed/succeeded
            # Move via legal edges only -- a WAITING job two-steps through RUNNING.
            # Never raise out of convergence (recovery must stay crash-proof).
            steps = (
                [JobStatus.RUNNING, target]
                if job.status == JobStatus.WAITING
                else [target]
            )
            current = job.status
            record = job
            for nxt in steps:
                if nxt == current:
                    continue
                if nxt not in JOB_TRANSITIONS.get(current, frozenset()):
                    break  # illegal edge for this state -- leave the job as-is
                record = dataclasses.replace(
                    record,
                    status=nxt,
                    finished_at=now if nxt in JOB_TERMINAL else record.finished_at,
                    started_at=record.started_at or now,
                )
                current = nxt
            if record.status == job.status:
                continue
            self._write(jpath, record)

    def _list_attempts_sync(self, task_id: str) -> "tuple[TaskAttemptRecord, ...]":
        owner = self._find_task_owner_job(task_id)
        if owner is None:
            return ()
        job_id, _ = owner
        d = self._attempts_dir(job_id)
        if not d.exists():
            return ()
        return tuple(
            from_jsonable(TaskAttemptRecord, self._read(f))  # type: ignore[arg-type]
            for f in sorted(d.iterdir())
            if from_jsonable(TaskAttemptRecord, self._read(f)).task_id == task_id  # type: ignore[arg-type]
        )

    def _list_orphan_run_ids_sync(self, limit: int) -> "tuple[str, ...]":
        """All run_ids referenced by SUPERSEDED attempts (deduped), independent
        of the recovered-task list. This is the source reconcile uses so a
        retried startup pass still re-finds orphans from a failed pass
        (recover_expired itself is idempotent)."""
        seen: "set[str]" = set()
        out: "list[str]" = []
        for job_dir in sorted((self._root / "jobs").iterdir()) if (self._root / "jobs").exists() else []:
            attempts_dir = job_dir / "attempts"
            if not attempts_dir.exists():
                continue
            for f in sorted(attempts_dir.iterdir()):
                try:
                    att = from_jsonable(
                        TaskAttemptRecord, self._read(f)  # type: ignore[arg-type]
                    )
                except Exception:  # noqa: BLE001 - skip unreadable attempt files
                    continue
                if (
                    att.status == AttemptStatus.SUPERSEDED
                    and att.run_id
                    and att.run_id not in seen
                ):
                    seen.add(att.run_id)
                    out.append(att.run_id)
                    if len(out) >= limit:
                        return tuple(out)
        return tuple(out)

    def _list_transitions_sync(self, job_id: str) -> "tuple[TaskTransitionRecord, ...]":
        d = self._transitions_dir(job_id)
        if not d.exists():
            return ()
        return tuple(
            from_jsonable(TaskTransitionRecord, self._read(f))  # type: ignore[arg-type]
            for f in sorted(d.iterdir())
        )

    def _read_attempt(self, job_id: str, attempt_id: str) -> TaskAttemptRecord:
        path = self._attempt_path(job_id, attempt_id)
        if not path.exists():
            raise TaskNotFoundError(attempt_id)
        return from_jsonable(TaskAttemptRecord, self._read(path))  # type: ignore[return-value]

    def _renew_lease_sync(
        self, *, task_id, attempt_id, worker_id, fencing_token, now, lease_seconds
    ) -> TaskRecord:
        owner = self._find_task_owner_job(task_id)
        if owner is None:
            raise TaskNotFoundError(task_id)
        job_id, _ = owner
        claim = TaskClaim(
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
        )
        task = self._guard(job_id, claim)
        renewed = dataclasses.replace(
            task,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            updated_at=now,
            version=task.version + 1,
        )
        self._write(self._task_path(job_id, task_id), renewed)
        return renewed

    def _bind_run_sync(
        self, *, task_id, attempt_id, fencing_token, worker_id, run_id
    ) -> TaskAttemptRecord:
        owner = self._find_task_owner_job(task_id)
        if owner is None:
            raise TaskNotFoundError(task_id)
        job_id, _ = owner
        claim = TaskClaim(
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
        )
        self._guard(job_id, claim)
        att = self._read_attempt(job_id, attempt_id)
        bound = dataclasses.replace(att, run_id=run_id)
        self._write(self._attempt_path(job_id, attempt_id), bound)
        return bound

    def _bind_runnable_sync(
        self,
        *,
        task_id,
        attempt_id,
        fencing_token,
        worker_id,
        runnable_id,
        revision,
        fingerprint,
    ) -> TaskRecord:
        owner = self._find_task_owner_job(task_id)
        if owner is None:
            raise TaskNotFoundError(task_id)
        job_id, _ = owner
        claim = TaskClaim(
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
        )
        task = self._guard(job_id, claim)
        now = self._clock.now()
        if task.resolved_runnable_id is None:
            # First resolution: pin it.
            bound = dataclasses.replace(
                task,
                resolved_runnable_id=runnable_id,
                resolved_runnable_revision=revision,
                resolved_runnable_fingerprint=fingerprint,
                updated_at=now,
                version=task.version + 1,
            )
            self._write(self._task_path(job_id, task_id), bound)
            return bound
        if (
            task.resolved_runnable_id == runnable_id
            and task.resolved_runnable_revision == revision
            and task.resolved_runnable_fingerprint == fingerprint
        ):
            return task  # idempotent re-bind on a retry
        raise RunnableBindingError(
            f"runnable binding drift on task {task_id}: pinned "
            f"{task.resolved_runnable_id}/"
            f"{task.resolved_runnable_revision}/"
            f"{task.resolved_runnable_fingerprint} != resolved "
            f"{runnable_id}/{revision}/{fingerprint}"
        )


def _retry_delay(policy, attempt_number: int) -> float:
    if attempt_number <= 1:
        base = policy.initial_delay_seconds
    else:
        base = policy.initial_delay_seconds * (
            policy.multiplier ** (attempt_number - 1)
        )
    base = min(base, policy.max_delay_seconds)
    if policy.jitter_ratio > 0:
        # Exponential backoff with full jitter: the delay stays non-negative
        # and within the configured bounds.
        base *= 1 + random.uniform(-policy.jitter_ratio, policy.jitter_ratio)
    return max(0.0, base)


__all__: "list[str]" = ["FileTaskStore"]
