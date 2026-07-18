#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskRuntime: the reliable-task facade.

The runtime is the downstream-facing entry point. A caller registers handlers
by name, then ``run`` drives a worker loop that claims tasks, builds a
:class:`TaskContext`, runs the matching :class:`TaskHandler`, and commits the
outcome. The runtime does NOT execute agents itself -- it only schedules
``TaskHandler`` implementations, one of which (later phase) wraps the existing
``linktools.ai.Runtime``.

``ensure_recovered`` runs lease-expiry recovery once at startup under a lock
so a restarted process re-converges before accepting work.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .models import (
    ActorChain,
    ActorRef,
    JobRecord,
    JobStatus,
    RetryPolicy,
    SideEffectPolicy,
    TaskAttemptRecord,
    TaskBudget,
    TaskPrincipal,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskTransitionRecord,
)
from .protocols import Clock, SystemClock, TaskHandler
from .metrics import NoopTaskMetrics, TaskMetrics
from .store import TaskRunTimeoutError, TaskCancellationDidNotConvergeError, TaskStore


@dataclass(frozen=True, slots=True)
class TaskRuntimeOptions:
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    cancel_grace_seconds: float = 5.0
    poll_interval_seconds: float = 0.5
    max_concurrency: int = 1
    max_payload_bytes: int = 1024 * 1024
    max_commands_per_task: int = 100
    max_child_tasks_per_task: int = 100
    max_signal_payload_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("heartbeat_seconds must be < lease_seconds")
        if self.cancel_grace_seconds <= 0:
            raise ValueError("cancel_grace_seconds must be > 0")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        for cap in (
            self.max_payload_bytes,
            self.max_commands_per_task,
            self.max_child_tasks_per_task,
            self.max_signal_payload_bytes,
        ):
            if cap <= 0:
                raise ValueError("capacity limits must be > 0")


class TaskStoreRequiredError(Exception):
    """Raised when TaskRuntime is built without a Storage.tasks store."""


class TaskRuntime:
    def __init__(
        self,
        *,
        storage,
        handlers: "Mapping[str, TaskHandler]",
        options: "TaskRuntimeOptions | None" = None,
        clock: "Clock | None" = None,
        metrics: "TaskMetrics | None" = None,
        run_canceler: "Callable[[str], Awaitable[None]] | None" = None,
        artifact_store=None,
    ) -> None:
        task_store = getattr(storage, "tasks", None)
        if task_store is None:
            raise TaskStoreRequiredError(
                "TaskRuntime requires Storage.tasks; wire a FileTaskStore or "
                "SqlAlchemyTaskStore into Storage"
            )
        self._storage = storage
        self._task_store: TaskStore = task_store
        self._handlers: "dict[str, TaskHandler]" = dict(handlers)
        self._options = options or TaskRuntimeOptions()
        self._clock = clock or SystemClock()
        self._metrics = metrics or NoopTaskMetrics()
        # Best-effort canceler for Runs orphaned when a worker crashes mid-run.
        # TaskRuntime coordinates; the caller wires the actual Runtime.cancel so
        # the task domain stays decoupled from Runtime internals.
        self._run_canceler = run_canceler
        # The artifact store resolves task input artifacts. If the caller did
        # not wire one explicitly, build it from the storage resource backend
        # when available; a runtime with neither simply cannot serve tasks that
        # declare an input artifact.
        self._artifact_store = artifact_store
        if self._artifact_store is None:
            resources = getattr(storage, "resources", None)
            if resources is not None:
                from ..artifact.store import ArtifactStore

                self._artifact_store = ArtifactStore(resources)
        # Runtime-backed handlers require both durable stores in production;
        # reject a silently degraded registration at the orchestration boundary.
        for name, handler in self._handlers.items():
            if type(handler).__name__ == "RuntimeTaskHandler":
                if getattr(handler, "_task_store", None) is None or self._artifact_store is None:
                    raise ValueError(
                        f"handler {name!r} requires task_store and artifact_store"
                    )
        self._recovered = False
        self._recovery_lock = asyncio.Lock()

    @property
    def options(self) -> TaskRuntimeOptions:
        return self._options

    @property
    def clock(self) -> Clock:
        return self._clock

    # ---- job / task API (thin delegation to the store) ----

    async def create_job(self, job: JobRecord, root_task) -> JobRecord:
        from .validation import (
            validate_handler_name,
            validate_job_budget,
            validate_metadata,
            validate_task_key,
            validate_task_policies,
        )

        validate_handler_name(root_task.handler)
        validate_task_key(root_task.key)
        validate_metadata(dict(root_task.metadata))
        validate_job_budget(job.budget)
        #  a NON_IDEMPOTENT root task must cap retries at 1.
        validate_task_policies(root_task.retry_policy, root_task.side_effect_policy)
        record = await self._task_store.create_job(job, root_task)
        await self._metrics.inc_counter(
            "job_created_total", labels={"status": record.status.value}
        )
        return record

    async def get_job(self, job_id: str) -> "JobRecord | None":
        return await self._task_store.get_job(job_id)

    async def get_task(self, task_id: str):
        return await self._task_store.get_task(task_id)

    async def list_tasks(self, job_id: str, *, status: "TaskStatus | None" = None):
        return await self._task_store.list_tasks(job_id, status=status)

    async def list_attempts(self, task_id: str) -> "tuple[TaskAttemptRecord, ...]":
        return await self._task_store.list_attempts(task_id)

    async def list_transitions(self, job_id: str) -> "tuple[TaskTransitionRecord, ...]":
        return await self._task_store.list_transitions(job_id)

    async def request_cancel(
        self, job_id: str, *, reason: "str | None" = None
    ) -> JobRecord:
        return await self._task_store.request_cancel(job_id, reason=reason)

    async def submit_signal(self, signal: TaskSignalRecord) -> TaskSignalRecord:
        return await self._task_store.submit_signal(signal)

    async def ensure_recovered(self) -> None:
        """Run lease-expiry recovery once (lock + double-check), then reconcile
        Runs orphaned by the tasks that were reset.

        Uses a lock + double-check so a restarted process re-converges before
        accepting work. Reconciliation runs STRICT at startup: if it raises, the
        recovered flag stays false so the next call retries -- the startup
        reconciliation path must not silently swallow every error."""
        async with self._recovery_lock:
            if self._recovered:
                return
            recover_commits = getattr(self._task_store, "recover_incomplete_commits", None)
            if recover_commits is not None:
                await recover_commits()
            recovered = await self._task_store.recover_expired(
                now=self._clock.now(), limit=500
            )
            # Advance WAITING tasks past their signal deadline on restart too,
            # so a bounded wait does not park forever across a crash.
            _due = await self._task_store.reconcile_due(now=self._clock.now(), limit=500)
            await self._reconcile_orphan_runs(recovered, strict=True)
            self._recovered = True

    # ---- worker driving ----

    async def _reconcile_orphan_runs(
        self, _recovered: "Sequence[TaskRecord]", *, strict: bool = False
    ) -> None:
        """Cancel Runs orphaned by superseded attempts (a worker crash can leave
        a Run RUNNING whose attempt was superseded). ``_recovered`` is accepted
        for the RecoverHook signature but the orphan source is
        ``list_orphan_run_ids`` -- a store-wide scan of superseded attempts --
        so a RETRIED startup pass still re-finds and re-cancels orphans from a
        failed pass (``recover_expired`` is idempotent, so the recovered list is
        empty on retry and cannot be the sole source).

        ``strict`` selects the failure mode: startup (True) accumulates cancel
        errors and re-raises so ``ensure_recovered`` leaves the recovered flag
        false and the next call retries; the periodic worker hook (False) stays
        best-effort. ``run_canceler`` must be idempotent (canceling an
        already-terminal Run is a no-op), so re-scanning on retry is safe."""
        if self._run_canceler is None:
            return
        try:
            orphan_runs = await self._task_store.list_orphan_run_ids(limit=500)
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise RuntimeError(f"orphan-run scan failed: {exc}") from exc
            return
        errors: "list[BaseException]" = []
        for run_id in orphan_runs:
            try:
                await self._run_canceler(run_id)
            except Exception as exc:  # noqa: BLE001
                if strict:
                    errors.append(exc)
        if strict and errors:
            raise RuntimeError(
                f"startup orphan-run reconciliation failed ({len(errors)} error(s))"
            )

    async def run(
        self,
        *,
        worker_id: str,
        shutdown: "asyncio.Event | None" = None,
    ) -> None:
        from .worker import TaskWorker

        await self.ensure_recovered()
        worker = TaskWorker(
            task_store=self._task_store,
            handlers=self._handlers,
            options=self._options,
            clock=self._clock,
            metrics=self._metrics,
            on_recovered=self._reconcile_orphan_runs if self._run_canceler else None,
            artifact_store=self._artifact_store,
        )
        await worker.run(worker_id=worker_id, shutdown=shutdown)

    async def run_one_task(
        self,
        handler_name: str,
        *,
        tenant_id: str,
        user_id: "str | None" = None,
        input_artifact_id: "str | None" = None,
        metadata: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        retry_policy: "RetryPolicy | None" = None,
        wait_timeout: float = 60.0,
        worker_id: "str | None" = None,
    ):
        """Submit a single task and drive it to a terminal state, returning the
        final :class:`TaskRecord`. A one-shot convenience over create_job + run:
        it spins up a worker, polls the job to completion, then shuts down.
        Enables eval/task-mode executors to run one case reliably (with retries)
        without re-implementing the worker loop or importing task models.
        ``wait_timeout`` bounds how long to wait for the job to finish."""
        now = self._clock.now()
        job_id = f"oneshot-{uuid.uuid4().hex[:10]}"
        task_id = f"task-{uuid.uuid4().hex[:10]}"
        principal = TaskPrincipal(tenant_id=tenant_id, user_id=user_id)
        actor = ActorRef(kind="user", id=user_id or "oneshot")
        job = JobRecord(
            id=job_id,
            status=JobStatus.PENDING,
            principal=principal,
            actor_chain=ActorChain(actors=(actor,)),
            budget=TaskBudget(),
            root_task_id=task_id,
            input_artifact_id=input_artifact_id,
            output_artifact_id=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        task = TaskRecord(
            id=task_id,
            job_id=job_id,
            parent_task_id=None,
            key="oneshot",
            handler=handler_name,
            status=TaskStatus.PENDING,
            input_artifact_id=input_artifact_id,
            output_artifact_id=None,
            dependencies=(),
            retry_policy=retry_policy or RetryPolicy(),
            side_effect_policy=SideEffectPolicy(),
            attempt_count=0,
            available_at=now,
            lease_owner=None,
            lease_expires_at=None,
            fencing_token=0,
            active_attempt_id=None,
            timeout_seconds=timeout_seconds,
            resource_snapshots=(),
            version=1,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        await self.create_job(job, task)

        shutdown = asyncio.Event()
        wt = asyncio.create_task(
            self.run(worker_id=worker_id or f"oneshot-{job_id}", shutdown=shutdown)
        )
        terminal = (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)
        # Drive the wait through the Clock so a FakeClock advances virtual time
        # (no real sleep) and the polling cadence is testable.
        deadline = self._clock.now() + timedelta(seconds=wait_timeout)
        reached_terminal = False
        cancel_error = None
        try:
            while self._clock.now() < deadline:
                job_now = await self.get_job(job_id)
                if job_now is not None and job_now.status in terminal:
                    reached_terminal = True
                    break
                await self._clock.sleep(0.02)
        finally:
            # Business cancellation must reach the claimed task before the
            # worker is stopped; otherwise an in-flight handler can outlive
            # this one-shot call and continue side effects.
            if not reached_terminal:
                try:
                    await self.request_cancel(job_id, reason="run_one_task wait timeout")
                    cancel_deadline = self._clock.now() + timedelta(
                        seconds=self._options.cancel_grace_seconds
                    )
                    while self._clock.now() < cancel_deadline:
                        current = await self.get_job(job_id)
                        if current is not None and current.status in terminal:
                            break
                        await self._clock.sleep(0.02)
                except Exception as exc:  # cancellation failure remains explicit
                    cancel_error = exc
            shutdown.set()
            shutdown_deadline = self._clock.now() + timedelta(
                seconds=self._options.cancel_grace_seconds
            )
            while not wt.done() and self._clock.now() < shutdown_deadline:
                await self._clock.sleep(0.02)
            if not wt.done():
                wt.cancel()
            await asyncio.gather(wt, return_exceptions=True)
            if cancel_error is not None:
                raise cancel_error
        if not reached_terminal:
            # Timed out without a terminal task: cancel the job so in-flight work
            # stops, then raise (never hand back a still-running task the caller
            # could mistake for finished).
            job_final = await self.get_job(job_id)
            task_final = await self.get_task(task_id)
            if (job_final is None or job_final.status not in terminal or
                    task_final is None or task_final.status not in terminal):
                raise TaskCancellationDidNotConvergeError(
                    job_id, task_id, getattr(job_final, "status", "missing"),
                    getattr(task_final, "status", "missing"),
                    self._options.cancel_grace_seconds)
            raise TaskRunTimeoutError(job_id, task_id, wait_timeout)
        return await self.get_task(task_id)


__all__: "list[str]" = [
    "TaskRuntimeOptions",
    "TaskRuntime",
    "TaskStoreRequiredError",
]
