#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskWorker: the claim-execute-commit loop.

A single worker claims tasks, builds a :class:`TaskContext`, runs the handler
under a heartbeat (lease renewal) + timeout, and commits the outcome. An
``asyncio.Semaphore`` gates concurrent handler coroutines up to
``max_concurrency``. A ``shutdown`` event or coroutine cancellation drains:
stop claiming, cancel in-flight handlers, wait for them to finish.

Heartbeat failure (the lease was lost to a reclaimer) triggers the task's
cancellation token so a cooperative handler can stop; commit then fails with
``TaskClaimLostError`` and the task is recovered by a later pass.
"""

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from .models import JobStatus, TaskFailureKind, TaskStatus
from .metrics import NoopTaskMetrics, TaskMetrics
from .protocols import (
    CancellationToken,
    Clock,
    TaskContext,
    TaskFailure,
    TaskHandler,
    TaskRequest,
    TaskSuccess,
)
from .runtime import TaskRuntimeOptions
from ..security.redact import redact_exception, redact_text
from .store import TaskClaimLostError, TaskStore

if TYPE_CHECKING:
    from .models import TaskRecord

# Post-recovery hook: given the tasks a recovery pass reset, reconcile any
# side effects outside the task store (e.g. canceling Runs orphaned by a
# crashed worker). Best-effort -- errors in the hook must not stall recovery.
RecoverHook = Callable[[Sequence["TaskRecord"]], Awaitable[None]]


class TaskWorker:
    def __init__(
        self,
        *,
        task_store: TaskStore,
        handlers: "dict[str, TaskHandler]",
        options: TaskRuntimeOptions,
        clock: Clock,
        metrics: "TaskMetrics | None" = None,
        on_recovered: "RecoverHook | None" = None,
    ) -> None:
        self._task_store = task_store
        self._handlers = handlers
        self._options = options
        self._clock = clock
        self._metrics: TaskMetrics = metrics or NoopTaskMetrics()
        self._on_recovered = on_recovered

    async def run(
        self,
        *,
        worker_id: str,
        shutdown: "asyncio.Event | None" = None,
    ) -> None:
        shutdown = shutdown or asyncio.Event()
        sem = asyncio.Semaphore(self._options.max_concurrency)
        inflight: "set[asyncio.Task]" = set()
        last_recover = self._clock.now()
        recover_every = max(self._options.lease_seconds * 2, 30.0)
        handlers = tuple(self._handlers) if self._handlers else None

        try:
            while not shutdown.is_set():
                if (self._clock.now() - last_recover).total_seconds() >= recover_every:
                    recovered = await self._task_store.recover_expired(
                        now=self._clock.now(), limit=100
                    )
                    if recovered:
                        await self._metrics.inc_counter(
                            "task_recovered_total", labels={"count_bucket": "batch"}
                        )
                        if self._on_recovered is not None:
                            # Best-effort: a hook error must not stall recovery.
                            try:
                                await self._on_recovered(recovered)
                            except Exception:  # noqa: BLE001
                                pass
                    last_recover = self._clock.now()

                # Acquire the permit BEFORE claiming: a claimed task without a
                # running heartbeat burns its lease while blocked on the sem.
                await sem.acquire()
                claimed = await self._task_store.claim(
                    worker_id=worker_id,
                    now=self._clock.now(),
                    lease_seconds=self._options.lease_seconds,
                    handlers=handlers,
                )
                if claimed is None:
                    sem.release()
                    try:
                        await asyncio.wait_for(
                            shutdown.wait(),
                            timeout=self._options.poll_interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                await self._metrics.inc_counter(
                    "task_claim_total", labels={"handler": claimed.task.handler}
                )
                claimed_at = self._clock.now()
                t = asyncio.create_task(self._run_one(sem, claimed, claimed_at))
                inflight.add(t)
                t.add_done_callback(inflight.discard)
        finally:
            await self._metrics.set_gauge("worker_active_tasks", float(len(inflight)))
            # Drain: cancel in-flight handlers (propagates CancelledError into
            # _execute, which triggers their cancellation tokens) and wait.
            for t in inflight:
                t.cancel()
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)

    async def _run_one(self, sem: asyncio.Semaphore, claimed, claimed_at) -> None:
        try:
            await self._execute(claimed, claimed_at)
        except asyncio.CancelledError:
            pass  # shutdown/cancel — the finally in _execute already committed
        except Exception:  # noqa: BLE001 - must not leak the semaphore permit
            pass
        finally:
            sem.release()

    async def _execute(self, claimed, claimed_at) -> None:
        claim = claimed.claim
        task = claimed.task
        job = claimed.job
        handler = self._handlers.get(task.handler)
        # Queue wait: time from claim to execute-start.
        await self._metrics.observe_duration(
            "task_wait_seconds",
            (self._clock.now() - claimed_at).total_seconds(),
            labels={"handler": task.handler},
        )

        if handler is None:
            try:
                await self._task_store.commit_failure(
                    claim,
                    TaskFailure(
                        kind=TaskFailureKind.HANDLER_NOT_FOUND,
                        error_type="HandlerNotFound",
                        message=f"no handler registered as {task.handler!r}",
                    ),
                )
            except TaskClaimLostError:
                pass
            return

        cancellation = CancellationToken()
        context = TaskContext(
            job_id=job.id,
            task_id=task.id,
            attempt_id=claim.attempt_id,
            fencing_token=claim.fencing_token,
            worker_id=claim.worker_id,
            principal=job.principal,
            actor_chain=(
                task.actor_chain if task.actor_chain is not None else job.actor_chain
            ),
            delegated_scopes=(
                task.delegated_scopes
                if task.delegated_scopes is not None
                else job.actor_chain.delegated_scopes
            ),
            budget=job.budget,
            resource_snapshots=task.resource_snapshots,
            cancellation=cancellation,
        )
        request = TaskRequest(
            input_artifact=None,
            metadata=dict(task.metadata),
        )

        heartbeat = asyncio.create_task(self._heartbeat(claim, cancellation))
        cancel_watcher = asyncio.create_task(
            self._watch_cancellation(claim, cancellation)
        )
        try:
            outcome: "TaskSuccess | TaskFailure | None" = None
            handler_cancelled = False
            started = self._clock.now()
            try:
                timeout = task.timeout_seconds
                coro = handler.execute(request, context)
                if timeout is not None:
                    outcome = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    outcome = await coro
            except asyncio.TimeoutError:
                cancellation.trigger()
                outcome = TaskFailure(
                    kind=TaskFailureKind.TIMEOUT,
                    error_type="Timeout",
                    message=f"handler exceeded {timeout}s",
                )
            except asyncio.CancelledError:
                # Worker shutdown mid-handler: leave CLAIMED for recovery; the
                # CANCELLED outcome is intentionally not committed.
                cancellation.trigger()
                handler_cancelled = True
            except Exception as exc:  # noqa: BLE001 - handler failures are typed here
                outcome = TaskFailure(
                    kind=TaskFailureKind.INTERNAL,
                    error_type=type(exc).__name__,
                    message=redact_exception(exc),
                )
            # Handler execution latency distribution (success + every failure
            # kind, including timeout/cancel).
            await self._metrics.observe_duration(
                "task_duration_seconds",
                (self._clock.now() - started).total_seconds(),
                labels={"handler": task.handler},
            )

            # Validate the handler's return: a buggy handler returning None or a
            # non-TaskOutcome crashes the store -- convert to INTERNAL.
            if not isinstance(outcome, (TaskSuccess, TaskFailure)):
                outcome = TaskFailure(
                    kind=TaskFailureKind.INTERNAL,
                    error_type="InvalidOutcome",
                    message=(
                        f"handler returned {type(outcome).__name__}, "
                        "expected TaskSuccess/TaskFailure"
                    ),
                )

            # Redact every persisted failure message (not only INTERNAL): a
            # handler-supplied TIMEOUT/CANCELLED/PERMANENT message can carry a
            # credential and is stored verbatim otherwise.
            if isinstance(outcome, TaskFailure):
                outcome = dataclasses.replace(
                    outcome, message=redact_text(outcome.message)
                )

            if handler_cancelled:
                return

            # Commit with the heartbeat still alive: a transient store error at
            # commit must not cause the handler to re-run. We retry the COMMIT
            # (re-reading to confirm we still hold the claim between tries), not
            # the handler. Only after the retry budget is exhausted do we leave
            # the task CLAIMED for recovery.
            await self._commit_with_retry(claim, outcome, handler=task.handler)
            # Observe job-level terminal transitions for operator metrics. Best
            # effort: a read failure never affects the committed task.
            try:
                updated_job = await self._task_store.get_job(job.id)
                if updated_job is not None and updated_job.status.value in (
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ):
                    await self._metrics.inc_counter(
                        "job_completed_total",
                        labels={"status": updated_job.status.value},
                    )
            except Exception:  # noqa: BLE001 - metrics are best-effort
                pass
        finally:
            heartbeat.cancel()
            cancel_watcher.cancel()

    async def _commit_with_retry(
        self,
        claim,
        outcome: "TaskSuccess | TaskFailure",
        *,
        handler: str,
    ) -> None:
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                if isinstance(outcome, TaskSuccess):
                    await self._task_store.commit_success(claim, outcome)
                    await self._metrics.inc_counter(
                        "task_success_total", labels={"handler": handler}
                    )
                else:
                    failed = await self._task_store.commit_failure(claim, outcome)
                    await self._metrics.inc_counter(
                        "task_failure_total",
                        labels={"handler": handler, "failure_kind": outcome.kind.value},
                    )
                    if failed.status == TaskStatus.RETRY_WAIT:
                        # The failure was retryable and the task re-queued.
                        await self._metrics.inc_counter(
                            "task_retry_total", labels={"handler": handler}
                        )
                return
            except TaskClaimLostError:
                # Lease was reclaimed by another worker; it owns the task now.
                return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - transient store error
                if attempt >= max_attempts:
                    # Commit unconfirmed; leave CLAIMED for recovery to converge.
                    return
                await self._clock.sleep(self._options.heartbeat_seconds)
                if not await self._still_holds_claim(claim):
                    # We no longer hold the claim (reclaimed, or the commit did
                    # land but its confirmation was lost). Stop either way.
                    return

    async def _still_holds_claim(self, claim) -> bool:
        try:
            task = await self._task_store.get_task(claim.task_id)
        except Exception:  # noqa: BLE001 - a failed re-read means we can't confirm
            return False
        return (
            task is not None
            and task.status in (TaskStatus.CLAIMED, TaskStatus.CANCELLING)
            and task.active_attempt_id == claim.attempt_id
        )

    async def _heartbeat(self, claim, cancellation: CancellationToken) -> None:
        """Renew the lease periodically while the handler runs. If renewal
        fails (lease reclaimed), trigger cancellation so the handler stops."""
        while True:
            await self._clock.sleep(self._options.heartbeat_seconds)
            try:
                await self._task_store.renew_lease(
                    task_id=claim.task_id,
                    attempt_id=claim.attempt_id,
                    worker_id=claim.worker_id,
                    fencing_token=claim.fencing_token,
                    now=self._clock.now(),
                    lease_seconds=self._options.lease_seconds,
                )
            except TaskClaimLostError:
                await self._metrics.inc_counter("task_lease_renew_failure_total")
                cancellation.trigger()
                return
            except Exception:  # noqa: BLE001 - transient store error: keep retrying
                # Count it (a silent death here would mask lease-renew failures
                # from the metrics) and keep going -- the lease is still ours
                # until it expires, and the sleep above paces the retries.
                await self._metrics.inc_counter("task_lease_renew_failure_total")
                continue

    async def _watch_cancellation(self, claim, cancellation: CancellationToken) -> None:
        """Poll the task status; if it transitions to CANCELLING (via
        request_cancel from another caller), trigger the handler's
        cancellation token so a cooperative handler can stop early."""
        while True:
            await self._clock.sleep(self._options.heartbeat_seconds)
            try:
                task = await self._task_store.get_task(claim.task_id)
                if task is None:
                    return
                if task.status.value in ("cancelling", "cancelled"):
                    cancellation.trigger()
                    return
            except Exception:  # noqa: BLE001 - poll failures are non-fatal
                return
            try:
                await self._task_store.renew_lease(
                    task_id=claim.task_id,
                    attempt_id=claim.attempt_id,
                    worker_id=claim.worker_id,
                    fencing_token=claim.fencing_token,
                    now=self._clock.now(),
                    lease_seconds=self._options.lease_seconds,
                )
            except TaskClaimLostError:
                cancellation.trigger()
                return


__all__: "list[str]" = ["TaskWorker"]
