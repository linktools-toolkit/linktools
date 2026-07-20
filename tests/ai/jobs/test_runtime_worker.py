#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JobRuntime + JobWorker end-to-end (plan section 28 phase-5 acceptance).

Drives the full claim → execute → commit loop over a FilesystemStorage-backed
FilesystemTaskStore with real asyncio timing (tiny intervals) and a SystemClock.
"""

import asyncio
import dataclasses

from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.jobs.models import (
    ActorChain,
    ActorRef,
    JobRecord,
    JobStatus,
    RetryPolicy,
    SideEffectPolicy,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
    TaskRecord,
    TaskStatus,
)
from linktools.ai.jobs.protocols import (
    CancellationToken,
    TaskContext,
    TaskFailure,
    TaskRequest,
    TaskSuccess,
)
from linktools.ai.jobs.runtime import JobRuntime, JobRuntimeOptions

FAST = JobRuntimeOptions(
    poll_interval_seconds=0.01, lease_seconds=2.0, heartbeat_seconds=0.1
)


def _job(clock) -> JobRecord:
    return JobRecord(
        id="j1",
        status=JobStatus.PENDING,
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        budget=TaskBudget(),
        root_task_id="t1",
        input_artifact_id=None,
        output_artifact_id=None,
        version=1,
        created_at=clock.now(),
        started_at=None,
        finished_at=None,
    )


def _task(clock, *, handler="echo") -> TaskRecord:
    return TaskRecord(
        id="t1",
        job_id="j1",
        parent_task_id=None,
        key="k",
        handler=handler,
        status=TaskStatus.PENDING,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=(),
        retry_policy=RetryPolicy(max_attempts=1),
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


class _EchoHandler:
    async def execute(self, request: TaskRequest, context: TaskContext) -> TaskSuccess:
        return TaskSuccess()


class _FailingHandler:
    async def execute(self, request: TaskRequest, context: TaskContext) -> TaskFailure:
        return TaskFailure(
            kind=TaskFailureKind.PERMANENT, error_type="BadInput", message="nope"
        )


class _RaisingHandler:
    async def execute(self, request: TaskRequest, context: TaskContext) -> TaskSuccess:
        raise RuntimeError("boom")


class _SlowHandler:
    async def execute(self, request: TaskRequest, context: TaskContext) -> TaskSuccess:
        # Sleeps well past the task's timeout_seconds so the worker's
        # asyncio.wait_for raises TimeoutError -> TaskFailureKind.TIMEOUT.
        await asyncio.sleep(0.3)
        return TaskSuccess()


async def _wait_for_job(runtime, job_id, status, timeout=3.0):
    elapsed = 0.0
    while elapsed < timeout:
        job = await runtime.get_job(job_id)
        if job and job.status == status:
            return job
        await asyncio.sleep(0.01)
        elapsed += 0.01
    return await runtime.get_job(job_id)


def test_end_to_end_success(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"echo": _EchoHandler()}, options=FAST
        )
        await runtime.create_job(_job(runtime.clock), _task(runtime.clock))
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        job = await _wait_for_job(runtime, "j1", JobStatus.SUCCEEDED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED
        assert (await runtime.get_task("t1")).status == TaskStatus.SUCCEEDED

    asyncio.run(run())


def test_permanent_failure_marks_task_failed(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"fail": _FailingHandler()}, options=FAST
        )
        await runtime.create_job(
            _job(runtime.clock), _task(runtime.clock, handler="fail")
        )
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        task = await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        assert task is not None
        assert (await runtime.get_task("t1")).status == TaskStatus.FAILED
        assert task.status == JobStatus.FAILED

    asyncio.run(run())


def test_handler_timeout_marks_task_failed_with_timeout_kind(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"slow": _SlowHandler()}, options=FAST
        )
        task = dataclasses.replace(
            _task(runtime.clock, handler="slow"), timeout_seconds=0.05
        )
        await runtime.create_job(_job(runtime.clock), task)
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        attempts = await runtime.list_attempts("t1")
        assert attempts[0].failure_kind == TaskFailureKind.TIMEOUT

    asyncio.run(run())


def test_handler_exception_is_internal_failure(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"raise": _RaisingHandler()}, options=FAST
        )
        await runtime.create_job(
            _job(runtime.clock), _task(runtime.clock, handler="raise")
        )
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        attempts = await runtime.list_attempts("t1")
        assert attempts[0].failure_kind == TaskFailureKind.INTERNAL
        assert attempts[0].error_type == "RuntimeError"

    asyncio.run(run())


def test_handler_not_found_marks_task_failed(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(storage=storage, handlers={}, options=FAST)
        await runtime.create_job(
            _job(runtime.clock), _task(runtime.clock, handler="missing")
        )
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        attempts = await runtime.list_attempts("t1")
        assert attempts[0].failure_kind == TaskFailureKind.HANDLER_NOT_FOUND

    asyncio.run(run())


def test_cancel_propagates_to_job(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"echo": _EchoHandler()}, options=FAST
        )
        await runtime.create_job(_job(runtime.clock), _task(runtime.clock))
        # Cancel before the worker picks it up.
        job = await runtime.request_cancel("j1", reason="user")
        assert job.status == JobStatus.CANCELLED
        assert (await runtime.get_task("t1")).status == TaskStatus.CANCELLED

    asyncio.run(run())


def test_shutdown_drains_cleanly(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"echo": _EchoHandler()}, options=FAST
        )
        # No tasks; just verify run() exits when shutdown fires.
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await asyncio.sleep(0.05)  # let it poll once
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)  # must exit cleanly, no hang

    asyncio.run(run())


class _CountingHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, request: TaskRequest, context: TaskContext) -> TaskSuccess:
        self.calls += 1
        return TaskSuccess()


class _LeakyHandler:
    async def execute(self, request: TaskRequest, context: TaskContext) -> TaskFailure:
        return TaskFailure(
            kind=TaskFailureKind.PERMANENT,
            error_type="Auth",
            message="Authorization: Bearer secrettoken123",
        )


class _FlakyCommitStore:
    """Delegates everything to an inner JobStore, but ``commit_success``
    raises a transient error the first ``fail_times`` times before delegating.
    Used to prove a transient commit error is retried, not treated as a handler
    failure."""

    def __init__(self, inner, fail_times: int) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def commit_success(self, claim, outcome):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("transient db outage")
        return await self._inner.commit_success(claim, outcome)


def test_transient_commit_error_is_retried_without_rerun(tmp_path) -> None:
    """A transient store error at commit must not re-run the handler: the
    worker retries the COMMIT (re-reading to confirm it still holds the claim),
    never the handler."""
    from linktools.ai.storage.filesystem.task import FilesystemTaskStore
    from linktools.ai.jobs.protocols import SystemClock
    from linktools.ai.jobs.worker import JobWorker

    async def run() -> None:
        clock = SystemClock()
        inner = FilesystemTaskStore(tmp_path, clock=clock)
        flaky = _FlakyCommitStore(inner, fail_times=2)
        handler = _CountingHandler()
        worker = JobWorker(
            task_store=flaky,
            handlers={"echo": handler},
            options=FAST,
            clock=clock,
        )
        await inner.create_job(_job(clock), _task(clock))

        shutdown = asyncio.Event()
        wt = asyncio.create_task(worker.run(worker_id="w", shutdown=shutdown))

        elapsed = 0.0
        succeeded = False
        while elapsed < 5.0:
            t = await flaky.get_task("t1")
            if t and t.status == TaskStatus.SUCCEEDED:
                succeeded = True
                break
            await asyncio.sleep(0.02)
            elapsed += 0.02

        shutdown.set()
        await asyncio.wait_for(wt, timeout=5)
        assert succeeded, "task never succeeded"
        assert handler.calls == 1, "handler must not be re-run on a commit error"
        assert flaky._calls == 3, "commit must have been retried until it succeeded"

    asyncio.run(run())


class _FlakyRenewStore:
    """Wraps an inner store; ``renew_lease`` always raises a transient error so
    the heartbeat's generic-Exception arm is exercised: it must count the
    failure and keep retrying, not silently die (which would mask
    lease-renew failures from the metrics)."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def renew_lease(self, **kwargs):
        raise RuntimeError("transient renew outage")


def test_heartbeat_renew_failure_is_counted_not_silently_dropped(tmp_path) -> None:
    from linktools.ai.storage.filesystem.task import FilesystemTaskStore
    from linktools.ai.jobs.metrics import CountersTaskMetrics
    from linktools.ai.jobs.protocols import SystemClock
    from linktools.ai.jobs.worker import JobWorker

    async def run() -> None:
        clock = SystemClock()
        inner = FilesystemTaskStore(tmp_path, clock=clock)
        flaky = _FlakyRenewStore(inner)
        metrics = CountersTaskMetrics()
        worker = JobWorker(
            task_store=flaky,
            handlers={"slow": _SlowHandler()},  # runs ~0.3s so heartbeat fires
            options=FAST,
            clock=clock,
            metrics=metrics,
        )
        await inner.create_job(_job(clock), _task(clock, handler="slow"))
        shutdown = asyncio.Event()
        wt = asyncio.create_task(worker.run(worker_id="w", shutdown=shutdown))
        elapsed = 0.0
        succeeded = False
        while elapsed < 5.0:
            t = await flaky.get_task("t1")
            if t and t.status == TaskStatus.SUCCEEDED:
                succeeded = True
                break
            await asyncio.sleep(0.02)
            elapsed += 0.02
        shutdown.set()
        await asyncio.wait_for(wt, timeout=5)
        assert succeeded, "task must still succeed despite renew failures"
        # The heartbeat counted each failed renewal instead of dying silently.
        assert metrics.counter("task_lease_renew_failure_total") >= 1.0

    asyncio.run(run())


def test_heartbeat_cancels_when_renew_failures_burn_past_lease(tmp_path) -> None:
    # When transient renew failures persist past the lease deadline, the
    # heartbeat must trigger the cancellation token so the handler stops
    # producing side effects under a stale claim (double-execution guard).
    import types
    from datetime import datetime, timedelta, timezone

    from linktools.ai.storage.filesystem.task import FilesystemTaskStore
    from linktools.ai.jobs.metrics import CountersTaskMetrics
    from linktools.ai.jobs.worker import JobWorker

    class _FakeClock:
        def __init__(self) -> None:
            self._t = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)

        def now(self) -> "datetime":
            return self._t

        async def sleep(self, seconds: float) -> None:
            self._t = self._t + timedelta(seconds=seconds)
            # Yield so asyncio.wait_for can interrupt this loop if the deadline
            # arm ever regresses (otherwise the tight loop hangs instead of
            # failing cleanly).
            await asyncio.sleep(0)

    async def run() -> None:
        clock = _FakeClock()
        inner = FilesystemTaskStore(tmp_path, clock=clock)
        flaky = _FlakyRenewStore(inner)  # renew_lease always raises
        metrics = CountersTaskMetrics()
        worker = JobWorker(
            task_store=flaky, handlers={}, options=FAST, clock=clock, metrics=metrics
        )
        claim = types.SimpleNamespace(
            task_id="t1", attempt_id="a1", worker_id="w", fencing_token=1
        )
        cancellation = CancellationToken()
        # The heartbeat loops: each iteration sleeps heartbeat_seconds (which
        # advances the fake clock), renew fails (counter++), until the clock
        # crosses the lease deadline (now > confirmed_lease_expires_at) -- then
        # cancellation triggers and the loop returns.
        initial_lease_expires_at = clock.now() + timedelta(seconds=FAST.lease_seconds)
        await asyncio.wait_for(
            worker._heartbeat(claim, cancellation, initial_lease_expires_at), timeout=5
        )
        assert cancellation.is_set, "cancellation must fire once the lease burns"
        assert metrics.counter("task_lease_renew_failure_total") >= 1.0

    asyncio.run(run())


def test_handler_failure_message_is_redacted(tmp_path) -> None:
    """A credential embedded in a handler-returned failure message (any kind,
    not only INTERNAL) is redacted before it reaches the audit trail."""
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage, handlers={"leak": _LeakyHandler()}, options=FAST
        )
        await runtime.create_job(
            _job(runtime.clock), _task(runtime.clock, handler="leak")
        )
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        attempts = await runtime.list_attempts("t1")
        msg = attempts[0].error_message
        assert "secrettoken123" not in msg
        assert "REDACTED" in msg

    asyncio.run(run())


def test_metrics_counters_fire_on_lifecycle(tmp_path) -> None:
    """CountersTaskMetrics records claim/success counters as the worker drives a
    task to completion (low-cardinality labels only -- handler)."""
    from linktools.ai.jobs.metrics import CountersTaskMetrics

    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        metrics = CountersTaskMetrics()
        runtime = JobRuntime(
            storage=storage,
            handlers={"echo": _EchoHandler()},
            options=FAST,
            metrics=metrics,
        )
        await runtime.create_job(_job(runtime.clock), _task(runtime.clock))
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.SUCCEEDED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        assert metrics.counter("task_claim_total") >= 1.0
        assert metrics.counter("task_success_total", labels={"handler": "echo"}) >= 1.0
        assert metrics.counter("job_created_total") >= 1.0
        assert metrics.counter(
            "job_completed_total", labels={"status": JobStatus.SUCCEEDED.value}
        ) >= 1.0
        # Handler execution latency is observed as a duration sample.
        assert metrics.samples("task_duration_seconds", labels={"handler": "echo"})
        # Queue wait (claim -> execute-start) is observed too.
        assert metrics.samples("task_wait_seconds", labels={"handler": "echo"})

    asyncio.run(run())


def test_metrics_emit_task_retry_total_on_retryable_failure(tmp_path) -> None:
    """A retryable (TRANSIENT) failure with max_attempts>2 re-queues the task;
    task_retry_total records each retry."""
    from linktools.ai.jobs.metrics import CountersTaskMetrics

    class _TransientHandler:
        async def execute(self, request, context):
            return TaskFailure(
                kind=TaskFailureKind.TRANSIENT, error_type="Flaky", message="retry me"
            )

    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        metrics = CountersTaskMetrics()
        runtime = JobRuntime(
            storage=storage,
            handlers={"flaky": _TransientHandler()},
            options=FAST,
            metrics=metrics,
        )
        task = dataclasses.replace(
            _task(runtime.clock, handler="flaky"),
            retry_policy=RetryPolicy(max_attempts=2),
        )
        await runtime.create_job(_job(runtime.clock), task)
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        # One retry happened (attempt 1 failed TRANSIENT -> RETRY_WAIT -> retry).
        assert metrics.counter("task_retry_total", labels={"handler": "flaky"}) >= 1.0

    asyncio.run(run())


def test_orphan_run_reconciler_cancels_superseded_runs(tmp_path) -> None:
    """When recovery supersedes an attempt that had bound a run, the run
    canceler is invoked so the orphaned Run is not left RUNNING (section 21.5)."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from linktools.ai.storage.filesystem.task import FilesystemTaskStore

    class _FakeClock:
        def __init__(self, start):
            self._t = start

        def now(self):
            return self._t

        def advance(self, seconds):
            self._t = self._t + timedelta(seconds=seconds)

        async def sleep(self, seconds):
            self.advance(seconds)

    async def run() -> None:
        clock = _FakeClock(datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc))
        task_store = FilesystemTaskStore(tmp_path, clock=clock)
        canceled: "list[str]" = []

        async def cancel(run_id: str) -> None:
            canceled.append(run_id)

        runtime = JobRuntime(
            storage=SimpleNamespace(tasks=task_store),
            handlers={"echo": _EchoHandler()},
            options=FAST,
            clock=clock,
            run_canceler=cancel,
        )
        await runtime.create_job(_job(clock), _task(clock))
        claimed = await task_store.claim(
            worker_id="w1", now=clock.now(), lease_seconds=30
        )
        await task_store.bind_run(
            task_id="t1",
            attempt_id=claimed.claim.attempt_id,
            fencing_token=claimed.claim.fencing_token,
            worker_id="w1",
            run_id="run-1",
        )
        clock.advance(60)  # expire the lease
        recovered = await task_store.recover_expired(now=clock.now(), limit=10)
        await runtime._reconcile_orphan_runs(recovered)
        assert canceled == ["run-1"]

    asyncio.run(run())
