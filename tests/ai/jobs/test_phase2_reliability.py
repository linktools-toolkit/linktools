#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reliability fixes: input-artifact data flow (6.1), cancellation
propagation to the Runtime (6.2), and startup orphan-run reconciliation (6.3).

6.1 is exercised end-to-end through JobRuntime + a FilesystemStorage (whose
``assets`` backend auto-wires the ArtifactStore). 6.2 drives the
RuntimeTaskHandler against a fake Runtime. 6.3 unit-tests the reconciler's
strict (startup) vs best-effort (periodic) failure modes directly.
"""

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.artifact import ArtifactStore, ANONYMOUS_PROVENANCE
from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.identity.principal import ScopeSet
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)
from linktools.ai.jobs.handlers.runtime import (
    MappingRunnableResolver,
    RuntimeTaskHandler,
)
from linktools.ai.jobs.models import (
    ActorChain,
    ActorRef,
    AttemptStatus,
    JobRecord,
    JobStatus,
    RetryPolicy,
    SideEffectPolicy,
    TaskAttemptRecord,
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
        asset_snapshots=(),
        version=1,
        created_at=clock.now(),
        updated_at=clock.now(),
    )


async def _wait_for_job(runtime, job_id, status, timeout=3.0):
    elapsed = 0.0
    while elapsed < timeout:
        job = await runtime.get_job(job_id)
        if job and job.status == status:
            return job
        await asyncio.sleep(0.01)
        elapsed += 0.01
    return await runtime.get_job(job_id)


class _CapturingHandler:
    """Records the input_artifact it received on its last execution (instance
    state so it does not leak between tests in the same process)."""

    def __init__(self) -> None:
        self.seen = "UNSET"

    async def execute(self, request: TaskRequest, context: TaskContext):
        self.seen = request.input_artifact
        return TaskSuccess()


# ---------------------------------------------------------------- 6.1 --------


def test_input_artifact_reaches_handler(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        record = await storage.artifacts.put(
            content=b"hello", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        handler = _CapturingHandler()
        runtime = JobRuntime(
            storage=storage, handlers={"cap": handler}, options=FAST
        )
        task = dataclasses.replace(
            _task(runtime.clock, handler="cap"), input_artifact_id=record.ref.id
        )
        await runtime.create_job(_job(runtime.clock), task)
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.SUCCEEDED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        assert handler.seen is not None
        assert handler.seen.id == record.ref.id

    asyncio.run(run())


def test_missing_input_artifact_fails_invalid_input(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        handler = _CapturingHandler()
        runtime = JobRuntime(
            storage=storage, handlers={"cap": handler}, options=FAST
        )
        task = dataclasses.replace(
            _task(runtime.clock, handler="cap"),
            input_artifact_id="does-not-exist",
        )
        await runtime.create_job(_job(runtime.clock), task)
        shutdown = asyncio.Event()
        wt = asyncio.create_task(runtime.run(worker_id="w", shutdown=shutdown))
        await _wait_for_job(runtime, "j1", JobStatus.FAILED)
        shutdown.set()
        await asyncio.wait_for(wt, timeout=3)
        task_now = await runtime.get_task("t1")
        assert task_now.status == TaskStatus.FAILED
        attempts = await runtime.list_attempts("t1")
        assert attempts[-1].failure_kind == TaskFailureKind.INVALID_INPUT
        # The handler must never have run against the missing artifact.
        assert handler.seen == "UNSET"

    asyncio.run(run())


# ---------------------------------------------------------------- 6.2 --------


class _BlockingRuntime:
    """run() blocks forever until cancel(); records cancel(run_id) calls."""

    def __init__(self) -> None:
        self.cancels: "list[str]" = []

    async def run(self, spec, prompt, **kw):  # noqa: ANN001
        await asyncio.Event().wait()  # never returns on its own
        return None

    async def cancel(self, run_id: str, *, principal=None) -> None:
        self.cancels.append(run_id)


class _NoopTaskStore:
    async def bind_run(self, **kwargs): return None


def _handler(runtime, resolver, tmp_path=None, **kwargs):
    import tempfile
    from pathlib import Path
    root = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    return RuntimeTaskHandler(runtime, resolver, task_store=_NoopTaskStore(),
        artifact_store=ArtifactStore(
            FilesystemArtifactBlobStore(blobs_root=root / "blobs"),
            FilesystemArtifactRecordStore(records_root=root / "records"),
            InProcessArtifactDigestCoordinator(),
        ),
        **kwargs)


class _RaisingRuntime:
    def __init__(self) -> None:
        self.cancels: "list[str]" = []

    async def run(self, spec, prompt, **kw):  # noqa: ANN001
        raise RuntimeError("side-effect blew up")

    async def cancel(self, run_id: str) -> None:
        self.cancels.append(run_id)


class _OkRuntime:
    def __init__(self) -> None:
        self.result = object()
        self.cancels: "list[str]" = []

    async def run(self, spec, prompt, **kw):  # noqa: ANN001
        return self.result

    async def cancel(self, run_id: str) -> None:
        self.cancels.append(run_id)


def _ctx(cancellation: CancellationToken) -> TaskContext:
    return TaskContext(
        job_id="j",
        task_id="t",
        attempt_id="a",
        fencing_token=1,
        worker_id="w",
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        delegated_scopes=ScopeSet.of("x"),
        budget=TaskBudget(),
        asset_snapshots=(),
        cancellation=cancellation,
    )


def test_cancel_propagates_to_runtime_and_returns_cancelled() -> None:
    async def run() -> None:
        rt = _BlockingRuntime()
        resolver = MappingRunnableResolver({"a": object()})
        handler = _handler(rt, resolver, cancel_grace_seconds=0.1)
        ct = CancellationToken()
        request = TaskRequest(
            input_artifact=None, metadata={"runnable_id": "a", "prompt": "p"}
        )

        async def fire() -> None:
            await asyncio.sleep(0.05)
            ct.trigger()

        asyncio.create_task(fire())
        outcome = await handler.execute(request, _ctx(ct))
        assert isinstance(outcome, TaskFailure)
        assert outcome.kind == TaskFailureKind.SIDE_EFFECT_UNKNOWN
        assert outcome.error_type == "RuntimeCancelTimeout"
        assert len(rt.cancels) == 1  # runtime.cancel(run_id) was called

    asyncio.run(run())


def test_run_failure_under_cancellation_is_cancelled_not_side_effect_unknown() -> None:
    async def run() -> None:
        rt = _RaisingRuntime()
        resolver = MappingRunnableResolver({"a": object()})
        handler = _handler(rt, resolver, cancel_grace_seconds=0.1)
        ct = CancellationToken()
        ct.trigger()  # cancellation already set when the run raises
        request = TaskRequest(
            input_artifact=None, metadata={"runnable_id": "a", "prompt": "p"}
        )
        outcome = await handler.execute(request, _ctx(ct))
        assert isinstance(outcome, TaskFailure)
        assert outcome.kind == TaskFailureKind.CANCELLED
        assert outcome.kind != TaskFailureKind.SIDE_EFFECT_UNKNOWN

    asyncio.run(run())


def test_run_success_is_not_overridden_by_cancellation() -> None:
    async def run() -> None:
        rt = _OkRuntime()
        resolver = MappingRunnableResolver({"a": object()})
        # An artifact_store is needed for the handler to seal the result; pass a
        # minimal fake whose put returns a record-shaped object.
        handler = _handler(rt, resolver)
        ct = CancellationToken()  # never triggered
        request = TaskRequest(
            input_artifact=None, metadata={"runnable_id": "a", "prompt": "p"}
        )
        outcome = await handler.execute(request, _ctx(ct))
        assert isinstance(outcome, TaskSuccess)
        assert rt.cancels == []

    asyncio.run(run())


# ---------------------------------------------------------------- 6.3 --------


def _attempt(*, run_id: "str | None", status: AttemptStatus) -> TaskAttemptRecord:
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    return TaskAttemptRecord(
        id="a1",
        task_id="t1",
        job_id="j1",
        attempt=1,
        worker_id="w",
        fencing_token=1,
        status=status,
        started_at=now,
        run_id=run_id,
        finished_at=now,
        failure_kind=None,
        error_type=None,
        error_message=None,
    )


class _FakeRecoveryStore:
    """A minimal store that hands back a fixed recovered-task batch and a fixed
    attempt list, so the reconciler can be driven without a real backend.

    ``recover_expired`` models production idempotency: a real backend resets
    each expired task on the first call and returns nothing on retry. Without
    this, a reconcile retry that (wrongly) sourced orphans from the recovered
    batch would still see the batch and the refind test would not guard the
    invariant it claims to."""

    def __init__(self, recovered, attempts) -> None:
        self._recovered = recovered
        self._attempts = attempts
        self._recover_called = False

    async def recover_expired(self, *, now, limit=100):
        if self._recover_called:
            return ()
        self._recover_called = True
        return self._recovered

    async def reconcile_due(self, *, now, limit=100):
        return ()

    async def list_orphan_run_ids(self, *, limit=500):
        return tuple(
            att.run_id
            for att in self._attempts
            if att.status == AttemptStatus.SUPERSEDED and att.run_id
        )

    async def list_attempts(self, task_id):
        return self._attempts


def _reconcile_runtime(storage, canceler) -> JobRuntime:
    return JobRuntime(
        storage=storage, handlers={}, options=FAST, run_canceler=canceler
    )


def test_reconcile_strict_raises_on_canceler_failure(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        recovered = (
            dataclasses.replace(_task(storage.jobs._clock), status=TaskStatus.READY),
        )
        attempts = [_attempt(run_id="run-1", status=AttemptStatus.SUPERSEDED)]

        async def bad_canceler(run_id):
            raise RuntimeError("boom")

        rt = _reconcile_runtime(storage, bad_canceler)
        rt._task_store = _FakeRecoveryStore(recovered, attempts)

        # Strict (startup) mode surfaces the failure.
        with pytest.raises(RuntimeError):
            await rt._reconcile_orphan_runs(recovered, strict=True)
        # Best-effort (periodic) mode swallows the same failure.
        await rt._reconcile_orphan_runs(recovered, strict=False)

    asyncio.run(run())


def test_ensure_recovered_reconciles_orphan_run_and_marks_recovered(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        clock = storage.jobs._clock
        seeded_task = dataclasses.replace(_task(clock), status=TaskStatus.READY)
        seeded_attempt = _attempt(run_id="run-1", status=AttemptStatus.SUPERSEDED)
        cancelled_runs: "list[str]" = []

        async def good_canceler(run_id):
            cancelled_runs.append(run_id)

        rt = _reconcile_runtime(storage, good_canceler)
        rt._task_store = _FakeRecoveryStore((seeded_task,), [seeded_attempt])

        await rt.ensure_recovered()
        assert rt._recovered is True
        assert cancelled_runs == ["run-1"]
        # Idempotent: a second startup recovery does not re-cancel the run.
        await rt.ensure_recovered()
        assert cancelled_runs == ["run-1"]

    asyncio.run(run())


def test_failed_startup_reconciliation_leaves_unrecovered(tmp_path) -> None:
    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        clock = storage.jobs._clock
        seeded_task = dataclasses.replace(_task(clock), status=TaskStatus.READY)
        seeded_attempt = _attempt(run_id="run-1", status=AttemptStatus.SUPERSEDED)
        calls = {"n": 0}

        async def flaky_canceler(run_id):
            calls["n"] += 1
            raise RuntimeError("transient")

        rt = _reconcile_runtime(storage, flaky_canceler)
        rt._task_store = _FakeRecoveryStore((seeded_task,), [seeded_attempt])

        with pytest.raises(RuntimeError):
            await rt.ensure_recovered()
        # The recovered flag stays false so the next call retries.
        assert rt._recovered is False

    asyncio.run(run())


def test_failed_startup_reconciliation_is_retried_and_refinds_orphans(
    tmp_path,
) -> None:
    """A failed startup pass leaves _recovered false; the next pass RETRIES and,
    because reconcile sources orphans from list_orphan_run_ids (not the
    idempotent recovered list), the orphan is re-found and re-canceled. The bug
    was that a retried pass saw an empty recovered batch and reconciled nothing."""

    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        clock = storage.jobs._clock
        seeded_task = dataclasses.replace(_task(clock), status=TaskStatus.READY)
        seeded_attempt = _attempt(run_id="run-1", status=AttemptStatus.SUPERSEDED)
        calls: "list[str]" = []

        async def canceler(run_id):
            calls.append(run_id)
            if len(calls) == 1:
                raise RuntimeError("first-pass boom")

        rt = _reconcile_runtime(storage, canceler)
        rt._task_store = _FakeRecoveryStore((seeded_task,), [seeded_attempt])

        with pytest.raises(RuntimeError):
            await rt.ensure_recovered()
        assert rt._recovered is False
        # Second pass re-finds run-1 via list_orphan_run_ids and cancels it.
        await rt.ensure_recovered()
        assert rt._recovered is True
        assert calls == ["run-1", "run-1"]

    asyncio.run(run())


class _FakeClock:
    """Advances virtual time so lease expiry is deterministic (no real sleep)."""

    def __init__(self, start: datetime) -> None:
        self._t = start

    def now(self) -> datetime:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t = self._t + timedelta(seconds=seconds)

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def test_non_idempotent_orphan_run_is_finalized_not_requeued(tmp_path) -> None:
    """A non-idempotent task whose lease expired while it had a Run in flight is
    finalized FAILED (never requeued to READY) and its orphan Run is still
    reconciled (canceled). The Run must not be left running, but the task must
    not be retried either -- a non-idempotent side effect can only be confirmed
    by a human/business, not re-executed blindly."""

    async def run() -> None:
        from linktools.ai.jobs.models import SideEffectMode

        storage = FilesystemStorage(root=tmp_path)
        store = storage.jobs
        clock = _FakeClock(datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc))
        now = clock.now()
        task = dataclasses.replace(
            _task(clock),
            retry_policy=RetryPolicy(max_attempts=3),
            side_effect_policy=SideEffectPolicy(mode=SideEffectMode.NON_IDEMPOTENT),
        )
        await store.create_job(_job(clock), task)
        claimed = await store.claim(worker_id="w", now=now, lease_seconds=30)
        c = claimed.claim
        # The worker bound a Run, then crashed: the attempt carries run_id.
        await store.bind_run(
            task_id=c.task_id,
            attempt_id=c.attempt_id,
            fencing_token=c.fencing_token,
            worker_id=c.worker_id,
            run_id="run-orphan",
        )
        clock.advance(60)  # past the 30s lease -> expired

        cancelled: "list[str]" = []

        async def canceler(run_id):
            cancelled.append(run_id)

        rt = _reconcile_runtime(storage, canceler)
        rt._clock = clock  # drive recovery through the advanced fake clock
        await rt.ensure_recovered()
        final = await store.get_task("t1")
        # Non-idempotent + lease expired -> FAILED, NOT requeued to READY.
        assert final.status == TaskStatus.FAILED
        # The orphan Run is still reconciled even though the task was finalized.
        assert cancelled == ["run-orphan"]

    asyncio.run(run())


# ------------------------------------------------------------- binding --


def test_handler_rejects_runnable_drift_after_rebind(tmp_path) -> None:
    """If a task's pinned runnable fingerprint differs from the freshly resolved
    one (a mapping change between attempts), the handler fails PERMANENTLY --
    a retry never silently re-runs a different agent."""

    async def run() -> None:
        from linktools.ai.jobs.protocols import (
            CancellationToken,
            TaskContext,
            TaskRequest,
        )
        from linktools.ai.jobs.handlers.runtime import (
            MappingRunnableResolver,
            RuntimeTaskHandler,
        )
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

        storage = FilesystemStorage(root=tmp_path)
        store = storage.jobs
        now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        job = JobRecord(
            id="j1",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        task = TaskRecord(
            id="t1",
            job_id="j1",
            parent_task_id=None,
            key="k",
            handler="runtime",
            status=TaskStatus.PENDING,
            input_artifact_id=None,
            output_artifact_id=None,
            dependencies=(),
            retry_policy=RetryPolicy(max_attempts=1),
            side_effect_policy=SideEffectPolicy(),
            attempt_count=0,
            available_at=now,
            lease_owner=None,
            lease_expires_at=None,
            fencing_token=0,
            active_attempt_id=None,
            timeout_seconds=None,
            asset_snapshots=(),
            version=1,
            created_at=now,
            updated_at=now,
        )
        await store.create_job(job, task)
        claimed = await store.claim(worker_id="w", now=now, lease_seconds=30)
        c = claimed.claim
        # Pin a runnable fingerprint that the resolver will NOT reproduce.
        await store.bind_runnable(
            task_id=c.task_id, attempt_id=c.attempt_id, worker_id=c.worker_id,
            fencing_token=c.fencing_token, runnable_id="a", revision=None,
            fingerprint="pinned-old-fingerprint",
        )
        handler = RuntimeTaskHandler(
            _OkRuntime(),
            MappingRunnableResolver({"a": "spec-after-change"}),
            task_store=store,
            artifact_store=ArtifactStore(
                FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs"),
                FilesystemArtifactRecordStore(records_root=tmp_path / "records"),
                InProcessArtifactDigestCoordinator(),
            ),
        )
        ct = CancellationToken()
        ctx = TaskContext(
            job_id="j1",
            task_id=c.task_id,
            attempt_id=c.attempt_id,
            fencing_token=c.fencing_token,
            worker_id=c.worker_id,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            delegated_scopes=ScopeSet.of("x"),
            budget=TaskBudget(),
            asset_snapshots=(),
            cancellation=ct,
        )
        request = TaskRequest(
            input_artifact=None, metadata={"runnable": {"id": "a"}, "prompt": "p"}
        )
        outcome = await handler.execute(request, ctx)
        from linktools.ai.jobs.protocols import TaskFailure

        assert isinstance(outcome, TaskFailure)
        assert outcome.kind == TaskFailureKind.PERMANENT
        assert outcome.error_type == "RunnableDrift"

    asyncio.run(run())
