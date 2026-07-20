#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3/4 reliability fixes: SQL task envelope determinism (7.1), resource
snapshot TOCTOU (7.4), complete claim-ownership checking (8.1), and the
Clock-driven run_one_task with TaskRunTimeoutError (8.3)."""

import asyncio
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.models import Asset, AssetInfo, AssetKind
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.storage.sqlalchemy.models import Base, TaskRow
from linktools.ai.storage.sqlalchemy.task import (
    TASK_ENVELOPE_SCHEMA_VERSION,
    SqlAlchemyTaskStore,
    _store_dt,
)
from linktools.ai.jobs.models import (
    ActorChain,
    ActorRef,
    AttemptStatus,
    JobRecord,
    JobStatus,
    RetryPolicy,
    ScopeSet,
    SideEffectPolicy,
    TaskBudget,
    TaskPrincipal,
    TaskRecord,
    TaskStatus,
)
from linktools.ai.jobs.protocols import TaskContext, TaskRequest, TaskSuccess
from linktools.ai.jobs.handlers.runtime import (
    MappingRunnableResolver,
    RunnableRef,
)
from linktools.ai.jobs.runtime import JobRuntime, JobRuntimeOptions
from linktools.ai.jobs.store import (
    TaskClaim,
    TaskRunTimeoutError,
    UnsupportedTaskSchemaError,
    claim_matches_task,
)
from linktools.ai.jobs.models import resolve_effective_scopes


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- 7.1 --------


async def _sql_store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/p3.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyTaskStore(session_factory=factory)


def _envelope_task(clock, *, depth=2) -> TaskRecord:
    return TaskRecord(
        id="t1",
        job_id="j1",
        parent_task_id=None,
        key="k",
        handler="h",
        status=TaskStatus.PENDING,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=("d1",),
        retry_policy=RetryPolicy(max_attempts=2),
        side_effect_policy=SideEffectPolicy(),
        attempt_count=0,
        available_at=clock,
        lease_owner=None,
        lease_expires_at=None,
        fencing_token=0,
        active_attempt_id=None,
        timeout_seconds=None,
        resource_snapshots=(),
        version=1,
        created_at=clock,
        updated_at=clock,
        depth=depth,
        delegated_scopes=ScopeSet.of("read", "write"),
        actor_chain=ActorChain(
            actors=(ActorRef("user", "alice"),), delegated_scopes=ScopeSet.of("read", "write")
        ),
    )


def test_sql_task_round_trip_preserves_depth_scopes_actor_chain(tmp_path) -> None:
    """The SQL envelope serializes depth/delegated_scopes/actor_chain on the
    write side (the old _task_envelope dropped them for root tasks) so they
    survive a round-trip and a replay can trust the narrowed permission chain."""

    async def run() -> None:
        store = await _sql_store(tmp_path)
        now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        job = JobRecord(
            id="j1",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),),
                delegated_scopes=ScopeSet.allow_all()),
            budget=TaskBudget(),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        await store.create_job(job, _envelope_task(now))
        round_tripped = await store.get_task("t1")
        assert round_tripped.depth == 2
        assert round_tripped.delegated_scopes == ScopeSet.of("read", "write")
        assert round_tripped.actor_chain is not None
        assert round_tripped.actor_chain.delegated_scopes == ScopeSet.of("read", "write")
        assert round_tripped.dependencies == ("d1",)

    _run(run())


def test_sql_task_envelope_carries_schema_version(tmp_path) -> None:
    """Every persisted task row carries the current envelope schema version."""

    async def run() -> None:
        store = await _sql_store(tmp_path)
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
        await store.create_job(job, _envelope_task(now, depth=0))
        factory = store._session_factory
        async with factory() as session:
            row = await session.get(TaskRow, "t1")
            env = json.loads(row.data_json)
        assert env["schema_version"] == TASK_ENVELOPE_SCHEMA_VERSION

    _run(run())


def test_unknown_task_schema_is_rejected(tmp_path) -> None:
    """A future envelope schema version is rejected on read, never silently
    misinterpreted (a security-sensitive field could otherwise be restored to a
    broader permission)."""

    async def run() -> None:
        store = await _sql_store(tmp_path)
        now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        future_env = {
            "schema_version": 999,
            "dependencies": [],
            "retry_policy": {"max_attempts": 1},
            "side_effect_policy": {"mode": "none"},
            "resource_snapshots": [],
            "depth": 0,
            "delegated_scopes": None,
            "actor_chain": None,
            "metadata": {},
        }
        await _insert_task_row(store, "future", now, future_env)
        with pytest.raises(UnsupportedTaskSchemaError):
            await store.get_task("future")

    _run(run())


def test_v1_envelope_missing_security_field_fails_closed(tmp_path) -> None:
    """A v1 envelope that omits a security field is rejected on read, so a
    partial/corrupt row never restores broader permission (unrestricted scopes /
    no actor chain) than the task held."""

    async def run() -> None:
        store = await _sql_store(tmp_path)
        now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        # v1 schema_version but delegated_scopes is MISSING (not even null).
        incomplete_env = {
            "schema_version": TASK_ENVELOPE_SCHEMA_VERSION,
            "dependencies": [],
            "retry_policy": {"max_attempts": 1},
            "side_effect_policy": {"mode": "none"},
            "resource_snapshots": [],
            "depth": 0,
            "actor_chain": None,
            "metadata": {},
        }
        await _insert_task_row(store, "incomplete", now, incomplete_env)
        with pytest.raises(UnsupportedTaskSchemaError):
            await store.get_task("incomplete")

    _run(run())


def test_v1_envelope_null_security_field_fails_closed(tmp_path) -> None:
    """A v1 envelope whose delegated_scopes is present but null (corruption or a
    hand-edited row) is rejected on read -- never silently widened to
    unrestricted. The same applies to a null delegated_scopes inside an
    otherwise-present actor_chain (it would widen via ActorChain's boundary
    normalization)."""

    async def run() -> None:
        store = await _sql_store(tmp_path)
        now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        base = {
            "schema_version": TASK_ENVELOPE_SCHEMA_VERSION,
            "dependencies": [],
            "retry_policy": {"max_attempts": 1},
            "side_effect_policy": {"mode": "none"},
            "resource_snapshots": [],
            "depth": 0,
            "wait_conditions": [],
            "wait_deadline_at": None,
            "metadata": {},
        }
        # Top-level delegated_scopes present but null.
        await _insert_task_row(
            store,
            "null-top",
            now,
            {**base, "delegated_scopes": None, "actor_chain": None},
            key="kt",
        )
        # actor_chain present but its delegated_scopes null.
        await _insert_task_row(
            store,
            "null-actor",
            now,
            {
                **base,
                "delegated_scopes": {"unrestricted": False, "values": ["read"]},
                "actor_chain": {"actors": [], "delegated_scopes": None},
            },
            key="ka",
        )
        for task_id in ("null-top", "null-actor"):
            with pytest.raises(UnsupportedTaskSchemaError):
                await store.get_task(task_id)

    _run(run())


async def _insert_task_row(store, task_id, now, env, *, key="k") -> None:
    factory = store._session_factory
    async with factory() as session:
        session.add(
            TaskRow(
                id=task_id,
                job_id="j1",
                parent_task_id=None,
                key=key,
                handler="h",
                status=TaskStatus.READY.value,
                input_artifact_id=None,
                output_artifact_id=None,
                attempt_count=0,
                available_at=_store_dt(now),
                lease_owner=None,
                lease_expires_at=None,
                fencing_token=0,
                active_attempt_id=None,
                timeout_seconds=None,
                version=1,
                created_at=_store_dt(now),
                updated_at=_store_dt(now),
                data_json=json.dumps(env),
            )
        )
        await session.commit()


# ---------------------------------------------------------------- 7.4 --------


class _CountingResource:
    """Wraps a AssetStore to count get/stat calls so the snapshot's
    single-read contract (no stat) can be asserted."""

    def __init__(self, inner: AssetStore) -> None:
        self._inner = inner
        self.get_calls = 0
        self.stat_calls = 0

    async def get(self, path):
        self.get_calls += 1
        return await self._inner.get(path)

    async def stat(self, path):
        self.stat_calls += 1
        return await self._inner.stat(path)


def test_snapshot_uses_single_resource_read(tmp_path) -> None:
    """The snapshot reads the resource ONCE (get), never stat+get, so the pinned
    version/etag and the sealed content come from the same read (no TOCTOU)."""

    async def run() -> None:
        from linktools.ai.asset.models import WriteOptions

        from linktools.ai.jobs.snapshot import snapshot_resource

        resources = AssetStore(primary=MemoryAssetBackend())
        await resources.put(
            AssetPath("/data/file.txt"),
            b"snapshot-me",
            options=WriteOptions(content_type="text/plain"),
        )
        counter = _CountingResource(resources)
        artifacts = build_artifact_store_from_assets(resources)  # separate store for the sealed blob
        snap = await snapshot_resource(
            counter, artifacts, "/data/file.txt", tenant_id="t1"
        )
        assert counter.get_calls == 1
        assert counter.stat_calls == 0
        assert snap.version >= 1

    _run(run())


def test_snapshot_picks_consistent_revision_under_concurrent_change(
    tmp_path,
) -> None:
    """If the resource changes between a hypothetical stat and get, the snapshot
    still pins the version/etag that matches the bytes it actually sealed (the
    single get's info), never a stale stat revision against new content."""

    async def run() -> None:
        import hashlib

        from linktools.ai.jobs.snapshot import snapshot_resource

        class _MutatingResource:
            async def get(self, path):
                # The single read returns the NEW content + its own info.
                return Asset(
                    info=AssetInfo(
                        path=path,
                        kind=AssetKind.FILE,
                        etag="etag-new",
                        version=7,
                        content_type="text/plain",
                        size=len(b"new-bytes"),
                        modified_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
                    ),
                    content=b"new-bytes",
                )

            async def stat(self, path):
                raise AssertionError("snapshot_resource must not call stat")

        artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
        snap = await snapshot_resource(
            _MutatingResource(), artifacts, "/data/x", tenant_id="t1"
        )
        assert snap.version == 7
        assert snap.etag == "etag-new"
        assert snap.sha256 == hashlib.sha256(b"new-bytes").hexdigest()

    _run(run())


# ---------------------------------------------------------------- 8.1 --------


def _task_with(*, status, lease_owner="w1", attempt_id="a1", fencing=5) -> TaskRecord:
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    return TaskRecord(
        id="t1",
        job_id="j1",
        parent_task_id=None,
        key="k",
        handler="h",
        status=status,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=(),
        retry_policy=RetryPolicy(),
        side_effect_policy=SideEffectPolicy(),
        attempt_count=1,
        available_at=now,
        lease_owner=lease_owner,
        lease_expires_at=None,
        fencing_token=fencing,
        active_attempt_id=attempt_id,
        timeout_seconds=None,
        resource_snapshots=(),
        version=2,
        created_at=now,
        updated_at=now,
    )


def test_claim_matches_task_checks_all_four_fields() -> None:
    claim = TaskClaim(task_id="t1", attempt_id="a1", worker_id="w1", fencing_token=5)
    # Fully matching CLAIMED task -> owns.
    assert claim_matches_task(claim, _task_with(status=TaskStatus.CLAIMED))
    # CANCELLING still belongs to the owning worker.
    assert claim_matches_task(claim, _task_with(status=TaskStatus.CANCELLING))
    # Each field drifting independently breaks ownership.
    assert not claim_matches_task(
        claim, _task_with(status=TaskStatus.CLAIMED, lease_owner="w2")
    )
    assert not claim_matches_task(
        claim, _task_with(status=TaskStatus.CLAIMED, attempt_id="other")
    )
    assert not claim_matches_task(
        claim, _task_with(status=TaskStatus.CLAIMED, fencing=99)
    )
    # A non-claim status never matches.
    assert not claim_matches_task(claim, _task_with(status=TaskStatus.READY))
    assert not claim_matches_task(claim, _task_with(status=TaskStatus.SUCCEEDED))


# ---------------------------------------------------------------- 8.4 --------


def test_resolve_effective_scopes_never_expands() -> None:
    """Effective-scope resolution only narrows or inherits -- it never grants a
    scope the parent did not hold."""
    unrestricted = ScopeSet.allow_all()
    ab = ScopeSet.of("a", "b")
    # None requested = inherit the parent's scopes exactly.
    assert resolve_effective_scopes(None, ab) == ab
    assert resolve_effective_scopes(None, unrestricted) == unrestricted
    # A concrete request intersects (parent not held -> dropped).
    assert resolve_effective_scopes(("a", "x"), ab) == ScopeSet.of("a")
    assert resolve_effective_scopes(("a", "x"), unrestricted) == ScopeSet.of("a", "x")
    assert resolve_effective_scopes(("z",), ab) == ScopeSet()  # no overlap -> denied
    # order preserved from the parent, not the request.
    assert resolve_effective_scopes(("b", "a"), ScopeSet.of("a", "b", "c")) == ScopeSet.of(
        "a", "b"
    )
    # An unrestricted requested ScopeSet collapses to the parent (cannot exceed it).
    assert resolve_effective_scopes(unrestricted, ab) == ab


def test_root_task_inherits_job_scopes_at_creation(tmp_path) -> None:
    """A root task with no explicit scopes inherits the job's actor-chain scopes
    at creation (persisted, not left as an unresolved None)."""
    from linktools.ai.storage.filesystem.task import FilesystemTaskStore

    async def run() -> None:
        store = FilesystemTaskStore(tmp_path / "t")
        now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        job = JobRecord(
            id="j1",
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(
                actors=(ActorRef("user", "alice"),),
                delegated_scopes=ScopeSet.of("read", "write"),
            ),
            budget=TaskBudget(),
            root_task_id="t1",
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        root = TaskRecord(
            id="t1",
            job_id="j1",
            parent_task_id=None,
            key="k",
            handler="h",
            status=TaskStatus.PENDING,
            input_artifact_id=None,
            output_artifact_id=None,
            dependencies=(),
            retry_policy=RetryPolicy(),
            side_effect_policy=SideEffectPolicy(),
            attempt_count=0,
            available_at=now,
            lease_owner=None,
            lease_expires_at=None,
            fencing_token=0,
            active_attempt_id=None,
            timeout_seconds=None,
            resource_snapshots=(),
            version=1,
            created_at=now,
            updated_at=now,
            delegated_scopes=ScopeSet.allow_all(),
        )
        await store.create_job(job, root)
        persisted = await store.get_task("t1")
        # Resolved from the job's scopes at creation -- not left as None.
        assert persisted.delegated_scopes == ScopeSet.of("read", "write")

    asyncio.run(run())


# ---------------------------------------------------------------- 7.5 --------


def test_mapping_resolver_honors_revision_not_silently_ignored() -> None:
    """An id-only MappingRunnableResolver cannot honor a revision: it raises
    rather than silently returning whatever the id currently maps to (which
    could be a different agent after a mapping change, silently re-running a
    retry against new code). Revision-less refs resolve as before."""

    async def run() -> None:
        resolver = MappingRunnableResolver({"a": "spec-current"})
        with pytest.raises(ValueError):
            await resolver.resolve(RunnableRef(id="a", revision="rev-1"))
        # No revision -> resolves the current mapping (unchanged behavior).
        assert await resolver.resolve(RunnableRef(id="a")) == "spec-current"
        with pytest.raises(KeyError):
            await resolver.resolve(RunnableRef(id="missing"))

    _run(run())


# ---------------------------------------------------------------- 8.3 --------


class _ForeverHandler:
    """Blocks until its cancellation token fires, then returns."""

    async def execute(self, request: TaskRequest, context: TaskContext):
        await context.cancellation.wait()
        return TaskSuccess()


def test_run_one_task_timeout_raises_and_cancels(tmp_path) -> None:
    """run_one_task drives its wait through the Clock and, on timeout, cancels
    the job and raises TaskRunTimeoutError -- it never hands back a still-running
    task the caller could mistake for finished."""

    async def run() -> None:
        storage = FilesystemStorage(root=tmp_path)
        runtime = JobRuntime(
            storage=storage,
            handlers={"forever": _ForeverHandler()},
            options=JobRuntimeOptions(
                poll_interval_seconds=0.01,
                lease_seconds=2.0,
                heartbeat_seconds=0.05,
            ),
        )
        now = runtime.clock.now()
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
            handler="forever",
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
            resource_snapshots=(),
            version=1,
            created_at=now,
            updated_at=now,
        )
        await runtime.create_job(job, task)
        with pytest.raises(TaskRunTimeoutError):
            await runtime.run_one_task(
                "forever", tenant_id="t1", wait_timeout=0.15
            )
        # The timeout must never leave the job falsely SUCCEEDED; the cancel
        # drives it toward CANCELLING/CANCELLED (best-effort, raced against the
        # shutting-down worker), so the invariant is "not done = not succeeded".
        job_now = await runtime.get_job("j1")
        assert job_now.status != JobStatus.SUCCEEDED
        task_now = await runtime.get_task("t1")
        assert task_now.status != TaskStatus.SUCCEEDED

    asyncio.run(run())
