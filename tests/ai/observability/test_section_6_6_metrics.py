#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coverage for the observability metric floor: each of the 11 required
categories is wired to a REAL recording site, not just named in the allowlist.
Every test triggers the actual code path that increments the counter and
asserts the InMemoryMetrics sink observed it."""

from linktools.ai.artifact import ANONYMOUS_PROVENANCE

import asyncio
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from linktools.ai.observability.metrics import (
    HARDENING_METRICS,
    InMemoryMetrics,
)


# Names required by the §6.6 floor (15 instruments across the 11 categories;
# categories 5, 6, 11 split into more than one instrument).
_REQUIRED_METRIC_NAMES = (
    "runtime_build_failure_total",
    "storage_capability_validation_failure_total",
    "event_codec_failure_total",
    "critical_event_persist_failure_total",
    "asset_cas_conflict_total",
    "artifact_digest_mismatch_total",
    "artifact_orphan_total",
    "job_lease_expiry_total",
    "job_stale_fence_total",
    "job_recovery_total",
    "approval_replay_reject_total",
    "catalog_revision_refresh_total",
    "external_adapter_conformance_failure_total",
    "artifact_blob_upload_failure_total",
    "artifact_orphan_cleanup_failure_total",
)


def test_required_metric_names_are_in_allowlist():
    """Every required metric name is registered in HARDENING_METRICS so an
    InMemoryMetrics sink actually records it."""
    missing = [n for n in _REQUIRED_METRIC_NAMES if n not in HARDENING_METRICS]
    assert not missing, f"required metric names missing from allowlist: {missing}"


# --- category 1 + 9: runtime build failure + storage capability validation ---


def test_runtime_build_failure_total_fires_on_capability_gate_shortfall(tmp_path):
    """When enforce_storage_capability_gate raises, both
    runtime_build_failure_total and storage_capability_validation_failure_total
    increment before the error propagates."""
    from linktools.ai._runtime.build import RuntimeBuildConfig, build_runtime_components
    from linktools.ai._runtime.dependencies import RuntimeDependencies
    from linktools.ai.errors import StorageRequirementsNotMetError
    from linktools.ai.run.requirements import RuntimeRequirements
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.features import CoordinationScope
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    metrics = InMemoryMetrics()
    storage = FilesystemStorage(root=tmp_path)
    config = RuntimeBuildConfig(
        storage=storage,
        providers=RuntimeDependencies(),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
        metrics=metrics,
        requirements=RuntimeRequirements(coordination=CoordinationScope.DISTRIBUTED),
    )
    with pytest.raises(StorageRequirementsNotMetError):
        build_runtime_components(config)
    assert metrics.counters.get("runtime_build_failure_total") == 1
    assert metrics.counters.get("storage_capability_validation_failure_total") == 1


# --- category 2: event codec failure (encode + migrate) ---


def test_event_codec_failure_total_fires_on_encode_of_unregistered_payload():
    """Encoding a payload whose class is not registered raises EventSchemaError
    and records event_codec_failure_total with phase=encode."""
    from linktools.ai.events.registry import EventCodec, EventSchemaError, build_default_registry

    metrics = InMemoryMetrics()
    codec = EventCodec(build_default_registry(), metrics=metrics)

    @dataclass(frozen=True)
    class _Unregistered:
        not_a_registered_payload: str = "x"

    with pytest.raises(EventSchemaError):
        codec.encode(_Unregistered())
    assert metrics.counters.get("event_codec_failure_total") == 1


def test_event_codec_failure_total_fires_on_migrate_with_future_schema_version():
    """Decoding an event whose schema_version is newer than the registered
    one raises EventSchemaError and records event_codec_failure_total with
    phase=migrate."""
    from linktools.ai.events.payloads import RunStarted
    from linktools.ai.events.registry import EventCodec, EventSchemaError, build_default_registry

    metrics = InMemoryMetrics()
    codec = EventCodec(build_default_registry(), metrics=metrics)

    with pytest.raises(EventSchemaError):
        codec.decode("RunStarted", schema_version=999, data={"run_id": "r", "runnable_id": "rn"})
    assert metrics.counters.get("event_codec_failure_total") == 1


# --- category 3: critical event persist failure (recovery) ---


class _FailingEventStore:
    """Wraps a real EventStore but makes append() raise, simulating a critical
    event persist failure during recovery. The recovery path records the
    failure rather than discarding the journal silently."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def append(self, **kwargs):
        raise RuntimeError("simulated persist failure")


def test_critical_event_persist_failure_total_fires_on_recovery_failure(tmp_path):
    """A FilesystemRunCommitCoordinator recovery that hits an event-store
    failure during critical-event reappend records the metric instead of
    silently dropping the journal."""
    from linktools.ai.events.context import EventContext
    from linktools.ai.run.commit import CompleteRunCommand
    from linktools.ai.run.context import RunContext
    from linktools.ai.run.models import RunInput, RunRecord, RunResult, RunStatus, RunnableType
    from linktools.ai.session.models import MessageRole, NewSessionMessage
    from linktools.ai.session.models import SessionRecord, SessionStatus
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
    from linktools.ai.storage.filesystem.journal import (
        TransactionJournal,
        TransactionKind,
    )

    metrics = InMemoryMetrics()

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="sess-x",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await storage.runs.create(
            RunRecord(
                id="run-x",
                root_run_id="run-x",
                parent_run_id=None,
                session_id="sess-x",
                runnable_id="agent-1",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="x"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            )
        )

        # Wrap the event store so any append fails. The coordinator uses this
        # store directly (no facade indirection), so the failure surfaces
        # inside _reappend_critical_events during recovery.
        failing_events = _FailingEventStore(storage.events)
        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=failing_events,
            transactions_root=tmp_path / "transactions",
            metrics=metrics,
        )
        # Land the complete commit through the REAL event store first so the
        # run reaches its commit point (SUCCEEDED) and recovery treats only
        # the critical-event reappend as the failing step.
        real_coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            transactions_root=tmp_path / "transactions-real",
        )
        await real_coordinator.complete(
            CompleteRunCommand(
                run_id="run-x",
                session_id="sess-x",
                expected_version=1,
                messages=(
                    NewSessionMessage(
                        role=MessageRole.USER, content="hi", run_id="run-x"
                    ),
                ),
                checkpoint_payload=b'{"m":[]}',
                result=RunResult(output="ok"),
                event_context=EventContext.from_run_context(
                    RunContext(
                        run_id="run-x",
                        root_run_id="run-x",
                        parent_run_id=None,
                        session_id="sess-x",
                        runnable_id="agent-1",
                        runnable_type=RunnableType.AGENT,
                        user_id=None,
                        tenant_id=None,
                        workspace=None,
                    )
                ),
            )
        )
        # Now stage an incomplete journal at the failing coordinator: the run
        # is already SUCCEEDED, so recovery's _reappend_critical_events path
        # runs -- and the failing event store makes it record the metric. Use
        # a distinct commit_id so _append_critical_event_once's dedup check
        # (which reads via the real store's list()) does not short-circuit
        # before the failing append.
        journal = TransactionJournal(tmp_path / "transactions")
        journal.begin(
            kind=TransactionKind.COMPLETE,
            run_id="run-x",
            target_run_status="succeeded",
            commit_id="complete:run-x:forced-recovery-failure",
            command={
                "session_id": "sess-x",
                "messages": [],
                "event_context": {
                    "stream_id": "run-x",
                    "run_id": "run-x",
                    "root_run_id": "run-x",
                    "parent_run_id": None,
                    "session_id": "sess-x",
                    "runnable_id": "agent-1",
                },
            },
        )
        await coordinator.recover_incomplete_commits()
        assert metrics.counters.get("critical_event_persist_failure_total") == 1

    asyncio.run(_run())


# --- category 4: asset CAS conflict ---


def test_asset_cas_conflict_total_fires_on_idempotency_key_reuse():
    """An AssetStore.put that reuses an idempotency key with a different body
    raises IdempotencyConflictError and records asset_cas_conflict_total."""
    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.models import WriteOptions
    from linktools.ai.asset.path import AssetPath
    from linktools.ai.asset.store import AssetStore
    from linktools.ai.errors import IdempotencyConflictError

    metrics = InMemoryMetrics()
    store = AssetStore(primary=MemoryAssetBackend(), metrics=metrics)

    async def _run():
        await store.put(
            AssetPath("/p"),
            b"first",
            options=WriteOptions(idempotency_key="k1"),
        )
        with pytest.raises(IdempotencyConflictError):
            await store.put(
                AssetPath("/p"),
                b"different-body",
                options=WriteOptions(idempotency_key="k1"),
            )

    asyncio.run(_run())
    assert metrics.counters.get("asset_cas_conflict_total") == 1


# --- category 5a: artifact digest mismatch ---


def test_artifact_digest_mismatch_total_fires_on_corrupt_blob():
    """A get() that reads back content whose sha256 does not match the pinned
    record digest raises ArtifactIntegrityError and records
    artifact_digest_mismatch_total."""
    from linktools.ai.artifact.store import ArtifactStore
    from linktools.ai.artifact.models import ArtifactIntegrityError
    from linktools.ai.storage.protocols import ArtifactBlobStore, ArtifactRecordStore

    metrics = InMemoryMetrics()

    # Tiny in-memory implementations of the two Protocols. The blob store is
    # rigged so the stored bytes do NOT match the digest the record pins, to
    # trigger the integrity check on get().
    class _CorruptBlobStore:
        async def put_if_absent(self, *, digest, source, size):
            async for _ in source:
                pass
            from linktools.ai.storage.protocols import BlobInfo
            return BlobInfo(digest=digest, size=size, content_type=None)

        @asynccontextmanager
        async def open(self, *, digest):
            async def _chunks():
                yield b"tampered-bytes"

            yield _chunks()

    class _MemRecordStore:
        def __init__(self):
            self._by_id: "dict[str, Any]" = {}

        async def put(self, record):
            self._by_id[record.ref.id] = record
            return record

        async def get(self, artifact_id, *, tenant_id):
            r = self._by_id.get(artifact_id)
            return r if r is not None and r.tenant_id == tenant_id else None

    store = ArtifactStore(_CorruptBlobStore(), _MemRecordStore(), metrics=metrics)

    async def _run():
        # The pinned sha256 is for "real-bytes"; the blob store returns
        # "tampered-bytes", so the read-time integrity check trips.
        record = await store.put(
            content=b"real-bytes",
            media_type="text/plain",
            tenant_id="t1",
            provenance=ANONYMOUS_PROVENANCE,
        )
        with pytest.raises(ArtifactIntegrityError):
            await store.get(artifact_id=record.ref.id, tenant_id="t1")

    asyncio.run(_run())
    assert metrics.counters.get("artifact_digest_mismatch_total") == 1


# --- category 5b: artifact orphan (swept) ---


def test_artifact_orphan_total_fires_when_sweep_deletes_orphan(tmp_path):
    """Each blob the sweep deletes increments artifact_orphan_total."""
    from linktools.ai.storage.filesystem.artifact import (
        FilesystemArtifactBlobStore,
        FilesystemArtifactRecordStore,
    )
    from linktools.ai.storage.orphan import (
        OrphanSweepConfig,
        sweep_orphan_blobs,
    )

    metrics = InMemoryMetrics()

    async def _run():
        blobs = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
        records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
        # Write one blob with no record pointing at it -> it is an orphan
        # candidate. Use the real digest so put_if_absent succeeds.
        content = b"orphaned-blob"
        digest = hashlib.sha256(content).hexdigest()

        async def _src():
            yield content

        await blobs.put_if_absent(digest=digest, source=_src(), size=len(content))
        # Make the orphan past the grace window by anchoring "now" far in the
        # future; the mtime stored by the filesystem backend is recent.
        stats = await sweep_orphan_blobs(
            blobs,
            records,
            OrphanSweepConfig(grace_period=timedelta(seconds=0)),
            now=datetime.now(timezone.utc) + timedelta(days=2),
            metrics=metrics,
        )
        assert stats.deleted == 1

    asyncio.run(_run())
    assert metrics.counters.get("artifact_orphan_total") == 1


# --- category 6: job lease expiry / stale fence / recovery ---


def test_job_lease_expiry_and_recovery_total_fire_on_recover_expired(tmp_path):
    """When JobWorker.run's recovery sweep resets expired tasks, both
    job_lease_expiry_total and job_recovery_total increment by the number of
    recovered tasks."""
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
    from linktools.ai.jobs.protocols import SystemClock
    from linktools.ai.jobs.runtime import JobRuntimeOptions
    from linktools.ai.jobs.worker import JobWorker
    from linktools.ai.storage.filesystem.job import FilesystemJobStore

    metrics = InMemoryMetrics()

    async def _run():
        clock = SystemClock()
        store = FilesystemJobStore(tmp_path, clock=clock)
        now = clock.now()
        await store.create_job(
            JobRecord(
                id="j-le",
                status=JobStatus.PENDING,
                principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
                actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
                budget=TaskBudget(),
                root_task_id="t-le",
                input_artifact_id=None,
                output_artifact_id=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            ),
            TaskRecord(
                id="t-le",
                job_id="j-le",
                parent_task_id=None,
                key="k",
                handler="echo",
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
            ),
        )
        # Claim the task so it becomes CLAIMED with a lease, then move the
        # clock forward past the lease so the next recovery sweep observes it
        # as expired and recovers it.
        claimed = await store.claim(
            worker_id="w",
            now=now,
            lease_seconds=1.0,
        )
        assert claimed is not None
        # Advance the clock past the lease expiry.
        future = now + timedelta(seconds=60)
        advanced_clock = _AdvancedClock(future)

        class _Echo:
            async def execute(self, request, context):
                from linktools.ai.jobs.protocols import TaskSuccess
                return TaskSuccess()

        worker = JobWorker(
            task_store=store,
            handlers={"echo": _Echo()},
            options=JobRuntimeOptions(
                poll_interval_seconds=0.01,
                lease_seconds=2.0,
                heartbeat_seconds=0.1,
            ),
            clock=advanced_clock,
            observability_metrics=metrics,
        )
        shutdown = asyncio.Event()
        wt = asyncio.create_task(worker.run(worker_id="w", shutdown=shutdown))
        # Give the worker a beat to enter its recover_expired branch.
        elapsed = 0.0
        while elapsed < 2.0 and metrics.counters.get("job_recovery_total", 0) < 1:
            await asyncio.sleep(0.02)
            elapsed += 0.02
        shutdown.set()
        await asyncio.wait_for(wt, timeout=5)
        assert metrics.counters.get("job_lease_expiry_total", 0) >= 1
        assert metrics.counters.get("job_recovery_total", 0) >= 1

    asyncio.run(_run())


class _AdvancedClock:
    """A clock anchored at a future moment that ADVANCES by 60s on each
    now() call. The worker's recover_expired branch runs only when
    (now - last_recover) >= recover_every (>= 30s by default), so a fixed
    clock would never enter the branch; the advance gets the worker past that
    gate on its first iteration."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        self._moment = self._moment + timedelta(seconds=60)
        return self._moment

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def test_job_stale_fence_total_fires_on_lost_claim(tmp_path):
    """When the worker's heartbeat loses the claim (TaskClaimLostError from
    renew_lease), job_stale_fence_total increments."""
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
    from linktools.ai.jobs.protocols import SystemClock
    from linktools.ai.jobs.runtime import JobRuntimeOptions
    from linktools.ai.jobs.worker import JobWorker
    from linktools.ai.storage.filesystem.job import FilesystemJobStore

    metrics = InMemoryMetrics()

    class _LostClaimStore:
        """Wraps the real store; renew_lease always loses the claim."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def renew_lease(self, **kwargs):
            from linktools.ai.jobs.store import TaskClaimLostError
            raise TaskClaimLostError("simulated stale fence")

    async def _run():
        clock = SystemClock()
        inner = FilesystemJobStore(tmp_path, clock=clock)
        store = _LostClaimStore(inner)
        now = clock.now()
        await inner.create_job(
            JobRecord(
                id="j-sf",
                status=JobStatus.PENDING,
                principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
                actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
                budget=TaskBudget(),
                root_task_id="t-sf",
                input_artifact_id=None,
                output_artifact_id=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            ),
            TaskRecord(
                id="t-sf",
                job_id="j-sf",
                parent_task_id=None,
                key="k",
                handler="slow",
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
            ),
        )

        class _SlowHandler:
            async def execute(self, request, context):
                # Long enough for at least one heartbeat cycle to fire.
                await asyncio.sleep(0.3)
                from linktools.ai.jobs.protocols import TaskSuccess
                return TaskSuccess()

        worker = JobWorker(
            task_store=store,
            handlers={"slow": _SlowHandler()},
            options=JobRuntimeOptions(
                poll_interval_seconds=0.01,
                lease_seconds=2.0,
                heartbeat_seconds=0.05,
            ),
            clock=clock,
            observability_metrics=metrics,
        )
        shutdown = asyncio.Event()
        wt = asyncio.create_task(worker.run(worker_id="w", shutdown=shutdown))
        elapsed = 0.0
        while elapsed < 3.0 and metrics.counters.get("job_stale_fence_total", 0) < 1:
            await asyncio.sleep(0.02)
            elapsed += 0.02
        shutdown.set()
        await asyncio.wait_for(wt, timeout=5)
        assert metrics.counters.get("job_stale_fence_total", 0) >= 1

    asyncio.run(_run())


# --- category 7: approval replay reject ---


def test_approval_replay_reject_total_fires_on_dedupe_conflict(tmp_path):
    """A create_or_get_pending call whose (run_id, tool_call_id) already exists
    with a different tool_name/arguments is rejected and records
    approval_replay_reject_total."""
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.errors import ApprovalConflictError

    metrics = InMemoryMetrics()
    store = FilesystemApprovalStore(root=tmp_path, metrics=metrics)

    binding = {
        "descriptor_fingerprint": "fp",
        "handler_revision": "h",
        "provider_revision": "p",
        "policy_revision": "pol",
        "capability_revision": "cap",
        "result_processor_revision": "rp",
        "arguments_hash": "ah-original",
    }

    async def _run():
        await store.create_or_get_pending(
            tenant_id="t1",
            run_id="r1",
            tool_call_id="tc1",
            tool_name="shell",
            reason="x",
            arguments={"cmd": "ls"},
            approval_id="ap1",
            binding=binding,
        )
        # Same dedupe key but a different tool_name + arguments_hash -> reject.
        with pytest.raises(ApprovalConflictError):
            await store.create_or_get_pending(
                tenant_id="t1",
                run_id="r1",
                tool_call_id="tc1",
                tool_name="different_tool",
                reason="x",
                arguments={"cmd": "rm"},
                approval_id="ap2",
                binding={**binding, "arguments_hash": "ah-different"},
            )

    asyncio.run(_run())
    assert metrics.counters.get("approval_replay_reject_total") == 1


# --- category 8: catalog revision refresh ---


def test_catalog_revision_refresh_total_fires_on_revision_change():
    """RevisionCache._ensure_fresh records catalog_revision_refresh_total each
    time the source's revision moves."""
    from linktools.ai.catalog.contracts import RevisionCache

    metrics = InMemoryMetrics()

    class _BumpingSource:
        """A source whose revision changes on each call (first call r1, then
        r2, then r3...)."""

        def __init__(self):
            self._n = 0

        async def revision(self):
            self._n += 1
            return f"r{self._n}"

        async def list_ids(self, suffix):
            return ()

        async def read(self, path):
            raise FileNotFoundError(path)

    class _NoopCodec:
        def decode(self, item_id, raw):
            return raw

    cache = RevisionCache(_BumpingSource(), _NoopCodec(), metrics=metrics)

    async def _run():
        # First call seeds the cache (revision moves from None -> r1).
        await cache._ensure_fresh()
        # Second call sees a new revision (r1 -> r2) -> another refresh.
        await cache._ensure_fresh()

    asyncio.run(_run())
    assert metrics.counters.get("catalog_revision_refresh_total") == 2


# --- category 10: external adapter conformance failure ---


def test_external_adapter_conformance_failure_total_fires_on_contract_failure():
    """A contract test method that raises records the metric through the
    contract's conformance_metrics sink before propagating the failure."""
    from testing.contracts import ArtifactBlobStoreContract

    metrics = InMemoryMetrics()

    class _FailingBlobStore:
        async def put_if_absent(self, *, digest, source, size):
            raise AssertionError("simulated contract failure")

        @asynccontextmanager
        async def open(self, *, digest):
            async def _chunks():
                yield b""

            yield _chunks()

    class _ContractSubclass(ArtifactBlobStoreContract):
        conformance_metrics = metrics

        def blob_store(self):
            return _FailingBlobStore()

    # The contract test method raises (the inner store's AssertionError) and
    # records the metric first.
    instance = _ContractSubclass()
    with pytest.raises(AssertionError):
        instance.test_put_if_absent_is_idempotent_on_digest()
    assert metrics.counters.get("external_adapter_conformance_failure_total") == 1


# --- category 11a: artifact blob upload failure ---


def test_artifact_blob_upload_failure_total_fires_on_digest_mismatch(tmp_path):
    """A put_if_absent whose claimed digest does not match the bytes' actual
    sha256 raises ArtifactIntegrityError and records
    artifact_blob_upload_failure_total at the blob-store level."""
    from linktools.ai.artifact.models import ArtifactIntegrityError
    from linktools.ai.storage.filesystem.artifact import (
        FilesystemArtifactBlobStore,
    )

    metrics = InMemoryMetrics()
    blobs = FilesystemArtifactBlobStore(
        blobs_root=tmp_path / "blobs", metrics=metrics
    )

    async def _run():
        async def _src():
            yield b"not-the-claimed-digest"

        with pytest.raises(ArtifactIntegrityError):
            await blobs.put_if_absent(
                digest="0" * 64, source=_src(), size=21
            )

    asyncio.run(_run())
    assert metrics.counters.get("artifact_blob_upload_failure_total") == 1


def test_artifact_blob_upload_failure_total_fires_on_size_mismatch(tmp_path):
    """A put_if_absent whose claimed size does not match the streamed byte count
    raises ArtifactIntegrityError (digest matches, size does not) and records
    artifact_blob_upload_failure_total."""
    import hashlib

    from linktools.ai.artifact.models import ArtifactIntegrityError
    from linktools.ai.storage.filesystem.artifact import (
        FilesystemArtifactBlobStore,
    )

    metrics = InMemoryMetrics()
    blobs = FilesystemArtifactBlobStore(
        blobs_root=tmp_path / "blobs", metrics=metrics
    )
    payload = b"payload"  # 7 bytes
    digest = hashlib.sha256(payload).hexdigest()

    async def _run():
        async def _src():
            yield payload

        with pytest.raises(ArtifactIntegrityError):
            await blobs.put_if_absent(
                digest=digest, source=_src(), size=999  # wrong size
            )

    asyncio.run(_run())
    assert metrics.counters.get("artifact_blob_upload_failure_total") == 1


def test_artifact_blob_upload_failure_total_fires_on_corrupt_existing(tmp_path):
    """A second put of an existing digest whose stored blob was tampered raises
    ArtifactIntegrityError (the dedup path re-hashes the existing blob) and
    records artifact_blob_upload_failure_total."""
    import hashlib

    from linktools.ai.artifact.models import ArtifactIntegrityError
    from linktools.ai.storage.filesystem.artifact import (
        FilesystemArtifactBlobStore,
    )

    metrics = InMemoryMetrics()
    blobs = FilesystemArtifactBlobStore(
        blobs_root=tmp_path / "blobs", metrics=metrics
    )
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()

    async def _run():
        async def _src():
            yield payload

        await blobs.put_if_absent(digest=digest, source=_src(), size=7)
        # Tamper the stored blob in place; the second put's dedup path re-hashes
        # it and refuses to record a reference to the corrupt blob.
        (tmp_path / "blobs" / digest[:2] / digest).write_bytes(b"TAMPERED")
        with pytest.raises(ArtifactIntegrityError):
            await blobs.put_if_absent(digest=digest, source=_src(), size=7)

    asyncio.run(_run())
    assert metrics.counters.get("artifact_blob_upload_failure_total") == 1


# --- category 11b: artifact orphan cleanup failure ---


def test_artifact_orphan_cleanup_failure_total_fires_on_delete_error(tmp_path):
    """A delete failure during the sweep records
    artifact_orphan_cleanup_failure_total and does not stall the sweep."""
    from linktools.ai.storage.filesystem.artifact import (
        FilesystemArtifactBlobStore,
        FilesystemArtifactRecordStore,
    )
    from linktools.ai.storage.orphan import (
        OrphanSweepConfig,
        sweep_orphan_blobs,
    )

    metrics = InMemoryMetrics()

    class _FailingDeleteBlobStore(FilesystemArtifactBlobStore):
        async def delete(self, *, digest):
            raise RuntimeError("simulated delete failure")

    async def _run():
        blobs = _FailingDeleteBlobStore(blobs_root=tmp_path / "blobs")
        records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
        content = b"orphaned-blob"
        digest = hashlib.sha256(content).hexdigest()

        async def _src():
            yield content

        await blobs.put_if_absent(digest=digest, source=_src(), size=len(content))
        stats = await sweep_orphan_blobs(
            blobs,
            records,
            OrphanSweepConfig(grace_period=timedelta(seconds=0)),
            now=datetime.now(timezone.utc) + timedelta(days=2),
            metrics=metrics,
        )
        # Delete failed -> nothing deleted, but the sweep ran cleanly.
        assert stats.deleted == 0

    asyncio.run(_run())
    assert metrics.counters.get("artifact_orphan_cleanup_failure_total") == 1
