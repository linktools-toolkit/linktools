#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol conformance contract mixins.

A downstream adapter subclasses a contract and implements the ``store`` (or
``coordinator``) factory, then runs the subclass under pytest. Every test
asserts an observable Protocol guarantee (idempotency, tenant isolation,
fencing monotonicity, unknown-event fail-closed, etc.), never an implementation
detail. The in-repo reference backends run these same contracts in
``tests/ai/storage/test_conformance_testkit.py``.

These mixins intentionally have no ``pytest`` base class -- they are plain
``object`` subclasses whose test_* methods are collected when a concrete
subclass inherits them alongside a backend fixture. That keeps the testkit
importable without pytest at module-eval time (the root package never imports
it)."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any


def _record_conformance_failure(
    instance: Any, name: str, exc: BaseException
) -> None:
    """Record an external-adapter conformance failure when the contract
    instance has a ``conformance_metrics`` attribute wired to an
    ObservabilityMetrics sink. Observability-only: the failure is re-raised
    after recording so the underlying contract test still surfaces the
    regression. The metric attribute carries the contract test's name and the
    exception class (both low-cardinality)."""
    sink = getattr(instance, "conformance_metrics", None)
    if sink is None:
        return
    sink.counter(
        "external_adapter_conformance_failure_total",
        attributes={"contract": name, "reason": type(exc).__name__},
    )


class _ConformanceRecorderMixin:
    """Internal mixin shared by every contract class. ``_contract_run`` wraps
    the asyncio.run call so a failure is observable through the optional
    ``conformance_metrics`` sink without changing the test's pass/fail
    semantics. The default (no sink wired) is a transparent passthrough."""

    def _contract_run(self, body: Any) -> None:
        try:
            asyncio.run(body())
        except BaseException as exc:
            _record_conformance_failure(self, self.__class__.__name__, exc)
            raise


class ArtifactBlobStoreContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`ArtifactBlobStore` Protocol.

    Subclasses must implement ``blob_store()`` returning a fresh, empty
    ArtifactBlobStore. The contract verifies the guarantees the artifact
    domain relies on: put_if_absent is idempotent on digest, digest mismatch
    fails, open streams back the exact bytes, stat/delete behave.
    """

    def blob_store(self) -> Any:
        raise NotImplementedError

    def test_put_if_absent_is_idempotent_on_digest(self) -> None:
        store = self.blob_store()
        content = b"artifact-bytes"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            info1 = await store.put_if_absent(
                digest=digest, source=_aiter(content), size=len(content)
            )
            # Second put with the same digest is a no-op reuse (idempotent),
            # not an error.
            info2 = await store.put_if_absent(
                digest=digest, source=_aiter(content), size=len(content)
            )
            assert info1.digest == info2.digest == digest

        self._contract_run(_run)

    def test_put_if_absent_rejects_digest_mismatch(self) -> None:
        store = self.blob_store()

        async def _run() -> None:
            from ...artifact.models import ArtifactIntegrityError

            with __import__("pytest").raises(ArtifactIntegrityError):
                await store.put_if_absent(
                    digest="0" * 64,
                    source=_aiter(b"not-the-claimed-digest"),
                    size=21,
                )

        self._contract_run(_run)

    def test_open_returns_the_pinned_bytes(self) -> None:
        store = self.blob_store()
        content = b"roundtrip"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            await store.put_if_absent(
                digest=digest, source=_aiter(content), size=len(content)
            )
            chunks = []
            async for chunk in store.open(digest=digest):
                chunks.append(chunk)
            assert b"".join(chunks) == content

        self._contract_run(_run)


class ArtifactRecordStoreContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`ArtifactRecordStore` Protocol: a record is
    retrievable by id+tenant, a foreign tenant learns nothing, delete is
    tenant-scoped."""

    def record_store(self) -> Any:
        raise NotImplementedError

    def _record(self, artifact_id: str, tenant: str) -> Any:
        from ...artifact.models import ArtifactRecord, ArtifactRef

        return ArtifactRecord(
            ref=ArtifactRef(id=artifact_id, sha256="x" * 64, media_type="", size=0),
            tenant_id=tenant,
            created_by_job_id=None,
            created_by_task_id=None,
            created_by_attempt_id=None,
            parent_artifact_ids=(),
            created_at=__import__("datetime").datetime(2026, 1, 1),
        )

    def test_get_is_tenant_scoped(self) -> None:
        store = self.record_store()

        async def _run() -> None:
            await store.put(self._record("a1", "t1"))
            assert (await store.get("a1", tenant_id="t1")) is not None
            # Foreign tenant learns nothing -- not even that the record exists.
            assert (await store.get("a1", tenant_id="t2")) is None

        self._contract_run(_run)

    def test_delete_is_tenant_scoped(self) -> None:
        store = self.record_store()

        async def _run() -> None:
            await store.put(self._record("a2", "t1"))
            deleted = await store.delete("a2", tenant_id="t1")
            assert deleted is True
            # A foreign tenant cannot delete (and the record is already gone
            # for the owner).
            assert (await store.delete("a2", tenant_id="t2")) is False

        self._contract_run(_run)


class AssetStoreContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`AssetStore` class: the primary+overlay
    composition a downstream adapter sits under.

    Subclasses must implement ``asset_store()`` returning a fresh, empty
    AssetStore. The contract verifies the guarantees the resource and
    artifact layers rely on: put/get/delete CRUD round-trip, a missing path
    returns None, putting the same path bumps version, and propfind paginates
    with depth filtering. The contract is backend-agnostic -- it never probes
    the underlying AssetBackend, only the observable AssetStore surface.
    """

    def asset_store(self) -> Any:
        raise NotImplementedError

    def test_put_get_roundtrip(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from ...asset.path import AssetPath

            path = AssetPath("/contract/roundtrip.txt")
            await store.put(path, b"hello")
            fetched = await store.get(path)
            assert fetched is not None
            assert fetched.content == b"hello"

        self._contract_run(_run)

    def test_get_missing_returns_none(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from ...asset.path import AssetPath

            assert (await store.get(AssetPath("/contract/never-existed"))) is None

        self._contract_run(_run)

    def test_delete_removes_resource(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from ...asset.path import AssetPath

            path = AssetPath("/contract/deleted.txt")
            await store.put(path, b"x")
            await store.delete(path)
            assert (await store.get(path)) is None

        self._contract_run(_run)

    def test_put_same_path_bumps_version(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from ...asset.path import AssetPath

            path = AssetPath("/contract/versioned.txt")
            first = await store.put(path, b"v1")
            second = await store.put(path, b"v2")
            assert second.info.version > first.info.version

        self._contract_run(_run)

    def test_propfind_depth_one_lists_immediate_children_only(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from ...asset.models import Depth
            from ...asset.path import AssetPath

            parent = AssetPath("/contract/depth-one")
            await store.put(parent.child("a"), b"a")
            await store.put(parent.child("b"), b"b")
            await store.put(parent.child("sub").child("c"), b"c")
            page = await store.propfind(parent, depth=Depth.ONE, limit=50)
            paths = {info.path.value for info in page.items}
            assert parent.child("a").value in paths
            assert parent.child("b").value in paths
            # Depth ONE excludes grandchildren.
            assert parent.child("sub").child("c").value not in paths

        self._contract_run(_run)

    def test_propfind_paginates_via_cursor(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from ...asset.models import Depth
            from ...asset.path import AssetPath

            parent = AssetPath("/contract/paginate")
            for i in range(5):
                await store.put(parent.child(f"f{i}"), b"x")
            first = await store.propfind(parent, depth=Depth.ONE, limit=2)
            assert len(first.items) == 2
            assert first.cursor is not None  # more pages available
            second = await store.propfind(
                parent, depth=Depth.ONE, limit=2, cursor=first.cursor
            )
            seen = {info.path.value for info in (*first.items, *second.items)}
            assert len(seen) == 4  # no overlap across the two pages

        self._contract_run(_run)


class EventStoreContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`EventStore` Protocol: append-only, and the
    store is the SOLE owner of sequence assignment. Callers pass the payload
    plus the run/stream context; the store mints event_id, sequence, and
    occurred_at atomically. The contract verifies that append mints the
    envelope fields, that two appends produce sequential sequence numbers,
    and that list returns an EventPage honoring after_sequence and limit.
    """

    def event_store(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def _append_kwargs() -> "dict[str, Any]":
        return dict(
            stream_id="s1",
            run_id="r1",
            root_run_id="r1",
            parent_run_id=None,
            session_id="sess-1",
            runnable_id="rn-1",
        )

    def test_append_mints_envelope_fields(self) -> None:
        store = self.event_store()

        async def _run() -> None:
            from ...events.payloads import RunStarted

            envelope = await store.append(
                payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                **self._append_kwargs(),
            )
            assert envelope.event_id  # non-empty uuid-string
            assert envelope.sequence >= 1
            assert envelope.occurred_at is not None
            assert isinstance(envelope.payload, RunStarted)

        self._contract_run(_run)

    def test_two_appends_mint_sequential_sequences(self) -> None:
        store = self.event_store()

        async def _run() -> None:
            from ...events.payloads import RunStarted

            a = await store.append(
                payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                **self._append_kwargs(),
            )
            b = await store.append(
                payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                **self._append_kwargs(),
            )
            assert b.sequence == a.sequence + 1

        self._contract_run(_run)

    def test_list_returns_events_in_append_order(self) -> None:
        store = self.event_store()

        async def _run() -> None:
            from ...events.payloads import RunStarted

            ids = []
            for _ in range(3):
                env = await store.append(
                    payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                    **self._append_kwargs(),
                )
                ids.append(env.event_id)
            page = await store.list("s1", limit=100)
            assert [e.event_id for e in page.items] == ids

        self._contract_run(_run)

    def test_list_after_sequence_filters_strictly(self) -> None:
        store = self.event_store()

        async def _run() -> None:
            from ...events.payloads import RunStarted

            for _ in range(4):
                await store.append(
                    payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                    **self._append_kwargs(),
                )
            page = await store.list("s1", after_sequence=2)
            assert [e.sequence for e in page.items] == [3, 4]

        self._contract_run(_run)

    def test_list_honors_limit_and_signals_more_via_cursor(self) -> None:
        store = self.event_store()

        async def _run() -> None:
            from ...events.payloads import RunStarted

            for _ in range(3):
                await store.append(
                    payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                    **self._append_kwargs(),
                )
            page = await store.list("s1", limit=2)
            assert len(page.items) == 2

        self._contract_run(_run)


class JobStoreContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`JobStore` Protocol: the reliable-task
    surface every backend (file, sqlalchemy, an external adapter) implements.
    Subclasses must implement ``job_store()`` returning a fresh, empty JobStore.
    The contract verifies create/read round-trip, claim-or-None semantics,
    fencing-token ownership (a wrong token is rejected), terminal-state
    transitions, and list_tasks filtering. It is backend-agnostic -- the only
    inputs are public JobStore, JobRecord, and TaskRecord shapes.
    """

    def job_store(self) -> Any:
        raise NotImplementedError

    @staticmethod
    def _now():
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)

    def _make_job(
        self, *, job_id: str = "j-contract", root_task_id: str = "t-contract-root"
    ) -> Any:
        from ...jobs.models import (
            ActorChain,
            ActorRef,
            JobRecord,
            JobStatus,
            TaskBudget,
            TaskPrincipal,
        )

        now = self._now()
        return JobRecord(
            id=job_id,
            status=JobStatus.PENDING,
            principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
            actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
            budget=TaskBudget(),
            root_task_id=root_task_id,
            input_artifact_id=None,
            output_artifact_id=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )

    def _make_root_task(
        self, *, job_id: str = "j-contract", task_id: str = "t-contract-root"
    ) -> Any:
        from ...jobs.models import (
            RetryPolicy,
            SideEffectPolicy,
            TaskRecord,
            TaskStatus,
        )

        now = self._now()
        return TaskRecord(
            id=task_id,
            job_id=job_id,
            parent_task_id=None,
            key="root",
            handler="runtime",
            status=TaskStatus.PENDING,
            input_artifact_id=None,
            output_artifact_id=None,
            dependencies=(),
            retry_policy=RetryPolicy(max_attempts=2),
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

    def test_create_job_round_trips_via_get_job_and_get_task(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            await store.create_job(
                self._make_job(), self._make_root_task()
            )
            job = await store.get_job("j-contract")
            assert job is not None and job.id == "j-contract"
            task = await store.get_task("t-contract-root")
            assert task is not None and task.id == "t-contract-root"

        self._contract_run(_run)

    def test_claim_returns_none_when_nothing_ready(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            claimed = await store.claim(
                worker_id="w-contract", now=self._now(), lease_seconds=30
            )
            assert claimed is None

        self._contract_run(_run)

    def test_claim_returns_claimed_task_when_ready(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            await store.create_job(
                self._make_job(), self._make_root_task()
            )
            claimed = await store.claim(
                worker_id="w-contract", now=self._now(), lease_seconds=30
            )
            assert claimed is not None
            assert claimed.claim.task_id == "t-contract-root"
            assert claimed.claim.worker_id == "w-contract"
            assert claimed.claim.fencing_token >= 1

        self._contract_run(_run)

    def test_renew_lease_with_valid_fencing_token_succeeds(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            await store.create_job(
                self._make_job(), self._make_root_task()
            )
            claimed = await store.claim(
                worker_id="w-contract", now=self._now(), lease_seconds=30
            )
            renewed = await store.renew_lease(
                task_id=claimed.claim.task_id,
                attempt_id=claimed.claim.attempt_id,
                worker_id=claimed.claim.worker_id,
                fencing_token=claimed.claim.fencing_token,
                now=self._now(),
                lease_seconds=30,
            )
            assert renewed.id == claimed.claim.task_id

        self._contract_run(_run)

    def test_renew_lease_with_wrong_fencing_token_is_rejected(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            from ...jobs.store import TaskClaimLostError

            await store.create_job(
                self._make_job(), self._make_root_task()
            )
            claimed = await store.claim(
                worker_id="w-contract", now=self._now(), lease_seconds=30
            )
            with __import__("pytest").raises(TaskClaimLostError):
                await store.renew_lease(
                    task_id=claimed.claim.task_id,
                    attempt_id=claimed.claim.attempt_id,
                    worker_id=claimed.claim.worker_id,
                    fencing_token=claimed.claim.fencing_token + 999,
                    now=self._now(),
                    lease_seconds=30,
                )

        self._contract_run(_run)

    def test_commit_success_transitions_task_to_succeeded(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            from ...jobs.models import TaskStatus
            from ...jobs.protocols import TaskSuccess

            await store.create_job(
                self._make_job(), self._make_root_task()
            )
            claimed = await store.claim(
                worker_id="w-contract", now=self._now(), lease_seconds=30
            )
            done = await store.commit_success(claimed.claim, TaskSuccess())
            assert done.status == TaskStatus.SUCCEEDED

        self._contract_run(_run)

    def test_list_tasks_filters_by_status(self) -> None:
        store = self.job_store()

        async def _run() -> None:
            from ...jobs.models import TaskStatus
            from ...jobs.protocols import TaskSuccess

            await store.create_job(
                self._make_job(), self._make_root_task()
            )
            claimed = await store.claim(
                worker_id="w-contract", now=self._now(), lease_seconds=30
            )
            await store.commit_success(claimed.claim, TaskSuccess())
            succeeded = await store.list_tasks(
                "j-contract", status=TaskStatus.SUCCEEDED
            )
            assert len(succeeded) == 1
            assert succeeded[0].status == TaskStatus.SUCCEEDED
            pending = await store.list_tasks(
                "j-contract", status=TaskStatus.PENDING
            )
            assert len(pending) == 0

        self._contract_run(_run)


class StorageTransactionManagerContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`StorageTransactionManager` Protocol.

    A subclass wires ``transaction_manager()`` to a fresh manager. The default
    ``is_supported()`` returns True -- the contract runs the supported-scope
    tests: a clean exit commits (yields a UnitOfWork), an exception inside the
    scope rolls back (no partial commit). For an unsupported-scope manager
    (``NoCrossStoreTransactions``), override ``is_supported()`` to return False
    -- the contract then verifies the unsupported path: ``transaction()``
    raises :class:`StorageTransactionNotSupportedError` AT THE CALL (not later
    inside ``__aenter__``), which is the honest failure mode. A fake success
    would let a caller think it had an atomic scope when it did not.
    """

    def transaction_manager(self) -> Any:
        raise NotImplementedError

    def is_supported(self) -> bool:
        """Return False if the manager does NOT support cross-store
        transactions (e.g. NoCrossStoreTransactions). The default True runs
        the supported-scope tests."""
        return True

    async def _write_inside_scope(self, uow: Any) -> None:
        """Hook: write something through ``uow`` so the rollback check has
        state to verify. Override when ``is_supported()`` is True and the
        backend exposes a known store on the UnitOfWork. Default: no-op
        (the rollback test then checks only that the exception propagates)."""
        return None

    async def _verify_rollback(self) -> None:
        """Hook: verify writes from ``_write_inside_scope`` did NOT persist
        after the scope rolled back. Default: no-op."""
        return None

    def test_transaction_yields_unit_of_work_on_clean_exit(self) -> None:
        if not self.is_supported():
            __import__("pytest").skip(
                "manager does not support cross-store transactions"
            )
        mgr = self.transaction_manager()

        async def _run() -> None:
            async with mgr.transaction() as uow:
                assert uow is not None

        self._contract_run(_run)

    def test_exception_inside_scope_propagates_and_rolls_back(self) -> None:
        if not self.is_supported():
            __import__("pytest").skip(
                "manager does not support cross-store transactions"
            )
        mgr = self.transaction_manager()

        async def _run() -> None:
            with __import__("pytest").raises(RuntimeError, match="rollback-test"):
                async with mgr.transaction() as uow:
                    await self._write_inside_scope(uow)
                    raise RuntimeError("rollback-test")
            await self._verify_rollback()

        self._contract_run(_run)

    def test_unsupported_scope_raises_at_the_call(self) -> None:
        if self.is_supported():
            __import__("pytest").skip(
                "manager supports cross-store transactions"
            )
        from ...errors import StorageTransactionNotSupportedError

        mgr = self.transaction_manager()
        # The error must surface when transaction() is CALLED, not later when
        # the async-with enters -- otherwise a caller would believe it had an
        # atomic scope when it did not.
        with __import__("pytest").raises(StorageTransactionNotSupportedError):
            mgr.transaction()


class LeaseCoordinatorContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`LeaseCoordinator` Protocol.

    The defining guarantees: acquire is mutually exclusive per key; renew keeps
    the same fencing token; a re-acquire after expiry yields a STRICTLY LARGER
    fencing token (monotonicity); release frees the key. A JobStore state
    commit checks the fencing token, so non-monotonic tokens are a correctness
    bug, not a performance one.
    """

    def coordinator(self) -> Any:
        raise NotImplementedError

    def test_acquire_is_mutually_exclusive(self) -> None:
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="k", owner_id="o1", ttl=timedelta(seconds=30))
            t2 = await coord.acquire(key="k", owner_id="o2", ttl=timedelta(seconds=30))
            assert t1 is not None
            assert t2 is None  # second acquirer loses while o1 holds the lease

        self._contract_run(_run)

    def test_renew_keeps_fencing_token_and_reacquire_increases_it(self) -> None:
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="k", owner_id="o1", ttl=timedelta(seconds=30))
            assert t1 is not None
            renewed = await coord.renew(token=t1, ttl=timedelta(seconds=30))
            assert renewed.fencing_token == t1.fencing_token  # renew is stable
            await coord.release(token=renewed)
            # Re-acquire after release must yield a STRICTLY LARGER token.
            t2 = await coord.acquire(key="k", owner_id="o1", ttl=timedelta(seconds=30))
            assert t2 is not None
            assert t2.fencing_token > t1.fencing_token

        self._contract_run(_run)

    def test_reacquire_after_expiry_increases_fencing_token(self) -> None:
        """A lease whose TTL has elapsed is reclaimable by another owner, and
        the new fencing token is STRICTLY LARGER than the expired one. This is
        the guarantee a JobStore state commit relies on to reject a stale write
        from a holder whose lease expired underneath it."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(
                key="k", owner_id="o1", ttl=timedelta(seconds=0)
            )
            assert t1 is not None
            # TTL 0 -> already expired; a different owner can reclaim and must
            # observe a larger fencing token than the stale holder saw.
            t2 = await coord.acquire(key="k", owner_id="o2", ttl=timedelta(seconds=30))
            assert t2 is not None
            assert t2.fencing_token > t1.fencing_token
            assert t2.owner_id == "o2"

        self._contract_run(_run)


async def _aiter(content: bytes):
    yield content


__all__: "list[str]" = [
    "ArtifactBlobStoreContract",
    "ArtifactRecordStoreContract",
    "AssetStoreContract",
    "EventStoreContract",
    "JobStoreContract",
    "LeaseCoordinatorContract",
    "StorageTransactionManagerContract",
]
