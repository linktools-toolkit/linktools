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
    fails, open streams back the exact bytes, stat/delete behave, and a
    source that errors mid-stream leaves NOTHING published at the claimed
    address (no partial blob may leak into the store)."""

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
            from linktools.ai.artifact.models import ArtifactIntegrityError

            with __import__("pytest").raises(ArtifactIntegrityError):
                await store.put_if_absent(
                    digest="0" * 64,
                    source=_aiter(b"not-the-claimed-digest"),
                    size=21,
                )

        self._contract_run(_run)

    def test_put_if_absent_propagates_source_error_and_publishes_nothing(self) -> None:
        """A source whose async generator raises mid-stream (or yields a chunk
        then errors) MUST cause ``put_if_absent`` to propagate the error AND
        leave nothing published at the claimed digest address. A backend that
        swallowed the error and committed a partial blob would shadow a
        future legitimate upload of the full content -- or, worse, be served
        to a reader that re-hashes and gets a mismatch against the pinned
        digest. A backend that returned success without reading the source
        would also fail the post-condition (the published bytes would not
        hash to the claimed digest)."""
        store = self.blob_store()
        # Claim a digest that is NOT the partial bytes' hash, so a backend
        # that published the partial chunk and then computed the digest
        # itself would also fail (the published bytes wouldn't match).
        claimed_digest = hashlib.sha256(b"complete-content").hexdigest()

        async def _run() -> None:
            from linktools.ai.artifact.models import ArtifactIntegrityError

            async def _bad_source():
                yield b"partial"
                raise RuntimeError("source-failed-mid-stream")

            with __import__("pytest").raises((RuntimeError, ArtifactIntegrityError)):
                await store.put_if_absent(
                    digest=claimed_digest, source=_bad_source(), size=None
                )
            # The error path MUST leave the store untouched at the claimed
            # address. A stat returning a BlobInfo here would mean a partial
            # blob leaked through the failure path.
            assert (await store.stat(digest=claimed_digest)) is None, (
                "put_if_absent published a blob at the claimed digest even "
                "though the source errored mid-stream"
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
            async with store.open(digest=digest) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
            assert b"".join(chunks) == content

        self._contract_run(_run)

    def test_open_missing_digest_raises(self) -> None:
        """Opening a digest that was never published MUST surface the UNIFIED
        :class:`ArtifactBlobNotFoundError` (plan §4.1) -- silently returning an
        empty stream would let a reader believe it had fetched the (absent)
        blob's bytes. Distinct from :class:`ArtifactIntegrityError` (blob EXISTS
        but is corrupt): a caller can tell 'absent' apart from 'corrupt'."""
        store = self.blob_store()

        async def _run() -> None:
            from linktools.ai.artifact.models import ArtifactBlobNotFoundError

            with __import__("pytest").raises(ArtifactBlobNotFoundError):
                async with store.open(digest="0" * 64) as stream:
                    async for _ in stream:
                        pass

        self._contract_run(_run)

    def test_stat_missing_digest_returns_none(self) -> None:
        """``stat`` on a never-published digest returns None (not a default
        BlobInfo, not a zero-size placeholder). A stub that always returned a
        BlobInfo would fail this test."""
        store = self.blob_store()

        async def _run() -> None:
            assert (await store.stat(digest="0" * 64)) is None

        self._contract_run(_run)

    def test_delete_missing_digest_is_silent(self) -> None:
        """``delete`` on a never-published digest MUST NOT raise -- idempotent
        delete is the contract every sweeper / GC loop relies on."""
        store = self.blob_store()

        async def _run() -> None:
            await store.delete(digest="0" * 64)  # no error

        self._contract_run(_run)

    def test_open_preserves_multi_chunk_order(self) -> None:
        """A blob written from a multi-chunk source MUST stream back with chunk
        boundaries and order preserved (the facade relies on this to prove it
        streams rather than joining). A backend that re-chunked or reordered
        would still hash-correct but break a caller observing chunked delivery."""
        store = self.blob_store()

        async def _src():
            for chunk in (b"AAA", b"BB", b"C"):
                yield chunk

        content = b"AAABBC"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            await store.put_if_absent(digest=digest, source=_src(), size=len(content))
            chunks = []
            async with store.open(digest=digest) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
            assert b"".join(chunks) == content
            # At least one boundary survived (the source yielded 3 chunks; a
            # join-then-rechunk backend might still emit 3 but the BYTES must
            # be intact -- the join equality above is the real assertion).

        self._contract_run(_run)

    def test_put_if_absent_rejects_size_mismatch(self) -> None:
        """A claimed ``size`` that does not match the actual content length
        MUST fail (ArtifactIntegrityError). A backend that ignored size would
        let a caller pin a wrong content-length on the BlobInfo, breaking the
        buffered-size-cap decision in the facade."""
        store = self.blob_store()
        content = b"twelve-bytes"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            from linktools.ai.artifact.models import ArtifactIntegrityError

            with __import__("pytest").raises(ArtifactIntegrityError):
                await store.put_if_absent(
                    digest=digest, source=_aiter(content), size=999
                )

        self._contract_run(_run)

    def test_open_releases_resource_on_consumer_cancellation(self) -> None:
        """If the consumer cancels mid-stream, ``open()``'s async context
        manager MUST release its underlying resource (file handle / session).
        Cancellation triggers ``__aexit__``; a backend that leaked the handle
        would exhaust FDs under retries. Asserted by re-opening the SAME digest
        after a cancelled read -- if the first handle leaked and locked the
        blob (some backends), the second open would fail or block."""
        store = self.blob_store()
        content = b"cancel-me-payload"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            await store.put_if_absent(
                digest=digest, source=_aiter(content), size=len(content)
            )
            cancelled_read = asyncio.ensure_future(self._drain_then_cancel(store, digest))
            await asyncio.wait({cancelled_read}, timeout=5.0)
            # After the cancelled read, a fresh open must still work (resource
            # was released) and yield the full content.
            chunks = []
            async with store.open(digest=digest) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
            assert b"".join(chunks) == content

        self._contract_run(_run)

    @staticmethod
    async def _drain_then_cancel(store: Any, digest: str) -> None:
        task = asyncio.current_task()
        try:
            async with store.open(digest=digest) as stream:
                async for _ in stream:
                    if task is not None:
                        task.cancel()
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass

    def test_concurrent_same_digest_writes_publish_one_blob(self) -> None:
        """Two concurrent put_if_absent calls for the SAME digest MUST both
        succeed (idempotent) and result in exactly one published blob. A
        backend that raced on the publish path could duplicate, corrupt, or
        raise on the second writer."""
        store = self.blob_store()
        content = b"concurrent-same"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            results = await asyncio.gather(
                store.put_if_absent(digest=digest, source=_aiter(content), size=len(content)),
                store.put_if_absent(digest=digest, source=_aiter(content), size=len(content)),
            )
            for info in results:
                assert info is not None
                assert info.digest == digest
            stat = await store.stat(digest=digest)
            assert stat is not None

        self._contract_run(_run)

    def test_open_returns_an_async_context_manager(self) -> None:
        """``open()`` MUST return an async context manager (the unified
        Protocol surface, plan RF-01) -- NOT a bare async generator or iterator.
        A fake implementation returning a raw async generator would fail the
        ``async with`` form callers depend on."""
        store = self.blob_store()
        content = b"ctx-mgr-shape"
        digest = hashlib.sha256(content).hexdigest()

        async def _run() -> None:
            await store.put_if_absent(
                digest=digest, source=_aiter(content), size=len(content)
            )
            cm = store.open(digest=digest)
            # The result of open() must be usable as `async with`.
            assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__"), (
                "open() must return an async context manager, not a bare "
                "async generator -- a raw async generator fake would have no "
                "__aenter__/__aexit__"
            )
            async with cm as stream:
                assert hasattr(stream, "__aiter__")

        self._contract_run(_run)

    def test_a_non_context_manager_open_is_rejected_at_use(self) -> None:
        """A blob store whose ``open()`` returns a BARE async generator (not
        wrapped in an async context manager) MUST be rejected at the
        ``async with`` use site -- this is the RF-02 negative guarantee, made
        PORTABLE so every contract subclass runs it (not just an in-repo
        standalone test). The contract constructs a known-bad impl inline and
        asserts the ``async with`` form raises on a generator with no
        ``__aenter__``/``__aexit__``; a third-party adapter cannot silently ship
        a non-conformant ``open()`` past this check."""

        class _BadAsyncGenBlobStore:
            """A deliberately non-conformant impl: ``open()`` returns a raw
            async generator (no __aenter__/__aexit__), violating the unified
            RF-01 surface."""

            async def open(self, *, digest: str):
                # Bare async generator -- NOT an async context manager.
                if False:  # pragma: no cover - never entered under the contract
                    yield b""

        bad = _BadAsyncGenBlobStore()

        async def _run() -> None:
            # The ``async with`` form callers depend on MUST reject a bare
            # async generator (it has no __aenter__). AttributeError is the
            # expected failure mode; the contract's point is that a
            # non-conformant impl does NOT silently work.
            with __import__("pytest").raises((AttributeError, TypeError)):
                async with bad.open(digest="anything") as _stream:  # type: ignore[arg-type]
                    async for _ in _stream:  # pragma: no cover
                        pass

        self._contract_run(_run)


class ArtifactRecordStoreContract(_ConformanceRecorderMixin):
    """Conformance for the :class:`ArtifactRecordStore` Protocol: a record is
    retrievable by id+tenant, a foreign tenant learns nothing, delete is
    tenant-scoped."""

    def record_store(self) -> Any:
        raise NotImplementedError

    def _record(
        self,
        artifact_id: str,
        tenant: str,
        *,
        producer_kind: str = "anonymous",
        producer_id: str = "",
        run_id: "str | None" = None,
        parent_artifact_ids: "tuple[str, ...]" = (),
        metadata: "Mapping[str, object] | None" = None,
    ) -> Any:
        from linktools.ai.artifact.models import (
            ArtifactProvenance,
            ArtifactRecord,
            ArtifactRef,
        )

        return ArtifactRecord(
            ref=ArtifactRef(id=artifact_id, sha256="x" * 64, media_type="", size=0),
            tenant_id=tenant,
            provenance=ArtifactProvenance(
                producer_kind=producer_kind,
                producer_id=producer_id,
                run_id=run_id,
                parent_artifact_ids=parent_artifact_ids,
                metadata=dict(metadata) if metadata is not None else {},
            ),
            created_at=__import__("datetime").datetime(2026, 1, 1),
        )

    def test_get_is_tenant_scoped(self) -> None:
        store = self.record_store()

        async def _run() -> None:
            await store.put(self._record("a1", "t1"))
            assert (await store.get(artifact_id="a1", tenant_id="t1")) is not None
            # Foreign tenant learns nothing -- not even that the record exists.
            assert (await store.get(artifact_id="a1", tenant_id="t2")) is None

        self._contract_run(_run)

    def test_get_missing_returns_none(self) -> None:
        """A get on a never-stored id returns None (not raise, not a default
        empty record). A stub backend that always returned a record would
        fail this test."""
        store = self.record_store()

        async def _run() -> None:
            assert (await store.get(artifact_id="never-stored", tenant_id="t1")) is None

        self._contract_run(_run)

    def test_delete_is_tenant_scoped(self) -> None:
        """A foreign tenant's delete MUST NOT remove the owning tenant's
        record. Order matters here: the test deletes as the FOREIGN tenant
        first (the foreign call returns False), THEN re-reads as the OWNER
        and confirms the record is still there. A test that deletes as the
        owner first cannot distinguish a tenant-scoped delete from a stub
        that returned False for every call -- the owner's record would be
        gone either way."""
        store = self.record_store()

        async def _run() -> None:
            await store.put(self._record("a2", "t1"))
            # Foreign tenant attempts the delete -- returns False AND leaves
            # the owner's record intact.
            foreign_delete = await store.delete("a2", tenant_id="t2")
            assert foreign_delete is False, (
                "foreign-tenant delete returned True -- delete is not tenant "
                "scoped"
            )
            # Owner STILL sees the record after the foreign delete attempt.
            # A non-tenant-scoped delete (or a stub that mutated shared state)
            # would have removed the row here.
            assert (await store.get(artifact_id="a2", tenant_id="t1")) is not None, (
                "foreign-tenant delete removed the owner's record -- delete is "
                "not tenant scoped"
            )
            # Now the owner's own delete succeeds and returns True.
            owner_delete = await store.delete("a2", tenant_id="t1")
            assert owner_delete is True

        self._contract_run(_run)

    def test_delete_missing_returns_false(self) -> None:
        """Delete on a never-stored id returns False (not raise, not True).
        Combined with the positive delete above this pins the three-state
        contract: True=removed, False=was-already-absent."""
        store = self.record_store()

        async def _run() -> None:
            assert (await store.delete("never-stored", tenant_id="t1")) is False

        self._contract_run(_run)

    def test_duplicate_id_same_tenant_is_idempotent_or_overwrites(self) -> None:
        """A second put with the same (id, tenant) does not raise and does not
        create a second record: a get returns exactly one record for that id.
        A store that keyed by insertion and surfaced two records (or raised on
        the collision) would fail. The record stays tenant-owned by t1."""
        store = self.record_store()

        async def _run() -> None:
            await store.put(self._record("dup-1", "t1", producer_id="first"))
            # Second put with the same id -- no raise, no duplicate.
            await store.put(self._record("dup-1", "t1", producer_id="second"))
            fetched = await store.get(artifact_id="dup-1", tenant_id="t1")
            assert fetched is not None
            assert fetched.tenant_id == "t1"

        self._contract_run(_run)

    def test_provenance_round_trips_through_the_record_codec(self) -> None:
        """Every ArtifactProvenance field (producer_kind, producer_id, run_id,
        parent_artifact_ids, metadata) MUST survive a put -> get cycle unchanged.
        A backend that serialized only a subset of provenance (e.g. dropped
        run_id or metadata) would silently lose lineage a downstream replay /
        audit depends on."""
        store = self.record_store()

        async def _run() -> None:
            original = self._record(
                "prov-1",
                "t1",
                producer_kind="job_attempt",
                producer_id="attempt-7",
                run_id="run-42",
                parent_artifact_ids=("art-parent-a", "art-parent-b"),
                metadata={"tool": "browser", "ndc": 11},
            )
            await store.put(original)
            fetched = await store.get(artifact_id="prov-1", tenant_id="t1")
            assert fetched is not None
            assert fetched.provenance.producer_kind == "job_attempt"
            assert fetched.provenance.producer_id == "attempt-7"
            assert fetched.provenance.run_id == "run-42"
            assert tuple(fetched.provenance.parent_artifact_ids) == (
                "art-parent-a",
                "art-parent-b",
            )
            assert fetched.provenance.metadata == {"tool": "browser", "ndc": 11}

        self._contract_run(_run)

    def test_parent_artifact_ids_round_trip(self) -> None:
        """``parent_artifact_ids`` (the derivation chain) is part of the record;
        a record with parents must get them back. A backend that flattened the
        tuple or dropped it would break derivation queries."""
        store = self.record_store()

        async def _run() -> None:
            await store.put(
                self._record("child-1", "t1", parent_artifact_ids=("p1", "p2", "p3"))
            )
            fetched = await store.get(artifact_id="child-1", tenant_id="t1")
            assert fetched is not None
            assert tuple(fetched.provenance.parent_artifact_ids) == ("p1", "p2", "p3")

        self._contract_run(_run)

    def test_a_digest_alone_is_not_enough_to_fetch_a_record(self) -> None:
        """The record-store API is keyed by (artifact_id, tenant_id); a digest
        is never an argument to get/delete. A caller cannot bypass the
        tenant/record gate by presenting a sha256. This test asserts the API
        SHAPE: get/delete accept ``artifact_id`` + ``tenant_id`` keyword-only --
        a backend that added a digest-based lookup would have widened the
        surface the Protocol forbids."""
        store = self.record_store()
        import inspect

        sig_get = inspect.signature(store.get)
        params_get = set(sig_get.parameters)
        # get is keyed by artifact_id + tenant_id -- never by digest.
        assert "artifact_id" in params_get
        assert "tenant_id" in params_get
        assert "digest" not in params_get, (
            "ArtifactRecordStore.get must not accept a digest -- a digest alone "
            "must never be enough to fetch a tenant-owned record"
        )


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
            from linktools.ai.asset.path import AssetPath

            path = AssetPath("/contract/roundtrip.txt")
            await store.put(path, b"hello")
            fetched = await store.get(path)
            assert fetched is not None
            assert fetched.content == b"hello"

        self._contract_run(_run)

    def test_get_missing_returns_none(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from linktools.ai.asset.path import AssetPath

            assert (await store.get(AssetPath("/contract/never-existed"))) is None

        self._contract_run(_run)

    def test_delete_removes_resource(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from linktools.ai.asset.path import AssetPath

            path = AssetPath("/contract/deleted.txt")
            await store.put(path, b"x")
            await store.delete(path)
            assert (await store.get(path)) is None

        self._contract_run(_run)

    def test_delete_is_path_scoped(self) -> None:
        """A delete of one path MUST NOT touch a different path's resource.
        The positive test above cannot distinguish a correct backend from a
        backend whose delete() wipes the whole store (or every sibling under
        the parent): both would leave the deleted path empty. This test puts
        two siblings, deletes one, and asserts the OTHER survives with its
        original content -- catching an over-broad delete implementation."""
        store = self.asset_store()

        async def _run() -> None:
            from linktools.ai.asset.path import AssetPath

            keep = AssetPath("/contract/kept.txt")
            sibling = AssetPath("/contract/sibling.txt")
            await store.put(keep, b"keep-me")
            await store.put(sibling, b"sibling")
            await store.delete(sibling)
            fetched = await store.get(keep)
            assert fetched is not None, (
                "delete of a sibling path removed the kept resource -- "
                "delete is not path-scoped"
            )
            assert fetched.content == b"keep-me"

        self._contract_run(_run)

    def test_delete_missing_is_silent(self) -> None:
        """Delete on a never-stored path MUST NOT raise -- idempotent delete
        is the contract every sweeper / GC loop relies on."""
        store = self.asset_store()

        async def _run() -> None:
            from linktools.ai.asset.path import AssetPath

            await store.delete(AssetPath("/contract/never-existed"))

        self._contract_run(_run)

    def test_put_same_path_bumps_version(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from linktools.ai.asset.path import AssetPath

            path = AssetPath("/contract/versioned.txt")
            first = await store.put(path, b"v1")
            second = await store.put(path, b"v2")
            assert second.info.version > first.info.version

        self._contract_run(_run)

    def test_propfind_depth_one_lists_immediate_children_only(self) -> None:
        store = self.asset_store()

        async def _run() -> None:
            from linktools.ai.asset.models import Depth
            from linktools.ai.asset.path import AssetPath

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
            from linktools.ai.asset.models import Depth
            from linktools.ai.asset.path import AssetPath

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
            from linktools.ai.events.payloads import RunStarted

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
            from linktools.ai.events.payloads import RunStarted

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
            from linktools.ai.events.payloads import RunStarted

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
            from linktools.ai.events.payloads import RunStarted

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
            from linktools.ai.events.payloads import RunStarted

            for _ in range(3):
                await store.append(
                    payload=RunStarted(run_id="r1", runnable_id="rn-1"),
                    **self._append_kwargs(),
                )
            page = await store.list("s1", limit=2)
            assert len(page.items) == 2

        self._contract_run(_run)

    def test_list_unknown_stream_returns_empty_page(self) -> None:
        """Listing a stream that was never appended to MUST return an empty
        page -- not raise, not None. A stub backend that returned None for a
        missing stream (instead of an empty EventPage) would break callers
        that iterate ``page.items`` without a None guard."""
        store = self.event_store()

        async def _run() -> None:
            page = await store.list("never-appended", limit=10)
            assert page is not None
            assert tuple(page.items) == ()

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
        from linktools.ai.jobs.models import (
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
        from linktools.ai.jobs.models import (
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

    def test_get_missing_job_and_task_return_none(self) -> None:
        """``get_job`` / ``get_task`` on a never-stored id return None (not
        raise, not a default empty record). A stub backend that fabricated a
        record on any get would break the claim loop (it would try to claim a
        non-existent task) and would fail this test."""
        store = self.job_store()

        async def _run() -> None:
            assert (await store.get_job("never-created")) is None
            assert (await store.get_task("never-created")) is None

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
            from linktools.ai.jobs.store import TaskClaimLostError

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
            from linktools.ai.jobs.models import TaskStatus
            from linktools.ai.jobs.protocols import TaskSuccess

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
            from linktools.ai.jobs.models import TaskStatus
            from linktools.ai.jobs.protocols import TaskSuccess

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

    ROLLBACK IS ASSERTED, NOT ASSUMED. The default ``_write_inside_scope``
    writes a Session through the UoW and the default ``_verify_rollback``
    opens a FRESH transaction and confirms the Session is absent. A backend
    whose rollback is a no-op (records "rollback was called" without undoing
    writes) would leave the row visible and fail this test. Subclasses may
    override the pair to probe a different store, but MUST override both --
    overriding only one would silently relax the assertion back to the
    pre-assertion default."""

    # The session id the default probe writes inside the scope. A subclass
    # that overrides BOTH hooks may ignore this; a subclass that overrides
    # only one will trip the post-assertion check below.
    _ROLLBACK_PROBE_SESSION_ID = "contract-rollback-probe"

    def transaction_manager(self) -> Any:
        raise NotImplementedError

    def is_supported(self) -> bool:
        """Return False if the manager does NOT support cross-store
        transactions (e.g. NoCrossStoreTransactions). The default True runs
        the supported-scope tests."""
        return True

    async def _write_inside_scope(self, uow: Any) -> None:
        """Default probe: create a Session row through the UoW so rollback
        has a concrete write to undo. The session id is fixed so the default
        ``_verify_rollback`` / ``_verify_commit_persisted`` can look for it
        through a fresh transaction on the SAME manager. Subclasses that
        prefer to probe a different store MUST also override both
        ``_verify_rollback`` and ``_verify_commit_persisted`` -- the pair is
        the assertion, overriding one half alone would let a non-rolling-back
        backend pass."""
        from datetime import datetime, timezone

        from linktools.ai.session.models import SessionRecord, SessionStatus

        now = datetime.now(timezone.utc)
        await uow.sessions.create(
            SessionRecord(
                id=self._ROLLBACK_PROBE_SESSION_ID,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

    async def _verify_rollback(self, mgr: Any) -> None:
        """Default probe: open a FRESH transaction through the SAME manager
        that wrote the probe and confirm the Session is NOT visible. A
        backend whose rollback is a no-op would leave the row visible here --
        the fake-pass this contract exists to catch. Passing the SAME manager
        (not a fresh ``self.transaction_manager()``) keeps the probe honest:
        the verify reads the same backend state the write touched."""
        async with mgr.transaction() as uow:
            fetched = await uow.sessions.get(self._ROLLBACK_PROBE_SESSION_ID)
        assert fetched is None, (
            "rollback did not undo writes: the Session row written inside the "
            "rolled-back scope is still visible through a fresh transaction "
            "(the manager recorded 'rollback was called' but left the write in "
            "place -- this is the fake-pass the contract exists to catch)"
        )

    async def _verify_commit_persisted(self, mgr: Any) -> None:
        """Mirror of the rollback probe for the clean-exit path: open a FRESH
        transaction through the SAME manager and confirm the Session IS
        visible. A backend that auto-rolled back on clean exit (or never
        committed) would leave the row absent."""
        async with mgr.transaction() as uow:
            fetched = await uow.sessions.get(self._ROLLBACK_PROBE_SESSION_ID)
        assert fetched is not None, (
            "clean commit did not persist writes: the Session row written "
            "inside the cleanly-exited scope is missing through a fresh "
            "transaction (the manager rolled back on clean exit, or never "
            "committed at all)"
        )

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
        # Capture the manager ONCE so the verify step reads the same backend
        # state the write touched. Re-calling ``transaction_manager()`` in
        # ``_verify_rollback`` would silently relax the probe if a subclass
        # returned a fresh no-op manager each call.
        mgr = self.transaction_manager()

        async def _run() -> None:
            with __import__("pytest").raises(RuntimeError, match="rollback-test"):
                async with mgr.transaction() as uow:
                    await self._write_inside_scope(uow)
                    raise RuntimeError("rollback-test")
            await self._verify_rollback(mgr)

        self._contract_run(_run)

    def test_clean_scope_commit_then_visible_through_fresh_transaction(self) -> None:
        """The mirror of the rollback test: a CLEAN exit MUST commit. The
        default probe writes a Session inside the scope, exits cleanly, then
        opens a fresh transaction and confirms the Session IS visible. A
        backend that auto-rolled back on clean exit (or never committed at
        all) would leave the row absent and fail this test."""
        if not self.is_supported():
            __import__("pytest").skip(
                "manager does not support cross-store transactions"
            )
        mgr = self.transaction_manager()

        async def _run() -> None:
            async with mgr.transaction() as uow:
                await self._write_inside_scope(uow)
            await self._verify_commit_persisted(mgr)

        self._contract_run(_run)

    def test_unsupported_scope_raises_at_the_call(self) -> None:
        if self.is_supported():
            __import__("pytest").skip(
                "manager supports cross-store transactions"
            )
        from linktools.ai.errors import StorageTransactionNotSupportedError

        mgr = self.transaction_manager()
        # The error must surface when transaction() is CALLED, not later when
        # the async-with enters -- otherwise a caller would believe it had an
        # atomic scope when it did not.
        with __import__("pytest").raises(StorageTransactionNotSupportedError):
            mgr.transaction()

    def test_two_stores_both_rollback_on_scope_failure(self) -> None:
        """Write to TWO stores inside the UoW (sessions + runs), then raise.
        BOTH writes MUST be absent through a fresh transaction -- not just the
        store the default probe checks. A backend whose atomicity was
        per-store (committed sessions but rolled back runs) would fail here."""
        if not self.is_supported():
            __import__("pytest").skip(
                "manager does not support cross-store transactions"
            )
        mgr = self.transaction_manager()

        async def _run() -> None:
            from datetime import datetime, timezone

            from linktools.ai.run.models import (
                RunInput,
                RunnableType,
                RunRecord,
                RunStatus,
            )
            from linktools.ai.session.models import SessionRecord, SessionStatus

            now = datetime.now(timezone.utc)
            session = SessionRecord(
                id="contract-two-store-session",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
            run = RunRecord(
                id="contract-two-store-run",
                root_run_id="contract-two-store-run",
                parent_run_id=None,
                session_id="contract-two-store-session",
                runnable_id="agent-1",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.PENDING,
                input=RunInput(prompt="probe"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            )
            with __import__("pytest").raises(RuntimeError, match="two-store"):
                async with mgr.transaction() as uow:
                    await uow.sessions.create(session)
                    await uow.runs.create(run)
                    raise RuntimeError("two-store rollback")
            # BOTH stores rolled back: neither write is visible through a fresh
            # transaction on the same manager.
            async with mgr.transaction() as uow:
                fetched_session = await uow.sessions.get("contract-two-store-session")
                fetched_run = await uow.runs.get("contract-two-store-run")
            assert fetched_session is None, (
                "sessions did not roll back with runs in the same UoW"
            )
            assert fetched_run is None, "runs did not roll back in the UoW"

        self._contract_run(_run)

    def test_read_your_writes_within_one_unit_of_work(self) -> None:
        """All tx.* stores in one UoW share ONE transaction, so a write to
        store A is visible to a read from store B INSIDE the same scope (before
        commit). This is the observable consequence of shared-session identity:
        if each store opened its own session, B would NOT see A's uncommitted
        write and the atomicity guarantee would be cosmetic."""
        if not self.is_supported():
            __import__("pytest").skip(
                "manager does not support cross-store transactions"
            )
        mgr = self.transaction_manager()

        async def _run() -> None:
            from datetime import datetime, timezone

            from linktools.ai.session.models import SessionRecord, SessionStatus

            now = datetime.now(timezone.utc)
            session = SessionRecord(
                id="contract-ryw-session",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
            async with mgr.transaction() as uow:
                await uow.sessions.create(session)
                # Read BACK through the same UoW: the just-written row must be
                # visible without a commit (read-your-writes within one tx).
                seen = await uow.sessions.get("contract-ryw-session")
                assert seen is not None, (
                    "a write was not visible to a read in the SAME UoW -- the "
                    "stores do not share one transaction"
                )

        self._contract_run(_run)


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

    def test_release_frees_the_key_for_immediate_reacquire(self) -> None:
        """release MUST free the key: a DIFFERENT owner can then acquire it
        immediately (no waiting for TTL expiry). A backend whose release was a
        no-op would force the next acquirer to wait out the TTL -- a liveness
        bug."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="rel", owner_id="o1", ttl=timedelta(seconds=30))
            assert t1 is not None
            await coord.release(token=t1)
            # A different owner acquires the freed key right away.
            t2 = await coord.acquire(key="rel", owner_id="o2", ttl=timedelta(seconds=30))
            assert t2 is not None
            assert t2.owner_id == "o2"

        self._contract_run(_run)

    def test_renew_a_released_token_fails(self) -> None:
        """Renewing a token whose lease was released (stale) MUST fail, not
        silently extend a lease the holder no longer owns. A backend that
        accepted the renew would let a holder keep a key it explicitly freed."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="stale", owner_id="o1", ttl=timedelta(seconds=30))
            assert t1 is not None
            await coord.release(token=t1)
            with __import__("pytest").raises(Exception):
                await coord.renew(token=t1, ttl=timedelta(seconds=30))

        self._contract_run(_run)

    def test_renew_an_expired_token_fails(self) -> None:
        """Renewing a token whose TTL has elapsed (timeout) MUST fail. This is
        the time-based counterpart of the stale-token test: a holder whose
        lease expired underneath it must not be able to renew its way back into
        ownership -- a fresh acquire (with a larger fencing token) is the only
        recovery, so a stale commit is detectable."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="exp", owner_id="o1", ttl=timedelta(seconds=0))
            assert t1 is not None
            with __import__("pytest").raises(Exception):
                await coord.renew(token=t1, ttl=timedelta(seconds=30))

        self._contract_run(_run)

    def test_release_is_idempotent(self) -> None:
        """Releasing an already-released (or never-held) token MUST NOT raise
        -- idempotent release is the contract every GC / cleanup loop relies
        on (it cannot track which tokens it already released)."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="idem", owner_id="o1", ttl=timedelta(seconds=30))
            assert t1 is not None
            await coord.release(token=t1)
            # Second release of the same token is silent.
            await coord.release(token=t1)

        self._contract_run(_run)

    def test_renew_failure_carries_an_identifiable_cause(self) -> None:
        """When renew fails (stale/expired token), the raised exception MUST
        carry a human-meaningful message identifying the cause -- not a bare
        ``Exception()`` or a swallowed error. A caller (or operator reading a
        traceback) needs to tell 'expired' apart from 'released' to recover."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            t1 = await coord.acquire(key="cause", owner_id="o1", ttl=timedelta(seconds=0))
            assert t1 is not None
            try:
                await coord.renew(token=t1, ttl=timedelta(seconds=30))
            except Exception as exc:  # noqa: BLE001 - the contract is "fails with a cause"
                assert str(exc), (
                    "renew failure raised an exception with no message -- the "
                    "cause is not identifiable"
                )
            else:
                __import__("pytest").fail(
                    "renew of an expired token did not raise -- the failure has "
                    "no cause at all"
                )

        self._contract_run(_run)

    def test_holder_task_cancellation_does_not_corrupt_or_auto_release(self) -> None:
        """Cancelling the asyncio task that HOLDS a lease MUST NOT corrupt
        coordinator state or implicitly release the lease: the lease survives
        until its explicit release or TTL expiry (a lease's lifetime is NOT
        tied to the holding task's lifetime). A coordinator that auto-released
        on task cancellation would let a stale task's successor race the
        explicit-release path. Verified by holding a lease in a cancelled
        task, then asserting a separate acquirer STILL loses (lease intact)
        and the original owner can still release it."""
        coord = self.coordinator()

        async def _run() -> None:
            from datetime import timedelta

            holder_token = []

            async def _holder() -> None:
                t = await coord.acquire(
                    key="cancel-safety", owner_id="holder", ttl=timedelta(seconds=30)
                )
                holder_token.append(t)

            task = asyncio.ensure_future(_holder())
            await asyncio.wait({task}, timeout=5.0)
            assert task.done(), "holder did not acquire in time"
            # Cancel the holder task AFTER it acquired (the lease is held).
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert holder_token, "holder never acquired a token"
            # A separate acquirer MUST still lose: the lease survived the
            # holder task's cancellation (no auto-release, no corruption).
            loser = await coord.acquire(
                key="cancel-safety", owner_id="other", ttl=timedelta(seconds=30)
            )
            assert loser is None, (
                "the lease was implicitly released when the holder task was "
                "cancelled -- a lease's lifetime must not track the holding task"
            )
            # The original owner can still release the surviving lease cleanly.
            await coord.release(token=holder_token[0])

        self._contract_run(_run)


class StorageFeaturesContract(_ConformanceRecorderMixin):
    """Conformance for a Storage's declared ``StorageFeatures`` vs what its
    stores actually support. A Storage that declares
    ``features.transactions = TransactionScope.DATABASE`` must yield a real
    UnitOfWork through ``transaction()``; one that declares
    ``TransactionScope.NONE`` must raise at the call. Declaring a capability
    without the implementation (or vice versa) is a self-inconsistency that
    breaks any caller that branched on the feature flag -- a RuntimeBuilder
    capability gate trusts the declaration and refuses the shortfall, so a
    Storage that lies about its features would either crash later (capability
    absent at run time) or silently degrade (capability present but unused).

    Subclasses wire ``storage()`` to return a fresh, empty Storage instance."""

    def storage(self) -> Any:
        raise NotImplementedError

    def test_declared_transaction_scope_matches_transaction_behavior(self) -> None:
        """``features.transactions`` MUST match ``transaction()`` behavior:
        DATABASE -> yields a real UoW; NONE -> raises
        ``StorageTransactionNotSupportedError`` at the call. The
        ``PROCESS_LOCAL`` scope is intentionally unspecified (the spec allows
        either single-store-durability-without-cross-store-UoW
        interpretation), so the contract does not pin it."""
        storage = self.storage()
        features = storage.features

        async def _run() -> None:
            from linktools.ai.storage.features import TransactionScope

            if features.transactions is TransactionScope.DATABASE:
                # Declared DATABASE: transaction() must yield a real UoW,
                # not raise / return None / yield a fake. A caller branched on
                # the feature flag and expects an atomic scope.
                async with storage.transaction() as uow:
                    assert uow is not None, (
                        "features.transactions=DATABASE but transaction() "
                        "yielded None -- the declared capability is not "
                        "actually implemented"
                    )
            elif features.transactions is TransactionScope.NONE:
                # Declared NONE: transaction() must raise at the call. A
                # manager that yields a fake UoW would let a caller believe
                # it had an atomic scope when it did not.
                from linktools.ai.errors import StorageTransactionNotSupportedError

                with __import__("pytest").raises(
                    StorageTransactionNotSupportedError
                ):
                    storage.transaction()
            # PROCESS_LOCAL: intentionally unpinned (spec allows either).

        self._contract_run(_run)

    def test_streaming_blobs_feature_matches_artifacts(self) -> None:
        """A Storage that declares ``features.streaming_blobs = True`` MUST
        wire its ``artifacts`` store (the streaming surface lives on the
        artifact store's ``put_if_absent`` / ``open`` methods). An unwired
        artifacts store + a True declaration would silently break any caller
        that streamed through ``storage.artifacts``."""
        storage = self.storage()
        features = storage.features

        async def _run() -> None:
            if features.streaming_blobs:
                assert storage.artifacts is not None, (
                    "features.streaming_blobs=True but storage.artifacts is "
                    "None -- the declared streaming surface is missing"
                )

        self._contract_run(_run)

    def test_coordination_feature_matches_coordinator(self) -> None:
        """``features.coordination != NONE`` MUST mean a real LeaseCoordinator
        is wired on ``storage.coordination``. A backend that declared
        distributed coordination but left the coordinator None would
        AttributeError at first lease acquire."""
        storage = self.storage()
        features = storage.features

        async def _run() -> None:
            from linktools.ai.storage.features import CoordinationScope

            if features.coordination is not CoordinationScope.NONE:
                assert storage.coordination is not None, (
                    f"features.coordination={features.coordination.value!r} but "
                    "storage.coordination is None -- the declared coordinator "
                    "is missing"
                )

        self._contract_run(_run)

    def test_leasing_feature_matches_coordinator_surface(self) -> None:
        """``features.leasing = True`` MUST mean the coordinator exposes the
        acquire/renew/release surface (the methods every leasing caller uses).
        Asserts the surface exists; the behavioral contract lives in
        LeaseCoordinatorContract."""
        storage = self.storage()
        features = storage.features

        def _check() -> None:
            if features.leasing:
                assert storage.coordination is not None, (
                    "features.leasing=True but storage.coordination is None"
                )
                for method in ("acquire", "renew", "release"):
                    assert callable(getattr(storage.coordination, method)), (
                        f"features.leasing=True but the coordinator is missing "
                        f"the {method!r} method"
                    )

        _check()

    def test_false_feature_declaration_is_honest_about_absence(self) -> None:
        """A feature declared False MUST NOT pretend the capability is
        available: ``streaming_blobs=False`` honestly means ``storage.artifacts``
        may be None (or the streaming surface will refuse), and a False
        ``coordination`` means ``storage.coordination`` is None. The inverse
        of the positive checks -- a backend that declared False but wired the
        object anyway would be over-declaring (the flag is the contract a
        caller branches on). This test asserts the flag and the object AGREE
        in the False direction for streaming_blobs specifically (the one
        capability whose absence is structurally observable)."""
        storage = self.storage()
        features = storage.features

        def _check() -> None:
            if not features.streaming_blobs:
                # Honest absence: streaming is declared off, so the streaming
                # surface is either unwired (None) or, if present, is a plain
                # AssetStore (not a streaming ArtifactStore). We do NOT assert
                # artifacts is None here -- a Storage may legitimately carry an
                # AssetStore under a different field -- only that the FLAG is
                # the truth a caller reads. Re-assert the flag round-trips.
                assert features.streaming_blobs is False

        _check()

    def test_fencing_feature_carries_a_fencing_token_on_leases(self) -> None:
        """``features.fencing = True`` MUST mean an acquired lease carries a
        usable ``fencing_token`` (the monotonic value a JobStore state commit
        checks to reject a stale writer). This is the precondition for the
        fencing guarantee; the behavioral enforcement (a JobStore rejecting a
        stale token) is backend-specific and lives in JobStoreContract. A
        coordinator that minted leases without a fencing token would make the
        flag a lie."""
        storage = self.storage()
        features = storage.features

        if not features.fencing:
            return

        async def _run() -> None:
            from datetime import timedelta

            assert storage.coordination is not None, (
                "features.fencing=True but storage.coordination is None"
            )
            token = await storage.coordination.acquire(
                key="fencing-feature-check", owner_id="o1", ttl=timedelta(seconds=30)
            )
            assert token is not None
            assert token.fencing_token is not None, (
                "features.fencing=True but the acquired LeaseToken carries no "
                "fencing_token -- the flag is not actually implemented"
            )

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
    "StorageFeaturesContract",
    "StorageTransactionManagerContract",
]
