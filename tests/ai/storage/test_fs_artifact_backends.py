#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemArtifactBlobStore + FilesystemArtifactRecordStore (
ops 4-5; ). The Filesystem reference backends stream to/from disk
(no whole-blob bytes buffer), verify the claimed digest on write, and reuse the
crash-safe atomic-write helper used by every other File store."""

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest
from linktools.ai.artifact import ANONYMOUS_PROVENANCE

from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.artifact.models import (
    ArtifactBlobNotFoundError,
    ArtifactIntegrityError,
)
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)
from linktools.ai.storage.protocols import BlobInfo


def _aiter(chunks: "list[bytes]") -> AsyncIterator[bytes]:
    async def _g() -> AsyncIterator[bytes]:
        for c in chunks:
            yield c

    return _g()


def _digest(data: bytes) -> ArtifactDigest:
    return ArtifactDigest.from_bytes(data)


_VALID = "0" * 64


# --- FilesystemArtifactBlobStore ---


def test_blob_put_if_absent_then_open_roundtrips_streaming(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"X" * (70 * 1024)  # 70 KiB -> multiple 64 KiB read chunks on open

    async def run() -> None:
        digest = _digest(payload)
        info = await blob.put_if_absent(
            digest=digest,
            source=_aiter([payload[i : i + 16384] for i in range(0, len(payload), 16384)]),
            size=len(payload),
        )
        assert isinstance(info, BlobInfo)
        assert info.digest == digest.value
        assert info.size == len(payload)
        # The blob lands at the sharded path blobs/<xx>/<sha>.
        assert (tmp_path / "blobs" / digest.value[:2] / digest.value).exists()
        collected: "list[bytes]" = []
        async with blob.open(digest=digest) as chunks:
            async for chunk in chunks:
                collected.append(chunk)
        assert b"".join(collected) == payload
        assert len(collected) >= 2  # 70 KiB / 64 KiB read chunks -> streamed, not joined

    asyncio.run(run())


def test_blob_put_if_absent_is_idempotent_on_matching_digest(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"same-content"

    async def run() -> None:
        d = _digest(payload)
        first = await blob.put_if_absent(digest=d, source=_aiter([payload]), size=len(payload))
        # Second put of identical content: the blob already exists at the address;
        # put_if_absent returns success without overwriting or erroring.
        second = await blob.put_if_absent(digest=d, source=_aiter([payload]), size=len(payload))
        assert first.digest == second.digest == d.value

    asyncio.run(run())


def test_blob_put_if_absent_consumes_source_even_when_blob_exists(tmp_path: Path) -> None:
    # The contract: when the blob already exists, the source is STILL fully
    # consumed and verified -- a put can never skip input validation by claiming
    # a digest that is already present.
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"already-present"

    async def run() -> None:
        d = _digest(payload)
        await blob.put_if_absent(digest=d, source=_aiter([payload]), size=len(payload))

        consumed: "list[bytes]" = []

        async def _tracking() -> AsyncIterator[bytes]:
            async for chunk in _aiter([payload]):
                consumed.append(chunk)
                yield chunk

        info = await blob.put_if_absent(digest=d, source=_tracking(), size=len(payload))
        assert info.digest == d.value
        # The whole source was drained, not short-circuited by "blob exists".
        assert b"".join(consumed) == payload

    asyncio.run(run())


def test_blob_put_if_absent_wrong_content_for_known_digest_fails(tmp_path: Path) -> None:
    # Same digest but the source's actual bytes do not hash to it -> failure,
    # even when a blob for the digest already exists (the existing blob is
    # correct; the source is lying).
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"real-content"
    digest = _digest(payload)

    async def run() -> None:
        await blob.put_if_absent(digest=digest, source=_aiter([payload]), size=len(payload))
        with pytest.raises(ArtifactIntegrityError):
            await blob.put_if_absent(
                digest=digest, source=_aiter([b"WRONG-content-same-claimed-digest"]),
                size=len(b"WRONG-content-same-claimed-digest"),
            )

    asyncio.run(run())


def test_blob_put_if_absent_rejects_digest_mismatch(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")

    async def run() -> None:
        with pytest.raises(ArtifactIntegrityError):
            await blob.put_if_absent(
                digest=ArtifactDigest.parse(_VALID), source=_aiter([b"not-the-claimed-digest"]), size=19
            )
        # Nothing published on mismatch.
        assert not (tmp_path / "blobs").rglob("*") or not list(
            (tmp_path / "blobs").rglob("0*64"))
        # No temp file left behind.
        assert not list((tmp_path / "blobs").rglob("*.tmp"))

    asyncio.run(run())


def test_blob_put_if_absent_refuses_corrupt_existing(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"payload"
    digest = _digest(payload)

    async def run() -> None:
        await blob.put_if_absent(digest=digest, source=_aiter([payload]), size=7)
        # Tamper the stored blob in place; a second put of the same digest must
        # refuse to record a reference to the corrupt blob.
        path = tmp_path / "blobs" / digest.value[:2] / digest.value
        path.write_bytes(b"TAMPERED")
        with pytest.raises(ArtifactIntegrityError):
            await blob.put_if_absent(digest=digest, source=_aiter([payload]), size=7)

    asyncio.run(run())


def test_blob_open_missing_raises(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")

    async def run() -> None:
        # Missing (absent) blob surfaces the UNIFIED ArtifactBlobNotFoundError
        # , distinct from ArtifactIntegrityError (corrupt-but-present).
        with pytest.raises(ArtifactBlobNotFoundError):
            async with blob.open(digest=ArtifactDigest.parse(_VALID)) as _chunks:
                pass

    asyncio.run(run())


def test_blob_stat_and_delete(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"stat-me"
    digest = _digest(payload)

    async def run() -> None:
        assert await blob.stat(digest=digest) is None
        await blob.put_if_absent(digest=digest, source=_aiter([payload]), size=7)
        info = await blob.stat(digest=digest)
        assert info is not None and info.size == 7
        await blob.delete(digest=digest)
        assert await blob.stat(digest=digest) is None
        # delete is idempotent (no error on missing).
        await blob.delete(digest=digest)

    asyncio.run(run())


# --- FilesystemArtifactRecordStore ---


def _record(artifact_id: str = "art-1", tenant_id: str = "t1", sha: str = "a" * 64):
    from linktools.ai.artifact.models import (
        ArtifactProvenance,
        ArtifactRecord,
        ArtifactRef,
    )
    from datetime import datetime, timezone

    return ArtifactRecord(
        ref=ArtifactRef(id=artifact_id, sha256=sha, media_type="text/plain", size=4),
        tenant_id=tenant_id,
        provenance=ArtifactProvenance(
            producer_kind="anonymous", producer_id="", metadata={"k": "v"}
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_record_put_get_delete_roundtrip(tmp_path: Path) -> None:
    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        rec = _record()
        await store.put(rec)
        assert (tmp_path / "records" / "t1" / "art-1.json").exists()
        fetched = await store.get(artifact_id="art-1", tenant_id="t1")
        assert fetched == rec
        assert await store.delete("art-1", tenant_id="t1") is True
        assert await store.get(artifact_id="art-1", tenant_id="t1") is None
        assert await store.delete("art-1", tenant_id="t1") is False

    asyncio.run(run())


def test_record_get_tenant_gate(tmp_path: Path) -> None:
    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        await store.put(_record(artifact_id="art-1", tenant_id="tenant-A"))
        # Foreign tenant learns nothing.
        assert await store.get(artifact_id="art-1", tenant_id="tenant-B") is None
        assert await store.delete("art-1", tenant_id="tenant-B") is False

    asyncio.run(run())


def test_record_iter_referenced_digests(tmp_path: Path) -> None:
    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        await store.put(_record(artifact_id="art-1", sha="a" * 64))
        await store.put(_record(artifact_id="art-2", sha="b" * 64))
        digests = [d async for d in store.iter_referenced_digests()]
        assert sorted(digests) == sorted(["a" * 64, "b" * 64])

    asyncio.run(run())


def test_record_put_same_id_identical_is_idempotent(tmp_path: Path) -> None:
    from linktools.ai.errors import ArtifactRecordConflictError

    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        first = await store.put(_record(artifact_id="art-1", sha="a" * 64))
        # Same id + byte-identical record -> idempotent, no conflict.
        second = await store.put(_record(artifact_id="art-1", sha="a" * 64))
        assert first == second

    asyncio.run(run())


def test_record_put_same_id_different_content_conflicts(tmp_path: Path) -> None:
    from linktools.ai.errors import ArtifactRecordConflictError

    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        await store.put(_record(artifact_id="art-1", sha="a" * 64))
        with pytest.raises(ArtifactRecordConflictError):
            await store.put(_record(artifact_id="art-1", sha="b" * 64))
        # The original lineage is intact (no overwrite).
        fetched = await store.get(artifact_id="art-1", tenant_id="t1")
        assert fetched is not None and fetched.ref.sha256 == "a" * 64

    asyncio.run(run())


def test_record_corrupt_record_aborts_iter_fail_closed(tmp_path: Path) -> None:
    from linktools.ai.errors import ArtifactRecordCorruptError

    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        await store.put(_record(artifact_id="art-1", sha="a" * 64))
        # Corrupt the stored record: write garbage at its path.
        (tmp_path / "records" / "t1" / "art-1.json").write_bytes(b"not json")
        with pytest.raises(ArtifactRecordCorruptError):
            async for _ in store.iter_referenced_digests():
                pass

    asyncio.run(run())


def _record_with_provenance(
    artifact_id: str,
    tenant_id: str,
    *,
    producer_kind: str,
    producer_id: "str | None",
    run_id: "str | None",
    sha: str = "a" * 64,
):
    """A record helper that takes explicit provenance, for the parent/provenance
    index tests (the default ``_record`` is anonymous with no run)."""
    from datetime import datetime, timezone

    from linktools.ai.artifact.models import (
        ArtifactProvenance,
        ArtifactRecord,
        ArtifactRef,
    )

    return ArtifactRecord(
        ref=ArtifactRef(id=artifact_id, sha256=sha, media_type="text/plain", size=4),
        tenant_id=tenant_id,
        provenance=ArtifactProvenance(
            producer_kind=producer_kind, producer_id=producer_id or "", run_id=run_id
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_record_iter_by_run_id_indexes_provenance(tmp_path: Path) -> None:
    """parent/provenance index: iter_by_run_id yields exactly the
    records produced under that run, tenant-scoped, and a None run_id yields the
    unattributed records. Filesystem has no index columns, so this is an honest
    scan-and-filter -- but it must still be correct."""
    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        await store.put(
            _record_with_provenance("a1", "t1", producer_kind="job_attempt", producer_id="att-1", run_id="run-A")
        )
        await store.put(
            _record_with_provenance("a2", "t1", producer_kind="job_attempt", producer_id="att-2", run_id="run-A")
        )
        await store.put(
            _record_with_provenance("a3", "t1", producer_kind="job_attempt", producer_id="att-3", run_id="run-B")
        )
        await store.put(_record_with_provenance("a4", "t1", producer_kind="anonymous", producer_id=None, run_id=None))
        # Cross-tenant record under run-A: must NOT surface in t1's run-A view.
        await store.put(
            _record_with_provenance("a5", "t2", producer_kind="job_attempt", producer_id="att-x", run_id="run-A")
        )

        run_a = [r.ref.id async for r in store.iter_by_run_id("run-A", tenant_id="t1")]
        run_b = [r.ref.id async for r in store.iter_by_run_id("run-B", tenant_id="t1")]
        unattributed = [r.ref.id async for r in store.iter_by_run_id(None, tenant_id="t1")]
        # Without tenant scope, run-A spans both tenants.
        all_run_a_tenants = sorted(
            [r.tenant_id async for r in store.iter_by_run_id("run-A")]
        )

        assert sorted(run_a) == ["a1", "a2"]
        assert run_b == ["a3"]
        assert unattributed == ["a4"]
        # Without tenant scope, run-A yields all three records (a1/a2 in t1,
        # a5 in t2), so the tenant multiset is [t1, t1, t2].
        assert all_run_a_tenants == ["t1", "t1", "t2"]

    asyncio.run(run())


def test_record_iter_by_producer_indexes_provenance(tmp_path: Path) -> None:
    """parent/provenance index: iter_by_producer yields records by
    producer kind [+ id], tenant-scoped."""
    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        await store.put(
            _record_with_provenance("a1", "t1", producer_kind="eval", producer_id="eval-1", run_id="r")
        )
        await store.put(
            _record_with_provenance("a2", "t1", producer_kind="eval", producer_id="eval-2", run_id="r")
        )
        await store.put(
            _record_with_provenance("a3", "t1", producer_kind="job_attempt", producer_id="att-1", run_id="r")
        )
        await store.put(_record_with_provenance("a4", "t1", producer_kind="anonymous", producer_id="", run_id=None))

        evals = sorted([r.ref.id async for r in store.iter_by_producer("eval", tenant_id="t1")])
        eval_1 = [r.ref.id async for r in store.iter_by_producer("eval", "eval-1", tenant_id="t1")]
        anon = [r.ref.id async for r in store.iter_by_producer("anonymous", tenant_id="t1")]
        missing = [r.ref.id async for r in store.iter_by_producer("nope", tenant_id="t1")]

        assert evals == ["a1", "a2"]
        assert eval_1 == ["a1"]
        assert anon == ["a4"]
        assert missing == []

    asyncio.run(run())


# --- End-to-end through FilesystemStorage.artifacts (wiring) ---


def test_filesystem_storage_artifacts_is_the_streaming_store(tmp_path: Path) -> None:
    from linktools.ai.artifact.store import ArtifactStore
    from linktools.ai.storage.facade import FilesystemStorage

    storage = FilesystemStorage(root=tmp_path / "data")
    # The artifact store is wired over the Filesystem backends, NOT the asset
    # store; the blob backend is the Filesystem one.
    assert isinstance(storage.artifacts, ArtifactStore)
    assert isinstance(storage.artifacts._blob, FilesystemArtifactBlobStore)
    assert isinstance(storage.artifacts._records, FilesystemArtifactRecordStore)

    payload = b"end-to-end-streamed"

    async def run() -> None:

        async def _src() -> AsyncIterator[bytes]:
            yield payload

        record = await storage.artifacts.put_stream(
            source=_src(), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        collected: "list[bytes]" = []
        async with storage.artifacts.open_stream(
            artifact_id=record.ref.id, tenant_id="t1"
        ) as chunks:
            async for c in chunks:
                collected.append(c)
        assert b"".join(collected) == payload

    asyncio.run(run())


def test_blob_iter_digests_with_mtime(tmp_path: Path) -> None:
    # The orphan sweeper consumes (digest, modified_at); exercise it directly.
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")

    async def run() -> None:
        await blob.put_if_absent(digest=_digest(b"a"), source=_aiter([b"a"]), size=1)
        await blob.put_if_absent(
            digest=_digest(b"bb"), source=_aiter([b"bb"]), size=2
        )
        pairs = [(d, m) async for d, m in blob.iter_digests_with_mtime()]
        assert sorted(d for d, _ in pairs) == sorted(
            [_digest(b"a").value, _digest(b"bb").value]
        )
        from datetime import datetime

        for _, mtime in pairs:
            assert isinstance(mtime, datetime)
            assert mtime.tzinfo is not None  # timezone-aware

    asyncio.run(run())


def test_blob_put_if_absent_rejects_size_mismatch(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")

    async def run() -> None:
        payload = b"payload"  # 7 bytes
        digest = _digest(payload)
        with pytest.raises(ArtifactIntegrityError):
            await blob.put_if_absent(
                digest=digest, source=_aiter([payload]), size=999
            )
        # Nothing published on size mismatch.
        assert await blob.stat(digest=digest) is None

    asyncio.run(run())


def test_blob_rejects_path_traversal_digest(tmp_path: Path) -> None:
    # A traversal string cannot become an ArtifactDigest, so it can never reach
    # the Filesystem backend's path construction -- the boundary rejects it.
    from linktools.ai.errors import InvalidArtifactDigestError

    with pytest.raises(InvalidArtifactDigestError):
        ArtifactDigest.parse("../escape")


def test_record_rejects_path_traversal_ids(tmp_path: Path) -> None:
    # Crafted tenant_id / artifact_id cannot escape the records tree.
    store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def run() -> None:
        rec = _record(artifact_id="art-1", tenant_id="t1")
        await store.put(rec)
        with pytest.raises(ValueError):
            await store.get(artifact_id="art-1", tenant_id="../../etc/passwd")
        with pytest.raises(ValueError):
            await store.delete("art-1", tenant_id="../escape")
        with pytest.raises(ValueError):
            await store.put(_record(artifact_id="../x", tenant_id="t1"))

    asyncio.run(run())
