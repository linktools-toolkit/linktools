#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore over the Filesystem reference backends -- content addressing,
immutability, lineage, tenant scoping, integrity (plan section 18, phase-1
acceptance)."""

import asyncio

import pytest

from linktools.ai.artifact import ArtifactIntegrityError, ArtifactStore, ANONYMOUS_PROVENANCE
from linktools.ai.artifact.models import ArtifactProvenance
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs"),
        FilesystemArtifactRecordStore(records_root=tmp_path / "records"),
    )


def test_identical_content_shares_blob_but_distinct_records(tmp_path) -> None:
    """Same content -> one shared blob (same sha256); each put -> its own record
    (distinct id), so the second producer's lineage is not lost."""
    store = _store(tmp_path)

    async def run() -> None:
        first = await store.put(content=b"hello", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        second = await store.put(content=b"hello", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        assert first.ref.sha256 == second.ref.sha256  # shared blob
        assert first.ref.id != second.ref.id  # distinct records
        assert first.ref.sha256 != first.ref.id  # id is a UUID, not the content hash

    asyncio.run(run())


def test_distinct_content_yields_distinct_artifacts(tmp_path) -> None:
    store = _store(tmp_path)

    async def run() -> None:
        a = await store.put(content=b"one", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        b = await store.put(content=b"two", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        assert a.ref.id != b.ref.id
        assert a.ref.sha256 != b.ref.sha256
        assert a.ref.size == 3

    asyncio.run(run())


def test_each_put_keeps_own_lineage(tmp_path) -> None:
    """Identical content from two producers yields two records, each carrying
    its OWN provenance (the old first-wins behavior lost the second's lineage)."""
    store = _store(tmp_path)

    async def run() -> None:
        first = await store.put(
            content=b"data",
            media_type="text/plain",
            tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="task-A"),
        )
        second = await store.put(
            content=b"data",
            media_type="text/plain",
            tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="task-B"),
        )
        assert first.provenance.producer_id == "task-A"
        assert second.provenance.producer_id == "task-B"
        assert first.ref.id != second.ref.id

    asyncio.run(run())


def test_blob_corruption_is_detected(tmp_path) -> None:
    store = _store(tmp_path)

    async def run() -> None:
        record = await store.put(content=b"payload", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        # Corrupt the stored blob in place: overwrite the file at the sha256-keyd
        # path so its bytes no longer hash to the address. The Filesystem backend
        # keeps blobs at a sharded path under blobs_root.
        path = store._blob._path(record.ref.sha256)
        path.write_bytes(b"TAMPERED")
        with pytest.raises(ArtifactIntegrityError):
            await store.get(artifact_id=record.ref.id, tenant_id="t1")

    asyncio.run(run())


def test_put_refuses_to_record_reference_to_corrupt_blob(tmp_path) -> None:
    # A second put of identical content hits the dedup path (the blob already
    # exists at the sha path). If that stored blob was corrupted/tampered, put
    # must FAIL rather than silently create a new ArtifactRecord pointing at the
    # corrupt blob.
    store = _store(tmp_path)

    async def run() -> None:
        record = await store.put(content=b"payload", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        # Corrupt the stored blob in place (content no longer hashes to its
        # sha256-keyed path).
        path = store._blob._path(record.ref.sha256)
        path.write_bytes(b"TAMPERED")
        with pytest.raises(ArtifactIntegrityError):
            await store.put(content=b"payload", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)

    asyncio.run(run())


def test_cross_tenant_access_denied(tmp_path) -> None:
    store = _store(tmp_path)

    async def run() -> None:
        record = await store.put(
            content=b"secret", media_type="text/plain", tenant_id="tenant-A", provenance=ANONYMOUS_PROVENANCE,
    )
        artifact_id = record.ref.id
        # Owning tenant sees both metadata and bytes.
        assert await store.stat(artifact_id=artifact_id, tenant_id="tenant-A") is not None
        assert await store.get(artifact_id=artifact_id, tenant_id="tenant-A") == b"secret"
        # A different tenant sees nothing on either path -- not even
        # confirmation the artifact exists.
        assert await store.stat(artifact_id=artifact_id, tenant_id="tenant-B") is None
        assert await store.get(artifact_id=artifact_id, tenant_id="tenant-B") is None

    asyncio.run(run())


def test_missing_artifact_returns_none(tmp_path) -> None:
    store = _store(tmp_path)

    async def run() -> None:
        assert await store.get(artifact_id="0" * 64, tenant_id="t1") is None
        assert await store.stat(artifact_id="0" * 64, tenant_id="t1") is None

    asyncio.run(run())


def test_lineage_and_metadata_preserved(tmp_path) -> None:
    store = _store(tmp_path)

    async def run() -> None:
        record = await store.put(
            content=b"out",
            media_type="application/json",
            tenant_id="t1",
            provenance=ArtifactProvenance(
                producer_kind="job_attempt",
                producer_id="attempt-1",
                run_id="run-1",
                parent_artifact_ids=("p1", "p2"),
                metadata={"kind": "output"},
            ),
        )
        assert record.provenance.parent_artifact_ids == ("p1", "p2")
        assert record.provenance.metadata == {"kind": "output"}
        assert record.provenance.producer_id == "attempt-1"
        assert record.provenance.run_id == "run-1"
        fetched = await store.stat(artifact_id=record.ref.id, tenant_id="t1")
        assert fetched == record

    asyncio.run(run())


def test_get_roundtrips(tmp_path) -> None:
    store = _store(tmp_path)

    async def run() -> None:
        record = await store.put(content=b"roundtrip", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        assert await store.get(artifact_id=record.ref.id, tenant_id="t1") == b"roundtrip"

    asyncio.run(run())


def test_same_content_dedupes_to_one_blob(tmp_path) -> None:
    """Identical content from several producers shares a SINGLE content blob;
    only the records multiply (one per put)."""
    store = _store(tmp_path)

    async def run() -> None:
        await store.put(content=b"shared", media_type="text/plain", tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="A"))
        await store.put(content=b"shared", media_type="text/plain", tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="B"))
        await store.put(content=b"shared", media_type="text/plain", tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="C"))
        # Count the files in the blob root's shard dirs: one blob file regardless
        # of how many records reference it.
        blob_files = [
            p
            for shard in (tmp_path / "blobs").iterdir() if shard.is_dir()
            for p in shard.iterdir()
            if p.is_file()
        ]
        record_files = list((tmp_path / "records").rglob("*.json"))
        assert len(blob_files) == 1  # content deduplicated
        assert len(record_files) == 3  # one lineage record per put

    asyncio.run(run())


def test_each_record_preserves_own_parents_and_lineage(tmp_path) -> None:
    """Two records of identical content each carry their own parent lineage and
    creator -- the second production event's blood line is not overwritten by
    the first (the original 7.2 bug)."""
    store = _store(tmp_path)

    async def run() -> None:
        first = await store.put(
            content=b"x",
            media_type="text/plain",
            tenant_id="t1",
            provenance=ArtifactProvenance(
                producer_kind="job", producer_id="job-1", parent_artifact_ids=("p1",),
            ),
        )
        second = await store.put(
            content=b"x",
            media_type="text/plain",
            tenant_id="t1",
            provenance=ArtifactProvenance(
                producer_kind="job",
                producer_id="job-2",
                parent_artifact_ids=("p2", "p3"),
            ),
        )
        fetched_first = await store.stat(artifact_id=first.ref.id, tenant_id="t1")
        fetched_second = await store.stat(artifact_id=second.ref.id, tenant_id="t1")
        assert fetched_first.provenance.parent_artifact_ids == ("p1",)
        assert fetched_first.provenance.producer_id == "job-1"
        assert fetched_second.provenance.parent_artifact_ids == ("p2", "p3")
        assert fetched_second.provenance.producer_id == "job-2"

    asyncio.run(run())
