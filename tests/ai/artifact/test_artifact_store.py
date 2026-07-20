#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore over AssetStore -- content addressing, immutability,
lineage, tenant scoping, integrity (plan section 18, phase-1 acceptance)."""

import asyncio

import pytest

from linktools.ai.artifact import ArtifactIntegrityError, ArtifactStore
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.store import AssetStore
from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets


def _store() -> ArtifactStore:
    return build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))


def test_identical_content_shares_blob_but_distinct_records() -> None:
    """Same content -> one shared blob (same sha256); each put -> its own record
    (distinct id), so the second producer's lineage is not lost."""
    store = _store()

    async def run() -> None:
        first = await store.put(b"hello", media_type="text/plain", tenant_id="t1")
        second = await store.put(b"hello", media_type="text/plain", tenant_id="t1")
        assert first.ref.sha256 == second.ref.sha256  # shared blob
        assert first.ref.id != second.ref.id  # distinct records
        assert first.ref.sha256 != first.ref.id  # id is a UUID, not the content hash

    asyncio.run(run())


def test_distinct_content_yields_distinct_artifacts() -> None:
    store = _store()

    async def run() -> None:
        a = await store.put(b"one", media_type="text/plain", tenant_id="t1")
        b = await store.put(b"two", media_type="text/plain", tenant_id="t1")
        assert a.ref.id != b.ref.id
        assert a.ref.sha256 != b.ref.sha256
        assert a.ref.size == 3

    asyncio.run(run())


def test_each_put_keeps_own_lineage() -> None:
    """Identical content from two producers yields two records, each carrying
    its OWN provenance (the old first-wins behavior lost the second's lineage)."""
    store = _store()

    async def run() -> None:
        first = await store.put(
            b"data",
            media_type="text/plain",
            tenant_id="t1",
            created_by_task_id="task-A",
        )
        second = await store.put(
            b"data",
            media_type="text/plain",
            tenant_id="t1",
            created_by_task_id="task-B",
        )
        assert first.created_by_task_id == "task-A"
        assert second.created_by_task_id == "task-B"
        assert first.ref.id != second.ref.id

    asyncio.run(run())


def test_blob_corruption_is_detected() -> None:
    store = _store()

    async def run() -> None:
        record = await store.put(b"payload", media_type="text/plain", tenant_id="t1")
        # Corrupt the stored blob (addressed by sha256, not the record id).
        backend = store._blob._assets._primary
        path_key = store._blob._path(record.ref.sha256).value
        _content, info = backend._entries[path_key]
        backend._entries[path_key] = (b"TAMPERED", info)
        with pytest.raises(ArtifactIntegrityError):
            await store.get(record.ref.id, tenant_id="t1")

    asyncio.run(run())


def test_put_refuses_to_record_reference_to_corrupt_blob() -> None:
    # A second put of identical content hits the if_none_match conflict path
    # (the blob already exists at the sha path). If that stored blob was
    # corrupted/tampered, put must FAIL rather than silently create a new
    # ArtifactRecord pointing at the corrupt blob.
    store = _store()

    async def run() -> None:
        record = await store.put(b"payload", media_type="text/plain", tenant_id="t1")
        # Corrupt the stored blob in place (content no longer hashes to its
        # sha256-keyed path).
        backend = store._blob._assets._primary
        path_key = store._blob._path(record.ref.sha256).value
        _content, info = backend._entries[path_key]
        backend._entries[path_key] = (b"TAMPERED", info)
        with pytest.raises(ArtifactIntegrityError):
            await store.put(b"payload", media_type="text/plain", tenant_id="t1")

    asyncio.run(run())


def test_cross_tenant_access_denied() -> None:
    store = _store()

    async def run() -> None:
        record = await store.put(
            b"secret", media_type="text/plain", tenant_id="tenant-A"
        )
        artifact_id = record.ref.id
        # Owning tenant sees both metadata and bytes.
        assert await store.stat(artifact_id, tenant_id="tenant-A") is not None
        assert await store.get(artifact_id, tenant_id="tenant-A") == b"secret"
        # A different tenant sees nothing on either path -- not even
        # confirmation the artifact exists.
        assert await store.stat(artifact_id, tenant_id="tenant-B") is None
        assert await store.get(artifact_id, tenant_id="tenant-B") is None

    asyncio.run(run())


def test_missing_artifact_returns_none() -> None:
    store = _store()

    async def run() -> None:
        assert await store.get("0" * 64, tenant_id="t1") is None
        assert await store.stat("0" * 64, tenant_id="t1") is None

    asyncio.run(run())


def test_lineage_and_metadata_preserved() -> None:
    store = _store()

    async def run() -> None:
        record = await store.put(
            b"out",
            media_type="application/json",
            tenant_id="t1",
            created_by_job_id="job-1",
            created_by_task_id="task-1",
            created_by_attempt_id="attempt-1",
            parent_artifact_ids=("p1", "p2"),
            metadata={"kind": "output"},
        )
        assert record.parent_artifact_ids == ("p1", "p2")
        assert record.metadata == {"kind": "output"}
        assert record.created_by_job_id == "job-1"
        fetched = await store.stat(record.ref.id, tenant_id="t1")
        assert fetched == record

    asyncio.run(run())


def test_get_roundtrips() -> None:
    store = _store()

    async def run() -> None:
        record = await store.put(b"roundtrip", media_type="text/plain", tenant_id="t1")
        assert await store.get(record.ref.id, tenant_id="t1") == b"roundtrip"

    asyncio.run(run())


def test_same_content_dedupes_to_one_blob() -> None:
    """Identical content from several producers shares a SINGLE content blob;
    only the records multiply (one per put)."""
    store = _store()

    async def run() -> None:
        await store.put(b"shared", media_type="text/plain", tenant_id="t1", created_by_task_id="A")
        await store.put(b"shared", media_type="text/plain", tenant_id="t1", created_by_task_id="B")
        await store.put(b"shared", media_type="text/plain", tenant_id="t1", created_by_task_id="C")
        backend = store._blob._assets._primary
        blob_keys = [k for k in backend._entries if "/blobs/sha256/" in k]
        record_keys = [k for k in backend._entries if "/records/" in k]
        assert len(blob_keys) == 1  # content deduplicated
        assert len(record_keys) == 3  # one lineage record per put

    asyncio.run(run())


def test_each_record_preserves_own_parents_and_lineage() -> None:
    """Two records of identical content each carry their own parent lineage and
    creator -- the second production event's blood line is not overwritten by
    the first (the original 7.2 bug)."""
    store = _store()

    async def run() -> None:
        first = await store.put(
            b"x",
            media_type="text/plain",
            tenant_id="t1",
            created_by_job_id="job-1",
            parent_artifact_ids=("p1",),
        )
        second = await store.put(
            b"x",
            media_type="text/plain",
            tenant_id="t1",
            created_by_job_id="job-2",
            parent_artifact_ids=("p2", "p3"),
        )
        fetched_first = await store.stat(first.ref.id, tenant_id="t1")
        fetched_second = await store.stat(second.ref.id, tenant_id="t1")
        assert fetched_first.parent_artifact_ids == ("p1",)
        assert fetched_first.created_by_job_id == "job-1"
        assert fetched_second.parent_artifact_ids == ("p2", "p3")
        assert fetched_second.created_by_job_id == "job-2"

    asyncio.run(run())
