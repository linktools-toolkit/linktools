#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore over ResourceStore -- content addressing, immutability,
lineage, tenant scoping, integrity (plan section 18, phase-1 acceptance)."""

import asyncio

import pytest

from linktools.ai.artifact import ArtifactIntegrityError, ArtifactStore
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.store import ResourceStore


def _store() -> ArtifactStore:
    return ArtifactStore(ResourceStore(primary=MemoryResourceBackend()))


def test_identical_content_reuses_same_artifact() -> None:
    store = _store()

    async def run() -> None:
        first = await store.put(b"hello", media_type="text/plain", tenant_id="t1")
        second = await store.put(b"hello", media_type="text/plain", tenant_id="t1")
        assert first.ref.id == second.ref.id
        assert first.ref.sha256 == second.ref.sha256
        assert first.ref.id == first.ref.sha256  # content-addressed

    asyncio.run(run())


def test_distinct_content_yields_distinct_artifacts() -> None:
    store = _store()

    async def run() -> None:
        a = await store.put(b"one", media_type="text/plain", tenant_id="t1")
        b = await store.put(b"two", media_type="text/plain", tenant_id="t1")
        assert a.ref.id != b.ref.id
        assert a.ref.size == 3

    asyncio.run(run())


def test_record_is_immutable_first_wins() -> None:
    store = _store()

    async def run() -> None:
        first = await store.put(
            b"data",
            media_type="text/plain",
            tenant_id="t1",
            created_by_task_id="task-A",
        )
        # Same content again with different provenance: the original record
        # wins, so the artifact is immutable.
        second = await store.put(
            b"data",
            media_type="text/plain",
            tenant_id="t1",
            created_by_task_id="task-B",
        )
        assert second.created_by_task_id == "task-A"
        assert second.created_at == first.created_at

    asyncio.run(run())


def test_corrupt_content_fails_integrity_check() -> None:
    store = _store()

    async def run() -> None:
        record = await store.put(b"payload", media_type="text/plain", tenant_id="t1")
        # Corrupt the stored blob under the content-addressed path.
        backend = store._resources._primary
        path_key = store._content_path(record.ref.id).value
        _content, info = backend._entries[path_key]
        backend._entries[path_key] = (b"TAMPERED", info)
        with pytest.raises(ArtifactIntegrityError):
            await store.get(record.ref.id, tenant_id="t1")

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
