#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blob orphan sweeper: proves the core orphan contract against the
backend-agnostic sweep. A blob is deletable only when (a) no record references
it AND (b) it is past the grace window -- so an in-flight transaction that has
written its blob but not yet committed its record is never corrupted."""

from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.artifact.models import (
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRef,
    ArtifactRecordCorruptError,
)
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)
from linktools.ai.storage.orphan import (
    OrphanSweepConfig,
    sweep_orphan_blobs,
)


def _stores(tmp_path):
    blob_store = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    record_store = FilesystemArtifactRecordStore(
        records_root=tmp_path / "records"
    )
    coordinator = InProcessArtifactDigestCoordinator()
    return blob_store, record_store, coordinator


async def _aiter(content: bytes):
    yield content


def _sha(content: bytes) -> ArtifactDigest:
    return ArtifactDigest.from_bytes(content)


def _record(artifact_id: str, sha: str, tenant: str = "t1") -> ArtifactRecord:
    return ArtifactRecord(
        ref=ArtifactRef(id=artifact_id, sha256=sha, media_type="", size=0),
        tenant_id=tenant,
        provenance=ArtifactProvenance(producer_kind="anonymous", producer_id=""),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_orphan_blob_past_grace_is_deleted_referenced_blob_is_kept(tmp_path):
    blob_store, record_store, coordinator = _stores(tmp_path)

    async def _seed():
        # A referenced blob: written AND pinned by a record.
        ref_content = b"referenced"
        ref_sha = _sha(ref_content)
        await blob_store.put_if_absent(
            digest=ref_sha, source=_aiter(ref_content), size=len(ref_content)
        )
        await record_store.put(_record("art-ref", ref_sha.value))
        # An orphan blob: written but NO record pins it.
        orphan_content = b"orphan"
        orphan_sha = _sha(orphan_content)
        await blob_store.put_if_absent(
            digest=orphan_sha, source=_aiter(orphan_content), size=len(orphan_content)
        )
        return ref_sha, orphan_sha

    ref_sha, orphan_sha = _run(_seed())

    # Sweep 25h later: the orphan is past the 24h grace -> deleted; the
    # referenced blob is in use -> kept.
    future = datetime.now(timezone.utc) + timedelta(hours=25)
    stats = _run(sweep_orphan_blobs(blob_store, record_store, coordinator, now=future))

    assert stats.deleted == 1
    assert stats.in_use == 1
    assert stats.kept_within_grace == 0
    # The orphan is gone; the referenced blob survives.
    assert _run(blob_store.stat(digest=orphan_sha)) is None
    assert _run(blob_store.stat(digest=ref_sha)) is not None


def test_orphan_blob_within_grace_is_kept(tmp_path):
    """An unreferenced blob inside the grace window must NOT be deleted -- the
    transaction that wrote it may still commit a pinning record."""
    blob_store, record_store, coordinator = _stores(tmp_path)

    async def _seed():
        orphan_content = b"fresh-orphan"
        orphan_sha = _sha(orphan_content)
        await blob_store.put_if_absent(
            digest=orphan_sha, source=_aiter(orphan_content), size=len(orphan_content)
        )
        return orphan_sha

    orphan_sha = _run(_seed())

    # Sweep only 1h later (well inside the 24h window).
    soon = datetime.now(timezone.utc) + timedelta(hours=1)
    stats = _run(sweep_orphan_blobs(blob_store, record_store, coordinator, now=soon))

    assert stats.deleted == 0
    assert stats.kept_within_grace == 1
    assert _run(blob_store.stat(digest=orphan_sha)) is not None


def test_sweep_is_idempotent(tmp_path):
    """Re-running the sweep over already-swept state deletes nothing more and
    raises no error."""
    blob_store, record_store, coordinator = _stores(tmp_path)

    async def _seed():
        orphan_content = b"gone"
        orphan_sha = _sha(orphan_content)
        await blob_store.put_if_absent(
            digest=orphan_sha, source=_aiter(orphan_content), size=len(orphan_content)
        )
        return orphan_sha

    _ = _run(_seed())
    future = datetime.now(timezone.utc) + timedelta(hours=25)

    first = _run(sweep_orphan_blobs(blob_store, record_store, coordinator, now=future))
    second = _run(sweep_orphan_blobs(blob_store, record_store, coordinator, now=future))

    assert first.deleted == 1
    assert second.deleted == 0  # nothing left to delete


def test_custom_grace_period_governs_deletion(tmp_path):
    """A caller may narrow the grace window; a blob just past the custom window
    is deletable even though it would be within the default."""
    blob_store, record_store, coordinator = _stores(tmp_path)

    async def _seed():
        content = b"short-fuse"
        sha = _sha(content)
        await blob_store.put_if_absent(
            digest=sha, source=_aiter(content), size=len(content)
        )
        return sha

    sha = _run(_seed())
    config = OrphanSweepConfig(grace_period=timedelta(minutes=1), sweep_interval=timedelta(minutes=1))

    # 5 minutes later: past the 1-minute custom window -> deleted.
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    stats = _run(sweep_orphan_blobs(blob_store, record_store, coordinator, config, now=future))
    assert stats.deleted == 1
    assert _run(blob_store.stat(digest=sha)) is None


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_corrupt_record_aborts_sweep_fail_closed(tmp_path):
    """A corrupt record file must abort the sweep and delete nothing -- the
    sweeper cannot know whether the broken record pins a blob, so it fails
    closed rather than risk deleting a referenced blob."""
    blob_store, record_store, coordinator = _stores(tmp_path)

    async def _seed():
        # A referenced blob pinned by a valid record.
        ref_content = b"referenced"
        ref_sha = _sha(ref_content)
        await blob_store.put_if_absent(
            digest=ref_sha, source=_aiter(ref_content), size=len(ref_content)
        )
        await record_store.put(_record("art-ref", ref_sha.value))
        # An orphan blob past the grace window that the sweeper WOULD delete if
        # the scan were healthy.
        orphan_content = b"would-be-deleted"
        orphan_sha = _sha(orphan_content)
        await blob_store.put_if_absent(
            digest=orphan_sha, source=_aiter(orphan_content), size=len(orphan_content)
        )
        return ref_sha, orphan_sha

    ref_sha, orphan_sha = _run(_seed())

    # Corrupt the record JSON in place (unparseable).
    record_path = tmp_path / "records" / "t1" / "art-ref.json"
    record_path.write_bytes(b"{not valid json")

    future = datetime.now(timezone.utc) + timedelta(hours=25)
    with pytest.raises(ArtifactRecordCorruptError):
        _run(sweep_orphan_blobs(blob_store, record_store, coordinator, now=future))

    # Nothing was deleted: the orphan survives because the scan aborted, and the
    # referenced blob is untouched.
    assert _run(blob_store.stat(digest=orphan_sha)) is not None
    assert _run(blob_store.stat(digest=ref_sha)) is not None
