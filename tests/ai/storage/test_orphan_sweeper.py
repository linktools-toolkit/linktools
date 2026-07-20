#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blob orphan sweeper: proves the core orphan contract against the asset-backed
reference sweep. A blob is deletable only when (a) no record references it AND
(b) it is past the grace window -- so an in-flight transaction that has written
its blob but not yet committed its record is never corrupted."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.artifact.models import ArtifactRecord, ArtifactRef
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.storage.artifact_backends import (
    AssetBackedArtifactBlobStore,
    AssetBackedArtifactRecordStore,
)
from linktools.ai.storage.orphan import (
    OrphanSweepConfig,
    sweep_asset_backed_orphan_blobs,
)


def _stores():
    assets = AssetStore(primary=MemoryAssetBackend())
    blob_store = AssetBackedArtifactBlobStore(
        assets, blobs_root=AssetPath("/artifacts/blobs/sha256")
    )
    record_store = AssetBackedArtifactRecordStore(
        assets, records_root=AssetPath("/artifacts/records")
    )
    return blob_store, record_store


async def _aiter(content: bytes):
    yield content


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _record(artifact_id: str, sha: str, tenant: str = "t1") -> ArtifactRecord:
    return ArtifactRecord(
        ref=ArtifactRef(id=artifact_id, sha256=sha, media_type="", size=0),
        tenant_id=tenant,
        created_by_job_id=None,
        created_by_task_id=None,
        created_by_attempt_id=None,
        parent_artifact_ids=(),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_orphan_blob_past_grace_is_deleted_referenced_blob_is_kept():
    blob_store, record_store = _stores()

    async def _seed():
        # A referenced blob: written AND pinned by a record.
        ref_content = b"referenced"
        ref_sha = _sha(ref_content)
        await blob_store.put_if_absent(
            digest=ref_sha, source=_aiter(ref_content), size=len(ref_content)
        )
        await record_store.put(_record("art-ref", ref_sha))
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
    stats = _run(sweep_asset_backed_orphan_blobs(blob_store, record_store, now=future))

    assert stats.deleted == 1
    assert stats.in_use == 1
    assert stats.kept_within_grace == 0
    # The orphan is gone; the referenced blob survives.
    assert _run(blob_store.stat(digest=orphan_sha)) is None
    assert _run(blob_store.stat(digest=ref_sha)) is not None


def test_orphan_blob_within_grace_is_kept():
    """An unreferenced blob inside the grace window must NOT be deleted -- the
    transaction that wrote it may still commit a pinning record."""
    blob_store, record_store = _stores()

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
    stats = _run(sweep_asset_backed_orphan_blobs(blob_store, record_store, now=soon))

    assert stats.deleted == 0
    assert stats.kept_within_grace == 1
    assert _run(blob_store.stat(digest=orphan_sha)) is not None


def test_sweep_is_idempotent():
    """Re-running the sweep over already-swept state deletes nothing more and
    raises no error."""
    blob_store, record_store = _stores()

    async def _seed():
        orphan_content = b"gone"
        orphan_sha = _sha(orphan_content)
        await blob_store.put_if_absent(
            digest=orphan_sha, source=_aiter(orphan_content), size=len(orphan_content)
        )
        return orphan_sha

    _ = _run(_seed())
    future = datetime.now(timezone.utc) + timedelta(hours=25)

    first = _run(sweep_asset_backed_orphan_blobs(blob_store, record_store, now=future))
    second = _run(sweep_asset_backed_orphan_blobs(blob_store, record_store, now=future))

    assert first.deleted == 1
    assert second.deleted == 0  # nothing left to delete


def test_custom_grace_period_governs_deletion():
    """A caller may narrow the grace window; a blob just past the custom window
    is deletable even though it would be within the default."""
    blob_store, record_store = _stores()

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
    stats = _run(sweep_asset_backed_orphan_blobs(blob_store, record_store, config, now=future))
    assert stats.deleted == 1
    assert _run(blob_store.stat(digest=sha)) is None


def _run(coro):
    import asyncio

    return asyncio.run(coro)
