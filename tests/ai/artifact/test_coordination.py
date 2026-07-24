#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact digest-coordination concurrency: the per-digest lock must make the
put path (blob reuse + record create) and the orphan sweep (re-check + delete)
mutually exclusive per digest.

The load-bearing scenario: a blob that is ALREADY an orphan candidate (past the
grace window, unreferenced) must NOT be deleted while a concurrent put is
mid-flight pinning it. With coordination, the sweeper blocks on the per-digest
lock until the put creates its record, then re-checks ``is_digest_referenced``
under the lock and keeps the blob. Without coordination (no lock) the sweeper
would see the stale "unreferenced" snapshot and delete the blob the put is about
to pin."""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.artifact.models import ArtifactProvenance
from linktools.ai.artifact.store import ArtifactStore
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)
from linktools.ai.storage.orphan import OrphanSweepConfig, sweep_orphan_blobs


def _sha(content: bytes) -> ArtifactDigest:
    return ArtifactDigest.from_bytes(content)


async def _aiter(content: bytes):
    yield content


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(producer_kind="anonymous", producer_id="")


class _PausableRecords:
    """Wraps a record store so its ``put`` signals then blocks on an event
    BEFORE creating the record -- simulating the window between blob reuse and
    record pin while the ArtifactStore holds the per-digest lock."""

    def __init__(self, inner, entered, proceed):
        self._inner = inner
        self._entered = entered
        self._proceed = proceed

    async def put(self, record):
        self._entered.set()
        await self._proceed.wait()
        return await self._inner.put(record)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.asyncio
async def test_sweep_keeps_blob_a_concurrent_put_is_pinning(tmp_path):
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    coord = InProcessArtifactDigestCoordinator()

    content = b"reused-blob"
    sha = _sha(content)
    # Pre-seed the blob as an UNREFERENCED orphan (no record yet).
    await blob.put_if_absent(digest=sha, source=_aiter(content), size=len(content))

    entered = asyncio.Event()
    proceed = asyncio.Event()
    store = ArtifactStore(blob, _PausableRecords(records, entered, proceed), coord)

    future = datetime.now(timezone.utc) + timedelta(hours=25)

    async def _put():
        # put holds the per-digest lock across blob reuse + record create. The
        # pausable records hold inside that lock until `proceed` is set.
        await store.put(
            tenant_id="t1",
            content=content,
            media_type="",
            provenance=_provenance(),
        )

    async def _sweep():
        # Wait until the put has entered its critical section (holds the lock),
        # then sweep: it must block on coordinator.hold(sha) until the put
        # releases it.
        await entered.wait()
        return await sweep_orphan_blobs(
            blob, records, coord, OrphanSweepConfig(grace_period=timedelta(0)), now=future
        )

    sweep_task = asyncio.create_task(_sweep())
    put_task = asyncio.create_task(_put())
    # Let the put finish (create the record + release the lock); the sweep then
    # acquires the lock, re-checks under it, and finds the blob referenced.
    await asyncio.sleep(0)  # let tasks schedule
    proceed.set()
    stats = await sweep_task
    await put_task

    assert stats.deleted == 0, "sweeper deleted a blob a concurrent put was pinning"
    # The blob survives and is now pinned by the put's record.
    assert await blob.stat(digest=sha) is not None
    assert await records.is_digest_referenced(sha) is True


@pytest.mark.asyncio
async def test_sweep_deletes_a_true_orphan_under_lock(tmp_path):
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    coord = InProcessArtifactDigestCoordinator()

    content = b"true-orphan"
    sha = _sha(content)
    await blob.put_if_absent(digest=sha, source=_aiter(content), size=len(content))

    future = datetime.now(timezone.utc) + timedelta(hours=25)
    stats = await sweep_orphan_blobs(blob, records, coord, now=future)

    assert stats.deleted == 1
    assert await blob.stat(digest=sha) is None


@pytest.mark.asyncio
async def test_per_digest_lock_does_not_serialize_unrelated_digests(tmp_path):
    # Two puts of DIFFERENT digests must run concurrently -- the coordinator
    # shards by digest, so there is no global artifact bottleneck.
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    coord = InProcessArtifactDigestCoordinator()
    store = ArtifactStore(blob, records, coord)

    inflight = asyncio.Semaphore(0)

    async def _slow_put(content):
        async with coord.hold(_sha(content)):
            # An unrelated digest is not blocked while this one is held.
            inflight.release()
            await asyncio.sleep(0.05)

    async with coord.hold(_sha(b"holding")):
        # While digest A is held, both digest-B and digest-C puts should proceed
        # (they do not contend on A's lock).
        b_task = asyncio.create_task(_slow_put(b"B"))
        c_task = asyncio.create_task(_slow_put(b"C"))
        done, pending = await asyncio.wait(
            {asyncio.create_task(inflight.acquire()), asyncio.create_task(inflight.acquire())},
            timeout=2.0,
        )
        assert len(done) == 2, "different-digest puts were serialized by the coordinator"
        await b_task
        await c_task
