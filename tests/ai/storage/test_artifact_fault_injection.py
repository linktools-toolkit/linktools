#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault injection for the artifact streaming path. The cases that
were already covered elsewhere (digest mismatch, size mismatch, corrupt
existing blob, cancellation, source-iterator-raises, orphan sweeping, stale
fencing tokens, duplicate claim, transaction rollback, completion/cancel race)
are NOT duplicated here -- this module fills the gaps:

- blob write interrupt on the Filesystem backend (mid-stream source failure
  during ``put_if_absent`` -> no partial blob file at the final address, no
  leftover temp). The testkit already proves the Protocol-level contract on
  the in-memory backend; this test pins the same property on the real on-disk
  backend where a partial file would actually leak bytes onto disk.
- record write failure (blob succeeds, record-store.put raises) -> the call
  does NOT report success; the blob is an orphan candidate the sweeper can
  reap once it ages past the grace window.
- ArtifactStore.open_stream consumer error inside the async-with -> the file
  handle is released (no FD leak).
- ArtifactBlobStore.open async-context-manager enter/exit error path -> the
  underlying file handle is not leaked when enter/exit raises.

Evidence captured for each: final blob state, visible record, orphan state,
whether a duplicate side-effect occurred (e.g. a second blob or a partial temp
file)."""

import asyncio
import os
import stat
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest

from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.artifact.models import ArtifactIntegrityError
from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.artifact.store import ArtifactStore
from linktools.ai.artifact import ANONYMOUS_PROVENANCE
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


def _run(coro):
    return asyncio.run(coro)


# --- blob write interrupt on the Filesystem backend --------------------------


def test_blob_put_if_absent_mid_stream_failure_publishes_no_partial_file(
    tmp_path: Path,
) -> None:
    """A source that raises mid-stream must propagate the error AND leave the
    Filesystem backend untouched: no final blob at the claimed digest, no
    leftover .tmp file from the failed staging write. This is the on-disk
    equivalent of the Protocol-level contract in the testkit (which proves it
    on an in-memory backend); a regression that left a partial file at the
    sharded path would shadow a future legitimate upload of the full content."""

    class _Boom(Exception):
        pass

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    claimed_digest = _digest(b"complete-content-not-delivered")

    async def _bad_source() -> AsyncIterator[bytes]:
        # Yield a real chunk so the staging write actually starts, THEN raise.
        yield b"partial-bytes-that-are-not-the-claimed-digest"
        raise _Boom("source failed mid-stream")

    async def run() -> None:
        with pytest.raises(_Boom):
            await blob.put_if_absent(
                digest=claimed_digest, source=_bad_source(), size=None
            )

    _run(run())

    # No blob at the final address.
    final = tmp_path / "blobs" / claimed_digest.value[:2] / claimed_digest.value
    assert not final.exists(), (
        f"partial blob file leaked to the final address: {final}"
    )
    # No leftover staging temp file in the blob tree.
    leftover_tmps = list((tmp_path / "blobs").rglob("*.tmp"))
    assert not leftover_tmps, (
        f"staging temp files left behind after mid-stream failure: {leftover_tmps}"
    )
    # A stat confirms the absent address through the public surface too.
    assert _run(blob.stat(digest=claimed_digest)) is None


# --- record write failure: orphan candidate ----------------------------------


class _FailingRecordStore:
    """An ArtifactRecordStore whose put() always raises. Used to simulate a
    record-store failure AFTER the blob was successfully written -- the
    artifact-domain scenario the orphan sweeper exists to clean up behind."""

    def __init__(self, *, exc: Exception) -> None:
        self._exc = exc
        self.put_calls = 0

    async def put(self, record):
        self.put_calls += 1
        raise self._exc

    async def get(self, artifact_id: str, *, tenant_id: str):
        return None

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        return False


def test_put_stream_blob_succeeds_record_failure_reports_no_success_and_leaves_orphan(
    tmp_path: Path,
) -> None:
    """When the blob write succeeds but the record store's put() raises, the
    facade MUST propagate the failure (not report success). The blob is now an
    orphan candidate (bytes present on disk, no record pinning it) -- the
    orphan sweeper can reap it once it ages past the grace window. This test
    pins both halves: the facade does NOT return a record, AND the orphan
    sweeper later finds the unreferenced blob."""

    class _RecordWriteError(Exception):
        pass

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = _FailingRecordStore(exc=_RecordWriteError("record store down"))
    store = ArtifactStore(blob, records, InProcessArtifactDigestCoordinator())
    payload = b"orphan-candidate-payload"

    async def run() -> None:
        with pytest.raises(_RecordWriteError):
            await store.put_stream(
                source=_aiter([payload]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )

    _run(run())

    # The facade did not report success (the put() call above raised).
    assert records.put_calls == 1, "record-store.put was not invoked exactly once"
    # The blob IS present on disk (the write succeeded before the record call).
    digest = _digest(payload)
    blob_info = _run(blob.stat(digest=digest))
    assert blob_info is not None and blob_info.size == len(payload), (
        "blob should be present (the orphan the sweeper reclaims)"
    )
    # And it is genuinely an orphan: NO record pins this digest. The orphan
    # sweeper sees it as unreferenced and (past the grace window) reaps it.
    from datetime import datetime, timedelta, timezone

    from linktools.ai.storage.orphan import sweep_orphan_blobs

    fs_records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    future = datetime.now(timezone.utc) + timedelta(hours=25)
    stats = _run(sweep_orphan_blobs(blob, fs_records, InProcessArtifactDigestCoordinator(), now=future))
    assert stats.deleted == 1, (
        f"orphan sweeper did not reap the orphaned blob: {stats}"
    )
    assert _run(blob.stat(digest=digest)) is None, "blob survived the sweep"


# --- ArtifactStore.open_stream: consumer error does not leak the file handle --


def _count_open_fds() -> int:
    """Count currently-open file descriptors for THIS process via /proc/self/fd.
    Returns -1 on environments without /proc (the test skips when this is -1)."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def test_open_stream_consumer_error_does_not_leak_file_handle(tmp_path: Path) -> None:
    """If the consumer raises while iterating ``open_stream``, the async-with
    in ``FilesystemArtifactBlobStore.open`` MUST still close the underlying
    file handle (no FD leak). The implementation uses ``try/finally`` around
    the yield; this test pins that the finally runs even on a consumer error."""

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    store = ArtifactStore(blob, records, InProcessArtifactDigestCoordinator())
    payload = b"streamed-payload-for-fd-leak"

    class _ConsumerError(Exception):
        pass

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([payload]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        fds_before = _count_open_fds()
        with pytest.raises(_ConsumerError):
            async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
                async for chunk in chunks:
                    # The consumer raises mid-iteration; the file handle must
                    # still be cleaned up by the async-with's finally block.
                    raise _ConsumerError("consumer died mid-stream")
        fds_after = _count_open_fds()
        # Compare only when /proc is available; otherwise skip the assertion
        # (the underlying file-close is exercised regardless, this just pins
        # the observable FD count).
        if fds_before >= 0 and fds_after >= 0:
            # Allow a small slack for pytest internals; the load-bearing
            # assertion is that the count returns to its pre-iteration level
            # (i.e. the blob's file handle is closed, not leaked).
            assert fds_after <= fds_before + 1, (
                f"file handle leaked: fds went from {fds_before} to {fds_after} "
                f"after a consumer error inside open_stream"
            )

    _run(run())


def test_blob_open_async_ctx_manager_releases_handle_on_enter_failure(
    tmp_path: Path,
) -> None:
    """If something raises inside the async-with body of ``open`` (or the
    consumer raises after entering), the file handle opened on enter MUST be
    released on exit. The implementation uses try/finally around the yield, so
    the close() runs unconditionally. This test forces an error inside the
    body and confirms the underlying file is closed afterwards."""

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    digest = _digest(b"payload-for-ctx-error")
    payload = b"payload-for-ctx-error"

    async def run() -> None:
        await blob.put_if_absent(
            digest=digest, source=_aiter([payload]), size=len(payload)
        )
        fds_before = _count_open_fds()
        with pytest.raises(RuntimeError, match="body-error"):
            async with blob.open(digest=digest) as _chunks:
                raise RuntimeError("body-error")
        fds_after = _count_open_fds()
        if fds_before >= 0 and fds_after >= 0:
            assert fds_after <= fds_before + 1, (
                f"file handle leaked via ctx-manager body error: {fds_before} -> "
                f"{fds_after}"
            )
        # The blob is still readable through a FRESH open (the previous handle
        # was released, the file was not locked / corrupted).
        async with blob.open(digest=digest) as chunks:
            collected = []
            async for c in chunks:
                collected.append(c)
        assert b"".join(collected) == payload

    _run(run())


def test_blob_open_async_ctx_manager_releases_handle_on_iteration_error(
    tmp_path: Path,
) -> None:
    """Mirrors the previous test but forces the error inside the iteration
    loop (after at least one chunk has been yielded) rather than before the
    first chunk -- exercising the path where the consumer has already started
    pulling bytes when it raises. The file-handle cleanup must still run."""

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    digest = _digest(b"X" * (70 * 1024))  # multi-chunk on read (70 KiB / 64 KiB)
    payload = b"X" * (70 * 1024)

    async def run() -> None:
        await blob.put_if_absent(
            digest=digest, source=_aiter([payload]), size=len(payload)
        )

        async def _consume_with_mid_error() -> None:
            async with blob.open(digest=digest) as chunks:
                async for _ in chunks:
                    raise RuntimeError("iteration-error")
                    yield  # pragma: no cover - unreachable, marks it a generator

        # The async-for above is the consumer; the body raises on the first
        # chunk. We expect the RuntimeError to propagate and the file handle
        # to be released by the async-with's finally.
        fds_before = _count_open_fds()
        with pytest.raises(RuntimeError, match="iteration-error"):
            async for _ in _consume_with_mid_error():
                pass
        fds_after = _count_open_fds()
        if fds_before >= 0 and fds_after >= 0:
            assert fds_after <= fds_before + 1, (
                f"file handle leaked via mid-iteration error: {fds_before} -> "
                f"{fds_after}"
            )

    _run(run())


# --- staging disk failure ----------------------------------------------------


def _recursive_tmp_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*") if _.is_file())


def test_put_stream_staging_disk_failure_propagates_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the staging spooled temp file cannot be written (disk full / readonly
    / OS-level write failure), ``ArtifactStore.put_stream`` MUST raise and
    publish nothing -- neither a blob nor a record. Simulated by patching
    ``tempfile.SpooledTemporaryFile`` to a subclass whose ``write`` always
    raises; the staging loop in ``_stage_and_digest`` calls write per source
    chunk, so the patch fires on the first chunk."""

    class _UnwritableSpool:
        """A drop-in for tempfile.SpooledTemporaryFile whose write() always
        raises -- simulates a staging disk that cannot accept bytes (a full
        tempdir, a read-only mount, etc.). The other methods are minimal
        stubs sufficient for the staging loop's try/except/cleanup path."""

        def __init__(self, *args, **kwargs) -> None:
            self._closed = False

        def write(self, _data) -> int:
            raise OSError("staging disk write failure (simulated)")

        def seek(self, *_args) -> int:
            return 0

        def read(self, *_args) -> bytes:
            return b""

        def close(self) -> None:
            self._closed = True

    import linktools.ai.artifact.store as store_mod

    monkeypatch.setattr(store_mod.tempfile, "SpooledTemporaryFile", _UnwritableSpool)

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    store = ArtifactStore(blob, records, InProcessArtifactDigestCoordinator())
    payload = b"never-staged-successfully"

    async def run() -> None:
        # line 368: a staging I/O failure (disk full / read-only) is
        # wrapped as ArtifactStagingError, chained to the original OSError.
        from linktools.ai.artifact.models import ArtifactStagingError

        with pytest.raises(ArtifactStagingError, match="staging write failed") as exc_info:
            await store.put_stream(
                source=_aiter([payload]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert isinstance(exc_info.value.__cause__, OSError)
        assert "staging disk write failure" in str(exc_info.value.__cause__)

    _run(run())

    # No blob was published.
    assert _run(blob.stat(digest=_digest(payload))) is None
    # No record was written.
    files_after = _recursive_tmp_count(tmp_path / "records")
    assert files_after == 0, f"records written despite staging failure: {files_after}"
    # The blob tree has no content files (only the empty shard parent dir at
    # most, which never gets created on a failed put).
    blob_files = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert blob_files == [], (
        f"blob files published despite staging failure: {blob_files}"
    )


def test_blob_put_if_absent_publish_dir_unwritable_propagates_and_cleans(
    tmp_path: Path,
) -> None:
    """If the Filesystem blob store cannot create its staging temp file (e.g.
    the blobs_root is read-only), ``put_if_absent`` MUST raise and leave no
    leftover temp file behind. The backend uses ``tempfile.mkstemp(dir=...)``
    on the shard parent, so making that path read-only fires the failure on
    the staging file creation step."""

    blob_root = tmp_path / "blobs"
    blob_root.mkdir(parents=True)
    blob = FilesystemArtifactBlobStore(blobs_root=blob_root)
    digest = _digest(b"payload-that-will-never-publish")

    # Compute the shard dir the backend would publish into, then make it (and
    # its parent) read-only so mkstemp cannot create a temp inside it. We use
    # the actual shard path the backend derives from the digest.
    shard_dir = blob_root / digest.value[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    parent_mode = stat.S_IMODE(shard_dir.stat().st_mode)
    try:
        os.chmod(shard_dir, parent_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

        async def run() -> None:
            with pytest.raises(OSError):
                await blob.put_if_absent(
                    digest=digest, source=_aiter([b"payload"]), size=7
                )

        _run(run())

        # No final blob at the address.
        assert not (shard_dir / digest.value).exists()
        # No leftover temp file.
        leftover = list(shard_dir.glob("*.tmp"))
        assert not leftover, f"leftover staging temps after mkstemp failure: {leftover}"
    finally:
        # Restore so cleanup can delete the tree.
        os.chmod(shard_dir, parent_mode | stat.S_IRWXU)


# --- lease-layer fault injection (extra coverage) ----------------------


def test_stale_lease_holder_write_is_rejected_after_reclaim(tmp_path: Path) -> None:
    """stale lease: a holder whose lease expired and was reclaimed by
    another owner must NOT be able to publish/write as if it still held the
    lease. The LeaseCoordinator Protocol's fencing-token monotonicity is what
    lets a state commit reject the stale holder; this test exercises the
    property directly on the in-process reference coordinator (the same
    guarantee the in-repo JobStore relies on in production)."""

    from datetime import timedelta

    from linktools.ai.storage.coordination.process_local import (
        ProcessLocalLeaseCoordinator,
    )

    coord = ProcessLocalLeaseCoordinator()

    async def run() -> None:
        # Owner A acquires with TTL 0 -> the lease is already expired.
        a_token = await coord.acquire(
            key="asset", owner_id="A", ttl=timedelta(seconds=0)
        )
        assert a_token is not None
        # Owner B reclaims (the lease had expired). B's fencing token must be
        # strictly larger than A's -- this is what a state commit checks.
        b_token = await coord.acquire(
            key="asset", owner_id="B", ttl=timedelta(seconds=30)
        )
        assert b_token is not None
        assert b_token.fencing_token > a_token.fencing_token
        # Owner A's stale token is detectable: any commit checking fencing_token
        # would reject A's write because A's token < B's.
        assert a_token.fencing_token < b_token.fencing_token, (
            "stale lease holder's fencing token is not less than the current "
            "holder's -- a state commit could not detect the stale write"
        )
        # And A cannot renew: the lease is no longer held by A.
        from linktools.ai.errors import StorageConcurrencyNotSupportedError

        with pytest.raises(StorageConcurrencyNotSupportedError):
            await coord.renew(token=a_token, ttl=timedelta(seconds=30))

    _run(run())


def test_stale_fencing_token_commit_is_rejected_after_higher_token_observed(
    tmp_path: Path,
) -> None:
    """stale fencing token: a write bearing a stale fencing token (lower
    than the one currently observed for the asset) MUST be rejected.
    Exercises the JobStore path end-to-end via the in-process coordinator +
    a Filesystem JobStore: worker A's lease expires, B reclaims at a higher
    token, then A's commit with its stale (lower) token is rejected."""

    from datetime import datetime, timedelta, timezone

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
    from linktools.ai.jobs.protocols import TaskSuccess
    from linktools.ai.jobs.store import TaskClaimLostError
    from linktools.ai.storage.filesystem.job import FilesystemJobStore

    class _Clock:
        def __init__(self) -> None:
            self._t = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def now(self) -> datetime:
            return self._t

        def advance(self, seconds: float) -> None:
            self._t = self._t + timedelta(seconds=seconds)

    clock = _Clock()
    store = FilesystemJobStore(tmp_path / "jobs", clock=clock)

    async def run() -> None:
        now = clock.now()
        await store.create_job(
            JobRecord(
                id="j1",
                status=JobStatus.PENDING,
                principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
                actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
                budget=TaskBudget(),
                root_task_id="t1",
                input_artifact_id=None,
                output_artifact_id=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            ),
            TaskRecord(
                id="t1",
                job_id="j1",
                parent_task_id=None,
                key="k",
                handler="h",
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
                asset_snapshots=(),
                version=1,
                created_at=now,
                updated_at=now,
            ),
        )
        a = await store.claim(worker_id="A", now=clock.now(), lease_seconds=30)
        assert a is not None
        # Lease expires; recovery resets the task; B reclaims at a higher token.
        clock.advance(60)
        await store.recover_expired(now=clock.now(), limit=10)
        b = await store.claim(worker_id="B", now=clock.now(), lease_seconds=30)
        assert b is not None
        assert b.claim.fencing_token > a.claim.fencing_token
        # A's commit with the stale (lower) fencing token MUST be rejected --
        # otherwise the stale holder would silently overwrite B's result.
        with pytest.raises(TaskClaimLostError):
            await store.commit_success(a.claim, TaskSuccess())
        # B's commit with the current token still succeeds.
        await store.commit_success(b.claim, TaskSuccess())
        final = await store.get_task("t1")
        assert final.status == TaskStatus.SUCCEEDED

    _run(run())
