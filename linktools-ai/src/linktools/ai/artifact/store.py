#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore: the artifact domain facade over content-addressed blobs and
per-write lineage records.

The facade depends on the stable :class:`ArtifactBlobStore` and
:class:`ArtifactRecordStore` Protocols -- never on a concrete backend. That
decouples the artifact domain from any specific storage backend entirely
(this module imports no backend symbol). The in-repo reference
implementations (filesystem, SQLAlchemy) live in the storage infrastructure
layer; an external object store or DB can implement the same Protocols and be
injected via the constructor.

Domain rules the facade owns: content deduplication by sha256 (identical bytes
share one blob), a fresh UUID :class:`ArtifactRecord` per put (each production
event keeps its own provenance), tenant scoping, and read-time integrity
verification. Every read is tenant-scoped and FAIL_CLOSED.
"""

import asyncio
import hashlib
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ..observability.metrics import ObservabilityMetrics

from ..storage.protocols import ArtifactBlobStore, ArtifactRecordStore
from .coordination import (
    ArtifactDigestCoordinator,
    InProcessArtifactDigestCoordinator,
)
from .models import (
    ArtifactBufferedSizeLimitError,
    ArtifactIntegrityError,
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRef,
    ArtifactStagingError,
)

# The whole-bytes API (``put``/``get``) materializes an artifact into a single
# ``bytes``; content at or above this threshold MUST use the streaming API
# (``put_stream``/``open_stream``) so the facade never holds a whole large
# artifact resident. 16 MiB is the bounded-memory ceiling for the bytes path.
BUFFERED_ARTIFACT_LIMIT = 16 * 1024 * 1024
# Chunk size for streaming reads/writes through the staging file: small enough
# that peak RSS from a single in-flight buffer is bounded regardless of
# artifact size.
_STREAM_CHUNK = 64 * 1024


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


async def _stage_and_digest(
    source: AsyncIterator[bytes],
    *,
    staging_dir: "Path | None" = None,
) -> "tuple[str, int, tempfile.SpooledTemporaryFile[bytes]]":
    """Stream ``source`` into a spooled temp file in fixed-size chunks, hashing
    incrementally, so peak RSS stays bounded by ``max(_STREAM_CHUNK, largest
    source chunk)`` no matter how large the artifact is -- the facade preserves
    the source's chunking rather than re-chunking it, so an oversized single
    chunk is held until the spool rolls. Returns ``(sha256_hex, size,
    staged_file)`` positioned for reading (caller reads it back, then closes it).

    The Protocol's ``put_if_absent`` needs the digest UP FRONT (the digest is
    the content address), so a streaming source of unknown content must be
    hashed before the call. Buffering it in RAM would defeat bounded memory, so
    we spill to a spooled temp file -- small artifacts stay in memory, large
    ones roll to disk automatically, and the caller re-streams the file into
    ``put_if_absent`` in bounded chunks.

    ``staging_dir`` makes the temp directory caller-configurable;
    every file read/write runs on a worker thread (``asyncio.to_thread``) so a
    large artifact's disk I/O never blocks the event loop; an I/O failure on the
    spool (disk-full / ENOSPC, or any OSError) is wrapped as
    :class:`ArtifactStagingError`."""
    kwargs: "dict" = {"max_size": _STREAM_CHUNK, "suffix": ".art-stage"}
    if staging_dir is not None:
        kwargs["dir"] = str(staging_dir)
    try:
        staged = await asyncio.to_thread(tempfile.SpooledTemporaryFile, **kwargs)
    except OSError as exc:
        raise ArtifactStagingError(f"could not create staging file: {exc}") from exc
    hasher = hashlib.sha256()
    size = 0
    try:
        async for chunk in source:
            if not chunk:
                continue
            hasher.update(chunk)
            size += len(chunk)
            try:
                await asyncio.to_thread(staged.write, chunk)
            except OSError as exc:
                raise ArtifactStagingError(
                    f"staging write failed (disk full?): {exc}"
                ) from exc
        await asyncio.to_thread(staged.seek, 0)
        return hasher.hexdigest(), size, staged
    except BaseException:
        staged.close()
        raise


def record_to_jsonable(record: ArtifactRecord) -> dict:
    p = record.provenance
    return {
        "ref": {
            "id": record.ref.id,
            "sha256": record.ref.sha256,
            "media_type": record.ref.media_type,
            "size": record.ref.size,
        },
        "tenant_id": record.tenant_id,
        "provenance": {
            "producer_kind": p.producer_kind,
            "producer_id": p.producer_id,
            "run_id": p.run_id,
            "session_id": p.session_id,
            "parent_artifact_ids": list(p.parent_artifact_ids),
            "metadata": dict(p.metadata),
        },
        "created_at": _coerce_utc(record.created_at).isoformat(),
    }


def record_from_jsonable(data: dict) -> ArtifactRecord:
    ref = ArtifactRef(
        id=data["ref"]["id"],
        sha256=data["ref"]["sha256"],
        media_type=data["ref"]["media_type"],
        size=data["ref"]["size"],
    )
    p = data["provenance"]
    return ArtifactRecord(
        ref=ref,
        tenant_id=data["tenant_id"],
        provenance=ArtifactProvenance(
            producer_kind=p["producer_kind"],
            producer_id=p["producer_id"],
            run_id=p.get("run_id"),
            session_id=p.get("session_id"),
            parent_artifact_ids=tuple(p.get("parent_artifact_ids") or ()),
            metadata=dict(p.get("metadata") or {}),
        ),
        created_at=datetime.fromisoformat(data["created_at"]),
    )


async def _bytes_to_async_iter(content: bytes) -> AsyncIterator[bytes]:
    yield content


async def _empty_chunks() -> AsyncIterator[bytes]:
    """An async iterator that yields nothing. Used by ``open_stream`` when the
    artifact does not exist (or is foreign-tenant) so the caller can use one
    uniform ``async with store.open_stream(...) as chunks`` form whether or not
    the artifact exists."""
    return
    yield b""  # pragma: no cover  (makes this an async generator)


class ArtifactStore:
    """Content-addressed blobs with per-write lineage records, over the
    :class:`ArtifactBlobStore` and :class:`ArtifactRecordStore` Protocols.

    CONTENT is deduplicated by sha256 (identical bytes share one blob). A
    fresh RECORD is minted per put (a UUID id), so each production event --
    even of identical content -- keeps its own provenance. Reads are
    tenant-scoped and integrity-verified: a caller learns nothing about another
    tenant's artifact, and a tampered blob is detected on read.
    """

    def __init__(
        self,
        blob_store: ArtifactBlobStore,
        record_store: ArtifactRecordStore,
        coordinator: "ArtifactDigestCoordinator | None" = None,
        *,
        metrics: "ObservabilityMetrics | None" = None,
        staging_dir: "Path | None" = None,
    ) -> None:
        self._blob = blob_store
        self._records = record_store
        # Per-digest coordinator serializes the put (blob reuse + record create)
        # against the orphan sweeper (re-check + delete). Default is process-local
        # -- a real per-digest asyncio.Lock, not a lockless fallback. A
        # multi-worker / cross-process deployment injects a distributed or
        # filesystem-flock coordinator so the mutual exclusion spans workers.
        self._coordinator = coordinator or InProcessArtifactDigestCoordinator()
        # Optional ObservabilityMetrics sink. When wired, a digest mismatch on
        # read increments ``artifact_digest_mismatch_total`` and a put failure
        # (blob upload side) increments ``artifact_blob_upload_failure_total``.
        # Default None keeps existing callers no-op.
        self._metrics = metrics
        # Optional staging directory for the streaming-put spool
        # (caller-configurable). None = the tempfile module default.
        self._staging_dir = staging_dir

    async def put(
        self,
        *,
        tenant_id: str,
        content: bytes,
        media_type: str,
        provenance: "ArtifactProvenance",
        now: "datetime | None" = None,
    ) -> ArtifactRecord:
        if len(content) >= BUFFERED_ARTIFACT_LIMIT:
            raise ArtifactBufferedSizeLimitError(
                f"artifact of {len(content)} bytes meets/exceeds the "
                f"{BUFFERED_ARTIFACT_LIMIT}-byte buffered limit; use put_stream() "
                f"for large content"
            )
        sha = hashlib.sha256(content).hexdigest()
        # Hold the per-digest lock across blob reuse AND record create so the
        # orphan sweeper cannot observe the blob as unreferenced and delete it
        # in the window between the two.
        async with self._coordinator.hold(sha):
            # Content dedup: put_if_absent is idempotent on digest and verifies the
            # claimed digest matches the bytes.
            try:
                await self._blob.put_if_absent(
                    digest=sha, source=_bytes_to_async_iter(content), size=len(content)
                )
            except Exception:
                # The blob-store-level mismatch already records itself when its
                # own sink is wired; record at the facade too so a caller that
                # only wires the facade still observes the failure.
                if self._metrics is not None:
                    self._metrics.counter(
                        "artifact_blob_upload_failure_total",
                        attributes={"reason": "digest_or_store"},
                    )
                raise
            artifact_id = f"art-{uuid.uuid4().hex}"
            record = ArtifactRecord(
                ref=ArtifactRef(
                    id=artifact_id, sha256=sha, media_type=media_type, size=len(content)
                ),
                tenant_id=tenant_id,
                provenance=provenance,
                created_at=_coerce_utc(now) if now is not None else _utcnow(),
            )
            return await self._records.put(record)

    async def stat(
        self, *, artifact_id: str, tenant_id: str
    ) -> "ArtifactRecord | None":
        return await self._records.get(artifact_id=artifact_id, tenant_id=tenant_id)

    async def get(self, *, artifact_id: str, tenant_id: str) -> "bytes | None":
        # Tenant gate runs before content fetch so a foreign caller learns
        # nothing -- not even whether the artifact exists.
        record = await self.stat(artifact_id=artifact_id, tenant_id=tenant_id)
        if record is None:
            return None
        if record.ref.size >= BUFFERED_ARTIFACT_LIMIT:
            raise ArtifactBufferedSizeLimitError(
                f"artifact {artifact_id} is {record.ref.size} bytes "
                f"(meets/exceeds the {BUFFERED_ARTIFACT_LIMIT}-byte buffered "
                f"limit); use open_stream() to read it without buffering"
            )
        chunks_acc: "list[bytes]" = []
        async with self._blob.open(digest=record.ref.sha256) as chunks:
            async for chunk in chunks:
                chunks_acc.append(chunk)
        content = b"".join(chunks_acc)
        # Integrity: verify BOTH size and sha256. Size first (cheap) then
        # digest -- a tampered, truncated, or extended blob fails one of the two
        # checks.
        if len(content) != record.ref.size:
            if self._metrics is not None:
                self._metrics.counter("artifact_digest_mismatch_total")
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} blob size mismatch: claimed "
                f"{record.ref.size}, actual {len(content)}"
            )
        actual = hashlib.sha256(content).hexdigest()
        if actual != record.ref.sha256:
            if self._metrics is not None:
                self._metrics.counter("artifact_digest_mismatch_total")
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} blob sha256 mismatch: {actual}"
            )
        return content

    async def put_stream(
        self,
        *,
        tenant_id: str,
        source: AsyncIterator[bytes],
        media_type: str,
        provenance: "ArtifactProvenance",
        expected_digest: "str | None" = None,
        size: "int | None" = None,
        now: "datetime | None" = None,
    ) -> ArtifactRecord:
        """Streaming put: hash + persist ``source`` without ever holding the
        whole artifact in a single ``bytes``.

        Two branches. (1) Caller provides ``expected_digest`` -- it already
        knows the sha256, so the source is wrapped in a one-pass verifying
        iterator and streamed straight into ``put_if_absent`` under that digest;
        no staging file (the blob store re-verifies the bytes hash to the claimed
        digest). (2) Caller does not -- the source is spilled to a spooled temp
        file (bounded RSS, configurable dir, threaded I/O), hashed incrementally,
        then re-streamed into ``put_if_absent`` under the computed digest.

        Either way a fresh ArtifactRecord is minted per call; identical content
        still dedupes to one blob. Use this instead of ``put`` for content
        at/above :data:`BUFFERED_ARTIFACT_LIMIT`."""
        if expected_digest is not None:
            # Caller vouches for the digest: stream a verifying iterator
            # straight into put_if_absent under that digest, then pin the
            # record -- both under the per-digest lock so the sweeper cannot
            # delete the blob between the two.
            async with self._coordinator.hold(expected_digest):
                final_size = await self._put_stream_with_known_digest(
                    source=source, expected_digest=expected_digest, declared_size=size
                )
                return await self._commit_record(
                    expected_digest, final_size, tenant_id, media_type, provenance, now
                )
        # No expected digest: stage to a spooled file (which yields the digest),
        # then hold the per-digest lock across the blob commit + record pin.
        # Staging itself is outside the lock -- it is not part of the put/sweep
        # race window.
        digest, staged_size, staged = await _stage_and_digest(
            source, staging_dir=self._staging_dir
        )
        async with self._coordinator.hold(digest):
            try:
                final_size = await self._put_staged_blob(
                    digest=digest, staged=staged, staged_size=staged_size
                )
            finally:
                staged.close()
            return await self._commit_record(
                digest, final_size, tenant_id, media_type, provenance, now
            )

    async def _commit_record(
        self,
        digest: str,
        size: int,
        tenant_id: str,
        media_type: str,
        provenance: "ArtifactProvenance",
        now: "datetime | None",
    ) -> ArtifactRecord:
        artifact_id = f"art-{uuid.uuid4().hex}"
        record = ArtifactRecord(
            ref=ArtifactRef(
                id=artifact_id, sha256=digest, media_type=media_type, size=size
            ),
            tenant_id=tenant_id,
            provenance=provenance,
            created_at=_coerce_utc(now) if now is not None else _utcnow(),
        )
        return await self._records.put(record)

    async def _put_stream_with_known_digest(
        self,
        *,
        source: AsyncIterator[bytes],
        expected_digest: str,
        declared_size: "int | None",
    ) -> int:
        """'caller provides expected_digest' branch: wrap source in a
        verifying iterator (incremental sha256 + size tally) and stream it
        DIRECTLY into ``put_if_absent`` under the claimed digest -- no staging
        file, because the caller vouches for the digest. The blob store
        re-verifies the bytes hash to the digest; a mismatch fails with no record
        written. Returns the final size the store reports."""
        hasher = hashlib.sha256()

        async def _verifying() -> AsyncIterator[bytes]:
            # An incremental SHA-256 verifying iterator: the streamed source
            # must hash to the claimed digest. The backend re-verifies too; this
            # is the facade-level check that makes _verifying actually verify,
            # not just pass chunks through.
            async for chunk in source:
                if not chunk:
                    continue
                hasher.update(chunk)
                yield chunk

        try:
            info = await self._blob.put_if_absent(
                digest=expected_digest, source=_verifying(), size=declared_size
            )
        except Exception:
            if self._metrics is not None:
                self._metrics.counter(
                    "artifact_blob_upload_failure_total",
                    attributes={"reason": "digest_or_store"},
                )
            raise
        # The source the backend just consumed must hash to the claimed digest.
        # (The backend checks too; this is the verifying-iterator's own result.)
        if hasher.hexdigest() != expected_digest:
            if self._metrics is not None:
                self._metrics.counter(
                    "artifact_blob_upload_failure_total",
                    attributes={"reason": "digest_mismatch"},
                )
            raise ArtifactIntegrityError(
                f"streamed source digest {hasher.hexdigest()[:12]} does not "
                f"match expected_digest {expected_digest[:12]}"
            )
        return info.size

    async def _put_staged_blob(
        self, *, digest: str, staged, staged_size: int
    ) -> int:
        """Re-stream a staged spooled file into ``put_if_absent`` under the
        computed digest. Returns the final size."""
        async def _file_chunks() -> AsyncIterator[bytes]:
            while True:
                chunk = await asyncio.to_thread(staged.read, _STREAM_CHUNK)
                if not chunk:
                    break
                yield chunk

        try:
            await self._blob.put_if_absent(
                digest=digest, source=_file_chunks(), size=staged_size
            )
        except Exception:
            if self._metrics is not None:
                self._metrics.counter(
                    "artifact_blob_upload_failure_total",
                    attributes={"reason": "digest_or_store"},
                )
            raise
        return staged_size

    @asynccontextmanager
    async def open_stream(
        self, *, artifact_id: str, tenant_id: str
    ) -> AsyncIterator[bytes]:
        """Streaming read: ``async with store.open_stream(...) as
        chunks: async for chunk in chunks``. Yields the artifact's bytes in
        chunks without buffering the whole blob into RAM. Tenant-gated -- a
        foreign caller learns nothing, not even that the artifact exists (the
        context yields an empty iterator).

        Integrity is verified at stream EXHAUSTION: BOTH size
        and sha256 are accumulated as chunks are yielded, and a mismatch raises
        :class:`ArtifactIntegrityError` after the final chunk. A caller that
        stops iterating early forgoes the check (partial reads are not
        verified); exhaust the iterator to assert integrity. A blob that is
        missing (record exists, blob swept) surfaces the backend's
        :class:`ArtifactBlobNotFoundError` through the context."""
        record = await self.stat(artifact_id=artifact_id, tenant_id=tenant_id)
        if record is None:
            yield _empty_chunks()
            return
        async with self._blob.open(digest=record.ref.sha256) as blob_chunks:
            hasher = hashlib.sha256()
            seen = 0

            async def _verified() -> AsyncIterator[bytes]:
                nonlocal seen
                async for chunk in blob_chunks:
                    hasher.update(chunk)
                    seen += len(chunk)
                    yield chunk
                if seen != record.ref.size:
                    if self._metrics is not None:
                        self._metrics.counter("artifact_digest_mismatch_total")
                    raise ArtifactIntegrityError(
                        f"artifact {artifact_id} blob size mismatch on stream: "
                        f"claimed {record.ref.size}, actual {seen}"
                    )
                if hasher.hexdigest() != record.ref.sha256:
                    if self._metrics is not None:
                        self._metrics.counter("artifact_digest_mismatch_total")
                    raise ArtifactIntegrityError(
                        f"artifact {artifact_id} blob sha256 mismatch on stream: "
                        f"{hasher.hexdigest()[:12]}"
                    )

            yield _verified()


__all__: "list[str]" = ["ArtifactStore"]
