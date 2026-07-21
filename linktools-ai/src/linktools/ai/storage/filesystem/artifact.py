#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem reference implementations of the artifact Protocols.

Content-addressed blobs are stored unsharded-by-prefix on disk under
``blobs_root/<xx>/<sha256>`` (the two-hex-char shard keeps any single directory
small); tenant-scoped lineage records are JSON files under
``records_root/<tenant>/<id>.json``. Both reuse the crash-safe atomic-write
helper used by every other File store (same-dir temp, fsync, ``os.replace``,
parent-dir fsync).

The blob write path streams the source into the temp file in fixed-size chunks
(never holding the whole blob in a single ``bytes``), hashes incrementally to
verify the claimed digest, and only then publishes -- so a digest mismatch
leaves no file at the final path. ``open`` streams the file back in chunks
through the Protocol's async-context-manager shape."""

import hashlib
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ...observability.metrics import ObservabilityMetrics

from ...artifact.models import (
    ArtifactBlobNotFoundError,
    ArtifactIntegrityError,
    ArtifactRecord,
)
from ...artifact.store import record_from_jsonable, record_to_jsonable
from ..protocols import ArtifactBlobStore, ArtifactRecordStore, BlobInfo
from .atomic import _fsync_directory

_CHUNK = 64 * 1024


def _safe_component(value: str, kind: str) -> str:
    """tenant_id / artifact_id / digest become path components; reject anything
    that could escape the store root (empty, path separators, ``.``/``..``).
    Defense-in-depth so a caller cannot traverse out of the records/blobs tree
    by crafting an id."""
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FilesystemArtifactBlobStore:
    """ArtifactBlobStore backed by the local filesystem. Blobs are
    content-addressed by sha256 under ``blobs_root/<xx>/<sha>``;
    ``put_if_absent`` streams to a same-dir temp file, verifies the digest, and
    publishes atomically; ``open`` streams the file back in bounded chunks."""

    def __init__(
        self,
        *,
        blobs_root: Path,
        chunk_size: int = _CHUNK,
        metrics: "ObservabilityMetrics | None" = None,
    ) -> None:
        self._root = Path(blobs_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._chunk = chunk_size
        # Optional ObservabilityMetrics sink. When wired, a digest mismatch, a
        # size mismatch, or a corrupt existing blob on put_if_absent increments
        # ``artifact_blob_upload_failure_total``. Default None = no-op so callers
        # that do not wire a sink keep their no-metrics behavior.
        self._metrics = metrics

    def _path(self, digest: str) -> Path:
        _safe_component(digest, "digest")
        return self._root / digest[:2] / digest

    def _sha256_file(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(self._chunk)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        final = self._path(digest)
        if final.exists():
            # Dedup: an existing blob at this address must hash back to the
            # digest, else it is corrupt and we refuse to record a reference.
            actual = self._sha256_file(final)
            if actual != digest:
                if self._metrics is not None:
                    self._metrics.counter(
                        "artifact_blob_upload_failure_total",
                        attributes={"reason": "corrupt"},
                    )
                raise ArtifactIntegrityError(
                    f"blob at sha256 {digest[:12]} is corrupt (actual {actual[:12]}); "
                    f"refusing to record a reference to it"
                )
            return BlobInfo(digest=digest, size=final.stat().st_size, content_type=None)
        # Not present: stream the source into a same-dir temp, hashing as we go,
        # verify the digest, then atomically publish.
        final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(final.parent), prefix=f".{final.name}.", suffix=".tmp"
        )
        hasher = hashlib.sha256()
        written = 0
        try:
            with os.fdopen(fd, "wb") as f:
                async for chunk in source:
                    if not chunk:
                        continue
                    hasher.update(chunk)
                    written += len(chunk)
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            actual = hasher.hexdigest()
            if actual != digest:
                if self._metrics is not None:
                    self._metrics.counter(
                        "artifact_blob_upload_failure_total",
                        attributes={"reason": "digest_mismatch"},
                    )
                raise ArtifactIntegrityError(
                    f"blob digest mismatch: claimed {digest[:12]}, actual {actual[:12]}"
                )
            if size is not None and size != written:
                if self._metrics is not None:
                    self._metrics.counter(
                        "artifact_blob_upload_failure_total",
                        attributes={"reason": "size_mismatch"},
                    )
                raise ArtifactIntegrityError(
                    f"blob size mismatch: claimed {size}, actual {written}"
                )
            os.replace(tmp_name, final)
            _fsync_directory(final.parent)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        return BlobInfo(digest=digest, size=written, content_type=None)

    @asynccontextmanager
    async def open(self, *, digest: str):
        path = self._path(digest)
        if not path.exists():
            raise ArtifactBlobNotFoundError(
                f"blob for sha256 {digest[:12]} missing"
            )
        f = open(path, "rb")
        try:
            chunk = self._chunk

            async def _chunks() -> AsyncIterator[bytes]:
                while True:
                    block = f.read(chunk)
                    if not block:
                        break
                    yield block

            yield _chunks()
        finally:
            f.close()

    async def stat(self, *, digest: str) -> "BlobInfo | None":
        path = self._path(digest)
        if not path.exists():
            return None
        return BlobInfo(digest=digest, size=path.stat().st_size, content_type=None)

    async def delete(self, *, digest: str) -> None:
        path = self._path(digest)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    async def iter_digests_with_mtime(self) -> "AsyncIterator[tuple[str, datetime]]":
        """Yield ``(digest, modified_at)`` for every stored blob, for orphan
        sweeping. Walks the two-hex-char shard dirs; the digest is the file
        name (the sha256 it was filed under)."""
        if not self._root.exists():
            return
        for shard in sorted(self._root.iterdir()):
            if not shard.is_dir():
                continue
            for blob in sorted(shard.iterdir()):
                if not blob.is_file():
                    continue
                yield blob.name, datetime.fromtimestamp(
                    blob.stat().st_mtime, tz=timezone.utc
                )


class FilesystemArtifactRecordStore:
    """ArtifactRecordStore backed by the local filesystem. Records are
    tenant-scoped JSON files under ``records_root/<tenant>/<id>.json``. The
    ArtifactStore facade mints a fresh UUID id per put, so each production event
    keeps its own lineage file."""

    def __init__(self, *, records_root: Path) -> None:
        self._root = Path(records_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, tenant_id: str, artifact_id: str) -> Path:
        _safe_component(tenant_id, "tenant_id")
        _safe_component(artifact_id, "artifact_id")
        return self._root / tenant_id / f"{artifact_id}.json"

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        from .atomic import atomic_write_bytes

        payload = json.dumps(record_to_jsonable(record)).encode("utf-8")
        atomic_write_bytes(self._path(record.tenant_id, record.ref.id), payload)
        return record

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        path = self._path(tenant_id, artifact_id)
        if not path.exists():
            return None
        record = record_from_jsonable(json.loads(path.read_text(encoding="utf-8")))
        if record.tenant_id != tenant_id:
            return None
        return record

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        path = self._path(tenant_id, artifact_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    async def iter_referenced_digests(self) -> AsyncIterator[str]:
        """Yield every sha256 referenced by some record, for orphan sweeping.
        Walks the tenant dirs, reads each record JSON via the public codec, and
        emits its pinned digest -- the set of blobs that are NOT orphans."""
        if not self._root.exists():
            return
        for tenant_dir in sorted(self._root.iterdir()):
            if not tenant_dir.is_dir():
                continue
            for record_file in sorted(tenant_dir.glob("*.json")):
                try:
                    record = record_from_jsonable(
                        json.loads(record_file.read_text(encoding="utf-8"))
                    )
                except (ValueError, KeyError, TypeError):
                    continue
                yield record.ref.sha256

    async def _iter_records(
        self, *, tenant_id: "str | None" = None
    ) -> AsyncIterator[ArtifactRecord]:
        """Walk every record (optionally tenant-scoped) in stable order. The
        Filesystem backend has no index columns, so parent/provenance queries
        resolve by loading each record via the public codec and filtering in
        memory -- honest O(records) scans, not a fake indexed lookup."""
        if not self._root.exists():
            return
        tenant_dirs = (
            [self._root / tenant_id] if tenant_id is not None
            else sorted(p for p in self._root.iterdir() if p.is_dir())
        )
        for tenant_dir in tenant_dirs:
            if not tenant_dir.is_dir():
                continue
            for record_file in sorted(tenant_dir.glob("*.json")):
                try:
                    record = record_from_jsonable(
                        json.loads(record_file.read_text(encoding="utf-8"))
                    )
                except (ValueError, KeyError, TypeError):
                    continue
                yield record

    async def iter_by_run_id(
        self, run_id: "str | None", *, tenant_id: "str | None" = None
    ) -> AsyncIterator[ArtifactRecord]:
        """Yield every record produced under ``run_id``, optionally
        tenant-scoped. A ``None`` run_id yields records with no run attribution
        (provenance.run_id is None), matching the SqlAlchemy column semantics."""
        async for record in self._iter_records(tenant_id=tenant_id):
            if record.provenance.run_id == run_id:
                yield record

    async def iter_by_producer(
        self,
        producer_kind: str,
        producer_id: "str | None" = None,
        *,
        tenant_id: "str | None" = None,
    ) -> AsyncIterator[ArtifactRecord]:
        """Yield every record from a given producer (kind [+ id]), optionally
        tenant-scoped. Mirrors the SqlAlchemy (producer_kind, producer_id)
        index as an in-memory filter."""
        async for record in self._iter_records(tenant_id=tenant_id):
            if record.provenance.producer_kind != producer_kind:
                continue
            if producer_id is not None and record.provenance.producer_id != producer_id:
                continue
            yield record


__all__: "list[str]" = [
    "FilesystemArtifactBlobStore",
    "FilesystemArtifactRecordStore",
]
