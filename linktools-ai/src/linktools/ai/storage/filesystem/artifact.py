#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem reference implementations of the artifact Protocols.

Content-addressed blobs live under ``blobs_root/<xx>/<sha256>`` (the two-hex-char
shard keeps any single directory small); tenant-scoped lineage records are JSON
files under ``records_root/<tenant>/<id>.json``.

The blob write path ALWAYS consumes and verifies the source: it streams the
source into a same-dir temp file in fixed-size chunks (never holding the whole
blob resident), hashes incrementally, verifies the claimed digest and size, and
only then publishes. If a blob at that address already exists, the source is
still consumed and verified, the existing blob is re-hashed and size-checked,
and the existing BlobInfo is returned -- a put can never skip input validation
by claiming a digest that is already present.

Every blocking disk operation runs on a worker thread via the :mod:`._io`
helpers so a large artifact's I/O never blocks the event loop. Records are
create-only: the same artifact id with identical content is idempotent; a
different value raises :class:`ArtifactRecordConflictError`. A corrupt record
file (unparseable, missing fields, mismatched id/tenant, malformed digest)
raises :class:`ArtifactRecordCorruptError` and is never silently skipped, so
the orphan sweeper cannot mistake a broken record for an unreferenced blob."""

import hashlib
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ...observability.metrics import ObservabilityMetrics

from ...artifact.models import (
    ArtifactBlobNotFoundError,
    ArtifactIntegrityError,
    ArtifactRecord,
    ArtifactRecordConflictError,
    ArtifactRecordCorruptError,
)
from ...artifact.store import record_from_jsonable, record_to_jsonable
from ..protocols import ArtifactBlobStore, ArtifactRecordStore, BlobInfo
from . import _io

_CHUNK = 64 * 1024
_DIGEST_HEX_LEN = 64


def _safe_component(value: str, kind: str) -> str:
    """tenant_id / artifact_id / digest become path components; reject anything
    that could escape the store root (empty, path separators, ``.``/``..``)."""
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _validate_digest(digest: str) -> None:
    # sha256 hex: exactly 64 hex chars. Defended here so a malformed digest can
    # never become a shard path or be mistaken for a valid content address.
    if len(digest) != _DIGEST_HEX_LEN:
        raise ValueError(f"invalid digest length: {digest!r}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"invalid digest (not hex): {digest!r}") from exc


class FilesystemArtifactBlobStore:
    """ArtifactBlobStore backed by the local filesystem. Blobs are
    content-addressed by sha256 under ``blobs_root/<xx>/<sha>``. ``put_if_absent``
    always consumes and verifies the source before deciding to publish a new
    blob or return an existing one; ``open`` streams the file back in bounded
    chunks."""

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
        self._metrics = metrics

    def _path(self, digest: str) -> Path:
        _safe_component(digest, "digest")
        _validate_digest(digest)
        return self._root / digest[:2] / digest

    def _fail(self, error: Exception, *, reason: str) -> None:
        if self._metrics is not None:
            self._metrics.counter(
                "artifact_blob_upload_failure_total", attributes={"reason": reason}
            )
        raise error

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        final = self._path(digest)
        await _io.async_makedirs(final.parent)
        fd, tmp_name = await _io.async_mkstemp(
            directory=final.parent, prefix=f".{final.name}.", suffix=".tmp"
        )
        hasher = hashlib.sha256()
        written = 0
        tmp_path = Path(tmp_name)
        try:
            # Always consume the source: spool to a same-dir temp, hashing as we
            # go, then verify the claimed digest and size before publishing.
            with os.fdopen(fd, "wb") as f:
                async for chunk in source:
                    if not chunk:
                        continue
                    hasher.update(chunk)
                    written += len(chunk)
                    await _io.async_write_chunk(f, chunk)
                await _io.async_fsync_file(f)
            actual = hasher.hexdigest()
            if actual != digest:
                self._fail(
                    ArtifactIntegrityError(
                        f"blob digest mismatch: claimed {digest[:12]}, actual {actual[:12]}"
                    ),
                    reason="digest_mismatch",
                )
            if size is not None and size != written:
                self._fail(
                    ArtifactIntegrityError(
                        f"blob size mismatch: claimed {size}, actual {written}"
                    ),
                    reason="size_mismatch",
                )
            # Source verified. Publish a new blob, or reconcile with an existing
            # one without overwriting it.
            if await _io.async_stat_size(final) is None:
                await _io.async_replace(tmp_name, final)
                await _io.async_fsync_directory(final.parent)
                return BlobInfo(digest=digest, size=written, content_type=None)
            return await self._verify_existing(final, digest, declared_size=size)
        finally:
            # Drop the temp if the publish path did not rename it away.
            if await _io.async_exists(tmp_path):
                await _io.async_unlink(tmp_path)

    async def _verify_existing(
        self, final: Path, digest: str, *, declared_size: "int | None"
    ) -> BlobInfo:
        existing_digest = await _io.async_hash_file(final, chunk_size=self._chunk)
        if existing_digest != digest:
            self._fail(
                ArtifactIntegrityError(
                    f"blob at sha256 {digest[:12]} is corrupt (actual {existing_digest[:12]}); "
                    f"refusing to record a reference to it"
                ),
                reason="corrupt",
            )
        existing_size = await _io.async_stat_size(final)
        if declared_size is not None and existing_size != declared_size:
            self._fail(
                ArtifactIntegrityError(
                    f"existing blob size mismatch: claimed {declared_size}, actual {existing_size}"
                ),
                reason="size_mismatch",
            )
        return BlobInfo(digest=digest, size=existing_size, content_type=None)

    @asynccontextmanager
    async def open(self, *, digest: str):
        path = self._path(digest)
        if await _io.async_stat_size(path) is None:
            raise ArtifactBlobNotFoundError(f"blob for sha256 {digest[:12]} missing")
        f = await _io.async_open_read(path)
        try:
            chunk = self._chunk

            async def _chunks() -> AsyncIterator[bytes]:
                while True:
                    block = await _io.async_read_chunk(f, chunk)
                    if not block:
                        break
                    yield block

            yield _chunks()
        finally:
            await _io.async_close(f)

    async def stat(self, *, digest: str) -> "BlobInfo | None":
        path = self._path(digest)
        size = await _io.async_stat_size(path)
        if size is None:
            return None
        return BlobInfo(digest=digest, size=size, content_type=None)

    async def delete(self, *, digest: str) -> None:
        path = self._path(digest)
        await _io.async_unlink(path)

    async def iter_digests_with_mtime(self) -> "AsyncIterator[tuple[str, datetime]]":
        """Yield ``(digest, modified_at)`` for every stored blob, for orphan
        sweeping. Walks the two-hex-char shard dirs; the digest is the file
        name (the sha256 it was filed under)."""
        if not await _io.async_exists(self._root):
            return
        for shard in await _io.async_list_subdirs(self._root):
            for blob in await _io.async_list_files(shard):
                yield blob.name, await _io.async_mtime(blob)


class FilesystemArtifactRecordStore:
    """ArtifactRecordStore backed by the local filesystem. Records are
    tenant-scoped JSON files under ``records_root/<tenant>/<id>.json``, created
    exclusively -- identical content is idempotent, a different value conflicts."""

    def __init__(self, *, records_root: Path) -> None:
        self._root = Path(records_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, tenant_id: str, artifact_id: str) -> Path:
        _safe_component(tenant_id, "tenant_id")
        _safe_component(artifact_id, "artifact_id")
        return self._root / tenant_id / f"{artifact_id}.json"

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        payload = json.dumps(record_to_jsonable(record)).encode("utf-8")
        path = self._path(record.tenant_id, record.ref.id)
        try:
            await _io.async_write_exclusive(path, payload)
        except FileExistsError:
            return await self._reconcile_existing(path, record, payload)
        return record

    async def _reconcile_existing(
        self, path: Path, record: ArtifactRecord, payload: bytes
    ) -> ArtifactRecord:
        # Exclusive create lost the race (or the id was reused). Load + validate
        # the stored record; identical content is idempotent, a different value
        # is a conflict (never a silent overwrite).
        existing = await self._load_record(
            path, expect_id=record.ref.id, expect_tenant=record.tenant_id
        )
        if json.dumps(record_to_jsonable(existing)).encode("utf-8") == payload:
            return existing
        raise ArtifactRecordConflictError(
            f"artifact {record.ref.id} already exists with different content"
        )

    async def _load_record(
        self,
        path: Path,
        *,
        expect_id: "str | None" = None,
        expect_tenant: "str | None" = None,
    ) -> ArtifactRecord:
        raw = await _io.async_read_bytes(path)
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ArtifactRecordCorruptError(
                f"record at {path} is not valid JSON: {exc}"
            ) from exc
        try:
            record = record_from_jsonable(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactRecordCorruptError(
                f"record at {path} is malformed: {exc}"
            ) from exc
        if expect_id is not None and record.ref.id != expect_id:
            raise ArtifactRecordCorruptError(
                f"record at {path} has id {record.ref.id!r}, expected {expect_id!r}"
            )
        if expect_tenant is not None and record.tenant_id != expect_tenant:
            raise ArtifactRecordCorruptError(
                f"record at {path} belongs to tenant {record.tenant_id!r}, "
                f"expected {expect_tenant!r}"
            )
        try:
            _validate_digest(record.ref.sha256)
        except ValueError as exc:
            raise ArtifactRecordCorruptError(
                f"record at {path} has a malformed sha256: {exc}"
            ) from exc
        return record

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        path = self._path(tenant_id, artifact_id)
        if await _io.async_stat_size(path) is None:
            return None
        return await self._load_record(
            path, expect_id=artifact_id, expect_tenant=tenant_id
        )

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        path = self._path(tenant_id, artifact_id)
        return await _io.async_unlink(path)

    async def iter_referenced_digests(self) -> AsyncIterator[str]:
        """Yield every sha256 referenced by some record, for orphan sweeping.
        A corrupt record aborts the scan (fail-closed) so the sweeper cannot
        delete a blob pinned by a record it failed to read."""
        async for record in self._iter_records():
            yield record.ref.sha256

    async def is_digest_referenced(self, digest: str) -> bool:
        """Whether any record pins ``digest`` (across tenants). Scans records
        fail-closed: a corrupt record aborts (raises) so the orphan sweeper
        cannot mistake a pinned blob for an orphan. Returns on the first
        matching record."""
        async for record in self._iter_records():
            if record.ref.sha256 == digest:
                return True
        return False

    async def _iter_records(
        self, *, tenant_id: "str | None" = None
    ) -> AsyncIterator[ArtifactRecord]:
        if not await _io.async_exists(self._root):
            return
        if tenant_id is not None:
            tenant_dirs = [self._root / tenant_id]
        else:
            tenant_dirs = await _io.async_list_subdirs(self._root)
        for tenant_dir in tenant_dirs:
            if not await _io.async_exists(tenant_dir):
                continue
            for record_file in await _io.async_list_files(tenant_dir):
                # The file name (minus .json) and parent dir ARE the id/tenant;
                # a mismatch means the record was moved or renamed out of band.
                record = await self._load_record(
                    record_file,
                    expect_id=record_file.stem,
                    expect_tenant=tenant_dir.name,
                )
                yield record

    async def iter_by_run_id(
        self, run_id: "str | None", *, tenant_id: "str | None" = None
    ) -> AsyncIterator[ArtifactRecord]:
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
