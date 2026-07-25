#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemArtifactRecordStore: the filesystem reference implementation of
the :class:`~linktools.ai.artifact.persistence.protocols.ArtifactRecordStore`
Protocol. Tenant-scoped lineage records are JSON files under
``records_root/<tenant>/<id>.json``, created exclusively -- identical content
is idempotent, a different value conflicts.

Lives in the artifact domain (the storage kernel's only consumers are the
Protocol itself + the shared filesystem async-I/O helpers it composes over)."""

import json
from typing import AsyncIterator
from pathlib import Path

from ..digest import ArtifactDigest
from ..models import (
    ArtifactRecord,
    ArtifactRecordConflictError,
    ArtifactRecordCorruptError,
)
from ..store import record_from_jsonable, record_to_jsonable
from ...storage.filesystem._util import _atomic_write, _validate_id_segment
from ...storage.filesystem import _io

_DIGEST_HEX_LEN = 64


def _validate_digest(digest: str) -> None:
    if len(digest) != _DIGEST_HEX_LEN:
        raise ValueError(f"invalid digest length: {digest!r}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"invalid digest (not hex): {digest!r}") from exc


def _safe_component(value: str, kind: str) -> str:
    """tenant_id / artifact_id become path components; reject anything that
    could escape the store root (empty, path separators, ``.``/``..``)."""
    if not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FilesystemArtifactRecordStore:
    """ArtifactRecordStore backed by the local filesystem. Records are
    tenant-scoped JSON files under ``records_root/<tenant>/<id>.json``, created
    exclusively -- identical content is idempotent, a different value conflicts.

    Lives in the storage layer (the artifact domain depends only on the
    ``ArtifactRecordStore`` Protocol defined in ``artifact.persistence``); the
    composition root injects this concrete backend into the ArtifactStore
    facade."""

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

    async def is_digest_referenced(self, digest: ArtifactDigest) -> bool:
        """Whether any record pins ``digest`` (across tenants). Scans records
        fail-closed: a corrupt record aborts (raises) so the orphan sweeper
        cannot mistake a pinned blob for an orphan. Returns on the first
        matching record."""
        async for record in self._iter_records():
            if record.ref.sha256 == digest.value:
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




__all__: "list[str]" = ["FilesystemArtifactRecordStore"]
