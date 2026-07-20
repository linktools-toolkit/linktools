#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore: the artifact domain facade over content-addressed blobs and
per-write lineage records.

The facade depends on the stable :class:`ArtifactBlobStore` and
:class:`ArtifactRecordStore` Protocols -- NOT on :class:`AssetStore`. That
decouples the artifact domain from the asset domain entirely (this module
imports no asset symbol). The asset-backed reference implementation of the
Protocols lives in the storage infrastructure layer
(:mod:`linktools.ai.storage.artifact_backends`); an external object store or
DB can implement the same Protocols and be injected via the constructor.

Domain rules the facade owns: content deduplication by sha256 (identical bytes
share one blob), a fresh UUID :class:`ArtifactRecord` per put (each production
event keeps its own provenance), tenant scoping, and read-time integrity
verification. Every read is tenant-scoped and FAIL_CLOSED.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from ..storage.protocols import ArtifactBlobStore, ArtifactRecordStore
from .models import ArtifactIntegrityError, ArtifactRecord, ArtifactRef


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def record_to_jsonable(record: ArtifactRecord) -> dict:
    return {
        "ref": {
            "id": record.ref.id,
            "sha256": record.ref.sha256,
            "media_type": record.ref.media_type,
            "size": record.ref.size,
        },
        "tenant_id": record.tenant_id,
        "created_by_job_id": record.created_by_job_id,
        "created_by_task_id": record.created_by_task_id,
        "created_by_attempt_id": record.created_by_attempt_id,
        "parent_artifact_ids": list(record.parent_artifact_ids),
        "created_at": _coerce_utc(record.created_at).isoformat(),
        "metadata": dict(record.metadata),
    }


def record_from_jsonable(data: dict) -> ArtifactRecord:
    ref = ArtifactRef(
        id=data["ref"]["id"],
        sha256=data["ref"]["sha256"],
        media_type=data["ref"]["media_type"],
        size=data["ref"]["size"],
    )
    return ArtifactRecord(
        ref=ref,
        tenant_id=data["tenant_id"],
        created_by_job_id=data["created_by_job_id"],
        created_by_task_id=data["created_by_task_id"],
        created_by_attempt_id=data["created_by_attempt_id"],
        parent_artifact_ids=tuple(data["parent_artifact_ids"]),
        created_at=datetime.fromisoformat(data["created_at"]),
        metadata=dict(data["metadata"]),
    )


async def _bytes_to_async_iter(content: bytes) -> AsyncIterator[bytes]:
    yield content


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
        *,
        metrics: "Any | None" = None,
    ) -> None:
        self._blob = blob_store
        self._records = record_store
        # Optional ObservabilityMetrics sink. When wired, a digest mismatch on
        # read increments ``artifact_digest_mismatch_total`` and a put failure
        # (blob upload side) increments ``artifact_blob_upload_failure_total``.
        # Default None keeps existing callers no-op.
        self._metrics = metrics

    async def put(
        self,
        content: bytes,
        *,
        media_type: str,
        tenant_id: str,
        created_by_job_id: "str | None" = None,
        created_by_task_id: "str | None" = None,
        created_by_attempt_id: "str | None" = None,
        parent_artifact_ids: "tuple[str, ...]" = (),
        metadata: "dict[str, object] | None" = None,
        now: "datetime | None" = None,
    ) -> ArtifactRecord:
        sha = hashlib.sha256(content).hexdigest()
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
            created_by_job_id=created_by_job_id,
            created_by_task_id=created_by_task_id,
            created_by_attempt_id=created_by_attempt_id,
            parent_artifact_ids=parent_artifact_ids,
            created_at=_coerce_utc(now) if now is not None else _utcnow(),
            metadata=dict(metadata) if metadata else {},
        )
        return await self._records.put(record)

    async def stat(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        return await self._records.get(artifact_id, tenant_id=tenant_id)

    async def get(self, artifact_id: str, *, tenant_id: str) -> "bytes | None":
        # Tenant gate runs before content fetch so a foreign caller learns
        # nothing -- not even whether the artifact exists.
        record = await self.stat(artifact_id, tenant_id=tenant_id)
        if record is None:
            return None
        content = b"".join([chunk async for chunk in self._blob.open(digest=record.ref.sha256)])
        # Integrity: the stored blob must hash back to the sha256 the record
        # pinned -- catches tampering or a stale address.
        actual = hashlib.sha256(content).hexdigest()
        if actual != record.ref.sha256:
            if self._metrics is not None:
                self._metrics.counter("artifact_digest_mismatch_total")
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} blob sha256 mismatch: {actual}"
            )
        return content


__all__: "list[str]" = ["ArtifactStore"]
