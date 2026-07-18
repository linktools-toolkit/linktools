#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore: immutable artifacts over ResourceStore, with content blobs
separated from per-write lineage records.

The blob store, versioning, conditional writes and idempotency already live in
:class:`ResourceStore`; this layer adds the artifact domain rules -- content
deduplication by sha256 (identical bytes share one blob via ``if_none_match``),
a fresh UUID :class:`ArtifactRecord` per put (so each production event keeps its
own provenance, even for identical content), tenant scoping, and read-time
integrity verification. No new file or database backend.

Every read is tenant-scoped and FAIL_CLOSED: a caller learns nothing about an
artifact owned by another tenant (not its metadata, not its bytes) -- lineage
in one record cannot be followed into a foreign artifact.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone

from ..errors import ResourcePreconditionFailedError
from ..storage.resource.models import WriteOptions
from ..storage.resource.path import ResourcePath
from ..storage.resource.store import ResourceStore
from .models import ArtifactIntegrityError, ArtifactRecord, ArtifactRef


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(moment: datetime) -> datetime:
    # A naive datetime has no defined zone; assume the caller meant UTC rather
    # than silently persisting a timestamp that breaks tz-aware comparisons.
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _record_to_jsonable(record: ArtifactRecord) -> dict:
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


def _record_from_jsonable(data: dict) -> ArtifactRecord:
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


class ArtifactStore:
    """Content-addressed blobs with per-write lineage records.

    The CONTENT is deduplicated by sha256 (identical bytes share one blob). The
    RECORD is minted fresh on every put (a UUID id), so each production event --
    even of identical content -- keeps its own provenance (created_by_*,
    parent_artifact_ids). Reads are tenant-scoped and integrity-verified: a
    caller learns nothing about another tenant's artifact, and a tampered blob
    is detected on read."""

    def __init__(self, resources: ResourceStore, *, root: str = "artifacts") -> None:
        self._resources = resources
        base = ResourcePath(f"/{root}")
        self._blobs_root = base.child("blobs").child("sha256")
        self._records_root = base.child("records")

    def _blob_path(self, sha: str) -> ResourcePath:
        return self._blobs_root.child(sha[:2]).child(sha)

    def _record_path(self, tenant_id: str, artifact_id: str) -> ResourcePath:
        # Records are tenant-scoped: identical content from different tenants
        # shares the content blob but each tenant owns a private record, so a
        # foreign caller cannot find (or be locked out of) another tenant's
        # artifact by guessing a content hash.
        return self._records_root.child(tenant_id).child(f"{artifact_id}.json")

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
        # Content is deduplicated by sha256: the same bytes land at one blob
        # path. if_none_match fails if the blob is already present -- treat that
        # as reuse, but VERIFY the stored blob actually hashes to ``sha``: a
        # mismatch means the blob at this path is corrupt/tampered, and creating
        # a new ArtifactRecord pointing at it would propagate corruption.
        try:
            await self._resources.put(
                self._blob_path(sha),
                content,
                options=WriteOptions(if_none_match=True, content_type=media_type),
            )
        except ResourcePreconditionFailedError:
            existing = await self._resources.get(self._blob_path(sha))
            if existing is None:
                raise ArtifactIntegrityError(
                    f"blob for sha256 {sha[:12]} vanished during put"
                )
            actual = hashlib.sha256(existing.content).hexdigest()
            if actual != sha:
                raise ArtifactIntegrityError(
                    f"blob at sha256 {sha[:12]} is corrupt (actual {actual[:12]}); "
                    f"refusing to record a reference to it"
                )

        # A fresh record PER put: distinct id even for identical content, so the
        # second producer keeps its own lineage (the bug was that identical
        # content reused one record and lost the second job/task's provenance).
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
        payload = json.dumps(_record_to_jsonable(record)).encode("utf-8")
        await self._resources.put(
            self._record_path(tenant_id, artifact_id),
            payload,
            options=WriteOptions(
                if_none_match=True, content_type="application/json"
            ),
        )
        stored = await self._resources.get(
            self._record_path(tenant_id, artifact_id)
        )
        if stored is None:  # pragma: no cover - invariant: records are never deleted
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} record missing after write"
            )
        return _record_from_jsonable(json.loads(stored.content.decode("utf-8")))

    async def stat(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        resource = await self._resources.get(
            self._record_path(tenant_id, artifact_id)
        )
        if resource is None:
            return None
        record = _record_from_jsonable(json.loads(resource.content.decode("utf-8")))
        # The record path is already tenant-scoped; this check is defense in
        # depth (a record should never carry a foreign tenant_id).
        if record.tenant_id != tenant_id:
            return None
        return record

    async def get(self, artifact_id: str, *, tenant_id: str) -> "bytes | None":
        # Tenant gate runs before content fetch so a foreign caller learns
        # nothing -- not even whether the artifact exists.
        record = await self.stat(artifact_id, tenant_id=tenant_id)
        if record is None:
            return None
        resource = await self._resources.get(self._blob_path(record.ref.sha256))
        if resource is None:
            return None
        # Integrity: the stored blob must hash back to the sha256 the record
        # pinned -- catches tampering or a stale address.
        actual = hashlib.sha256(resource.content).hexdigest()
        if actual != record.ref.sha256:
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} blob sha256 mismatch: {actual}"
            )
        return resource.content


__all__: "list[str]" = ["ArtifactStore"]
