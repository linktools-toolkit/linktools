#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactStore: immutable, content-addressed artifacts over ResourceStore.

The blob store, versioning, conditional writes and idempotency already live in
:class:`ResourceStore`; this layer adds the artifact domain rules -- content
addressing (id == sha256), immutability (``if_none_match`` writes that reuse
identical content instead of duplicating it), parent lineage, tenant scoping
and read-time integrity verification. No new file or database backend.

Every read is tenant-scoped and FAIL_CLOSED: a caller learns nothing about an
artifact owned by another tenant (not its metadata, not its bytes) -- lineage
in one record cannot be followed into a foreign artifact.
"""

import hashlib
import json
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
    def __init__(self, resources: ResourceStore, *, root: str = "artifacts") -> None:
        self._resources = resources
        base = ResourcePath(f"/{root}")
        self._content_root = base.child("sha256")
        self._records_root = base.child("records")

    def _content_path(self, sha: str) -> ResourcePath:
        return self._content_root.child(sha[:2]).child(sha)

    def _record_path(self, tenant_id: str, artifact_id: str) -> ResourcePath:
        # Records are tenant-scoped: identical content from different tenants
        # shares the content blob (content-addressed) but each tenant owns a
        # private record, so neither is locked out of their own write. Provenance is per-tenant; content is deduplicated.
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
        # Content is content-addressed: the same bytes land at the same path.
        # if_none_match makes the write fail if identical content is already
        # present, which we treat as reuse rather than error.
        try:
            await self._resources.put(
                self._content_path(sha),
                content,
                options=WriteOptions(if_none_match=True, content_type=media_type),
            )
        except ResourcePreconditionFailedError:
            pass

        record = ArtifactRecord(
            ref=ArtifactRef(
                id=sha, sha256=sha, media_type=media_type, size=len(content)
            ),
            tenant_id=tenant_id,
            created_by_job_id=created_by_job_id,
            created_by_task_id=created_by_task_id,
            created_by_attempt_id=created_by_attempt_id,
            parent_artifact_ids=parent_artifact_ids,
            created_at=_coerce_utc(now) if now is not None else _utcnow(),
            metadata=dict(metadata) if metadata else {},
        )
        # Record is first-wins: a second put of identical content keeps the
        # original provenance (callers needing distinct provenance must produce
        # distinct content).
        payload = json.dumps(_record_to_jsonable(record)).encode("utf-8")
        try:
            await self._resources.put(
                self._record_path(tenant_id, sha),
                payload,
                options=WriteOptions(
                    if_none_match=True, content_type="application/json"
                ),
            )
        except ResourcePreconditionFailedError:
            pass
        stored = await self._resources.get(self._record_path(tenant_id, sha))
        if stored is None:  # pragma: no cover - invariant: records are never deleted
            raise ArtifactIntegrityError(f"artifact {sha} record missing after write")
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
        if await self.stat(artifact_id, tenant_id=tenant_id) is None:
            return None
        resource = await self._resources.get(self._content_path(artifact_id))
        if resource is None:
            return None
        # Integrity: the stored bytes must hash back to the id we addressed.
        actual = hashlib.sha256(resource.content).hexdigest()
        if actual != artifact_id:
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} content sha256 mismatch: {actual}"
            )
        return resource.content


__all__: "list[str]" = ["ArtifactStore"]
