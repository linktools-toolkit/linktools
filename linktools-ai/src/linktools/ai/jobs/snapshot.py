#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset snapshotting: pin a live Asset into an immutable Artifact so task
retries and replay read the exact bytes that were used, not the latest version.

The snapshot is content-addressed through ArtifactStore (which itself reuses the
existing AssetStore), and the returned ResourceSnapshotRef records the
original path/version/etag plus the artifact id and sha256 so a later replay can
validate integrity before re-executing."""

from ..artifact.store import ArtifactStore
from ..asset.path import AssetPath
from ..asset.store import AssetStore
from .models import ResourceSnapshotRef


class ResourceSnapshotError(Exception):
    """Raised when a resource cannot be snapshotted (missing or unreadable)."""


async def snapshot_resource(
    resources: AssetStore,
    artifact_store: ArtifactStore,
    path: str,
    *,
    tenant_id: str,
    media_type: "str | None" = None,
) -> ResourceSnapshotRef:
    """Read the resource at ``path``, seal its bytes into an ArtifactStore, and
    return a :class:`ResourceSnapshotRef` pinning the path/version/etag plus the
    content-addressed artifact id and sha256.

    Uses a SINGLE ``resources.get`` so the version/etag and the sealed content
    come from one read -- a separate ``stat`` then ``get`` would be a TOCTOU
    window where the resource changes between the two calls and the snapshot
    pins stale metadata against new bytes."""
    rpath = AssetPath(path)
    resource = await resources.get(rpath)
    if resource is None:
        raise ResourceSnapshotError(f"resource not found: {path}")
    info = resource.info
    record = await artifact_store.put(
        resource.content,
        media_type=media_type or info.content_type or "application/octet-stream",
        tenant_id=tenant_id,
    )
    return ResourceSnapshotRef(
        path=path,
        version=info.version,
        etag=info.etag,
        artifact_id=record.ref.id,
        sha256=record.ref.sha256,
    )


__all__: "list[str]" = ["ResourceSnapshotError", "snapshot_resource"]
