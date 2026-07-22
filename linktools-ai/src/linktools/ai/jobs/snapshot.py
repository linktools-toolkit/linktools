#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset snapshotting: pin a live Asset into an immutable Artifact so task
retries and replay read the exact bytes that were used, not the latest version.

The snapshot is content-addressed through ArtifactStore (which itself reuses the
existing AssetStore), and the returned AssetSnapshotRef records the
original path/version/etag plus the artifact id and sha256 so a later replay can
validate integrity before re-executing."""

from ..artifact.models import ArtifactProvenance
from ..artifact.store import ArtifactStore
from ..asset.path import AssetPath
from ..asset.store import AssetStore
from .models import AssetSnapshotRef


class AssetSnapshotError(Exception):
    """Raised when a asset cannot be snapshotted (missing or unreadable)."""


async def snapshot_asset(
    assets: AssetStore,
    artifact_store: ArtifactStore,
    path: str,
    *,
    tenant_id: str,
    run_id: "str | None" = None,
    media_type: "str | None" = None,
) -> AssetSnapshotRef:
    """Read the asset at ``path``, seal its bytes into an ArtifactStore, and
    return a :class:`AssetSnapshotRef` pinning the path/version/etag plus the
    content-addressed artifact id and sha256.

    Uses a SINGLE ``assets.get`` so the version/etag and the sealed content
    come from one read -- a separate ``stat`` then ``get`` would be a TOCTOU
    window where the asset changes between the two calls and the snapshot
    pins stale metadata against new bytes."""
    rpath = AssetPath(path)
    asset = await assets.get(rpath)
    if asset is None:
        raise AssetSnapshotError(f"asset not found: {path}")
    info = asset.info
    record = await artifact_store.put(
        content=asset.content,
        media_type=media_type or info.content_type or "application/octet-stream",
        tenant_id=tenant_id,
        provenance=(
            ArtifactProvenance(
                producer_kind="run_snapshot", producer_id=run_id, run_id=run_id
            )
            if run_id is not None
            # No run context: an anonymous snapshot, consistent with the store's
            # default for unattributed artifacts (not "run_snapshot" with an
            # empty producer_id).
            else ArtifactProvenance(producer_kind="anonymous", producer_id="")
        ),
    )
    return AssetSnapshotRef(
        path=path,
        version=info.version,
        etag=info.etag,
        artifact_id=record.ref.id,
        sha256=record.ref.sha256,
    )


__all__: "list[str]" = ["AssetSnapshotError", "snapshot_asset"]
