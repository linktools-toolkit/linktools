#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-backed reference implementations of the artifact Protocols.

The artifact domain facade (:mod:`linktools.ai.artifact.store`) depends only
on the :class:`ArtifactBlobStore` / :class:`ArtifactRecordStore` Protocols.
These adapters are the in-repo "private content-addressed blob primitive" --
they bridge the asset store (one possible backend) to those Protocols. They
live in the storage infrastructure layer (not in the artifact domain) so the
artifact package never imports the asset package: an external object store or
DB can implement the same Protocols and be injected without the asset store.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import AsyncIterator

from ..asset.models import Depth, WriteOptions
from ..asset.path import AssetPath
from ..asset.store import AssetStore
from ..artifact.models import ArtifactIntegrityError, ArtifactRecord
from ..artifact.store import record_from_jsonable, record_to_jsonable
from ..errors import ResourcePreconditionFailedError
from .protocols import ArtifactBlobStore, ArtifactRecordStore, BlobInfo


def build_artifact_store_from_assets(
    assets: AssetStore, *, root: str = "artifacts"
):
    """The asset-backed reference :class:`ArtifactStore` factory.

    This is the ONLY place the asset domain is wired into the artifact
    domain: an AssetStore backs the :class:`ArtifactBlobStore` /
    :class:`ArtifactRecordStore` Protocols, then the Protocol-only
    :class:`linktools.ai.artifact.store.ArtifactStore` facade is constructed
    over them. The factory lives in the storage infrastructure layer (not in
    the artifact domain) so the artifact package never names AssetStore; an
    external object store or DB can implement the same Protocols and be
    passed to the facade constructor directly, with no asset store involved.
    """
    from ..artifact.store import ArtifactStore

    base = AssetPath(f"/{root}")
    return ArtifactStore(
        AssetBackedArtifactBlobStore(
            assets, blobs_root=base.child("blobs").child("sha256")
        ),
        AssetBackedArtifactRecordStore(assets, records_root=base.child("records")),
    )


class AssetBackedArtifactBlobStore:
    """ArtifactBlobStore backed by an AssetStore. Blobs are content-addressed
    by sha256 under ``{blobs_root}/<xx>/<sha>``; ``put_if_absent`` is idempotent
    via AssetStore CAS (``if_none_match``)."""

    def __init__(
        self,
        assets: AssetStore,
        *,
        blobs_root: AssetPath,
        metrics: "Any | None" = None,
    ) -> None:
        self._assets = assets
        self._blobs_root = blobs_root
        # Optional ObservabilityMetrics sink. When wired, a digest-mismatch or
        # corruption failure on put_if_absent increments
        # ``artifact_blob_upload_failure_total``. Default None = no-op so
        # existing callers (build_artifact_store_from_assets) keep their
        # no-metrics behavior.
        self._metrics = metrics

    def _path(self, sha: str) -> AssetPath:
        return self._blobs_root.child(sha[:2]).child(sha)

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        content = b"".join([chunk async for chunk in source])
        actual = hashlib.sha256(content).hexdigest()
        if actual != digest:
            if self._metrics is not None:
                self._metrics.counter(
                    "artifact_blob_upload_failure_total",
                    attributes={"reason": "digest_mismatch"},
                )
            raise ArtifactIntegrityError(
                f"blob digest mismatch: claimed {digest[:12]}, actual {actual[:12]}"
            )
        try:
            await self._assets.put(
                self._path(digest),
                content,
                options=WriteOptions(if_none_match=True),
            )
        except ResourcePreconditionFailedError:
            existing = await self._assets.get(self._path(digest))
            if existing is None:
                if self._metrics is not None:
                    self._metrics.counter(
                        "artifact_blob_upload_failure_total",
                        attributes={"reason": "vanished"},
                    )
                raise ArtifactIntegrityError(
                    f"blob for sha256 {digest[:12]} vanished during put"
                )
            found = hashlib.sha256(existing.content).hexdigest()
            if found != digest:
                if self._metrics is not None:
                    self._metrics.counter(
                        "artifact_blob_upload_failure_total",
                        attributes={"reason": "corrupt"},
                    )
                raise ArtifactIntegrityError(
                    f"blob at sha256 {digest[:12]} is corrupt (actual {found[:12]}); "
                    f"refusing to record a reference to it"
                )
        return BlobInfo(digest=digest, size=len(content), content_type=None)

    async def open(self, *, digest: str) -> AsyncIterator[bytes]:
        existing = await self._assets.get(self._path(digest))
        if existing is None:
            raise ArtifactIntegrityError(f"blob for sha256 {digest[:12]} missing")
        yield existing.content

    async def stat(self, *, digest: str) -> "BlobInfo | None":
        info = await self._assets.stat(self._path(digest))
        if info is None:
            return None
        return BlobInfo(digest=digest, size=info.size, content_type=info.content_type)

    async def delete(self, *, digest: str) -> None:
        await self._assets.delete(self._path(digest))

    async def iter_digests_with_mtime(self) -> AsyncIterator:
        """Yield ``(digest, modified_at)`` for every stored blob, for orphan
        sweeping. The digest is the blob's path leaf (the sha256 it was filed
        under); ``modified_at`` is the blob's write time, the timestamp the
        grace window is measured against."""
        cursor: "str | None" = None
        while True:
            page = await self._assets.propfind(
                self._blobs_root, depth=Depth.INFINITY, limit=200, cursor=cursor
            )
            for info in page.items:
                # Blob path leaf IS the digest (blobs_root/<xx>/<sha256>).
                yield info.path.parts[-1], info.modified_at
            if page.cursor is None:
                return
            cursor = page.cursor


class AssetBackedArtifactRecordStore:
    """ArtifactRecordStore backed by an AssetStore. Records are tenant-scoped
    JSON files under ``{records_root}/<tenant>/<id>.json``; ``if_none_match``
    enforces the fresh-id-per-write invariant."""

    def __init__(self, assets: AssetStore, *, records_root: AssetPath) -> None:
        self._assets = assets
        self._records_root = records_root

    def _path(self, tenant_id: str, artifact_id: str) -> AssetPath:
        return self._records_root.child(tenant_id).child(f"{artifact_id}.json")

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        payload = json.dumps(record_to_jsonable(record)).encode("utf-8")
        await self._assets.put(
            self._path(record.tenant_id, record.ref.id),
            payload,
            options=WriteOptions(if_none_match=True, content_type="application/json"),
        )
        return record

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        resource = await self._assets.get(self._path(tenant_id, artifact_id))
        if resource is None:
            return None
        record = record_from_jsonable(json.loads(resource.content.decode("utf-8")))
        if record.tenant_id != tenant_id:
            return None
        return record

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        existed = await self.get(artifact_id, tenant_id=tenant_id)
        if existed is None:
            return False
        await self._assets.delete(self._path(tenant_id, artifact_id))
        return True

    async def iter_referenced_digests(self) -> AsyncIterator[str]:
        """Yield every sha256 referenced by some record, for orphan sweeping.
        Walks the records tree, reads each record JSON, and emits its pinned
        digest -- the set of blobs that are NOT orphans."""
        cursor: "str | None" = None
        while True:
            page = await self._assets.propfind(
                self._records_root, depth=Depth.INFINITY, limit=200, cursor=cursor
            )
            for info in page.items:
                if not info.path.parts[-1].endswith(".json"):
                    continue
                resource = await self._assets.get(info.path)
                if resource is None:
                    continue
                try:
                    # Parse via the public record codec so the sweeper does not
                    # hand-couple to the JSON shape; a malformed record is not a
                    # blob reference and is skipped, not fatal to the sweep.
                    record = record_from_jsonable(
                        json.loads(resource.content.decode("utf-8"))
                    )
                except (ValueError, KeyError, TypeError):
                    continue
                yield record.ref.sha256
            if page.cursor is None:
                return
            cursor = page.cursor


__all__: "list[str]" = ["AssetBackedArtifactBlobStore", "AssetBackedArtifactRecordStore"]
