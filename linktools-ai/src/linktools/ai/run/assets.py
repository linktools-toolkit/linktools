#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactService: a domain service over ObjectStore for run-scoped artifacts.
Owns naming/run-association/content-type/metadata; ObjectStore continues to own
content, ETag, version, and idempotency underneath it."""

from typing import Mapping

from ..storage.object.models import Depth, ObjectInfo, StorageKey, StoredObject, WriteOptions
from ..storage.object.store import ObjectStore


class ArtifactService:
    def __init__(self, *, assets: ObjectStore) -> None:
        self._assets = assets

    def _path(self, *, tenant_id: str, run_id: str, artifact_name: str) -> StorageKey:
        return StorageKey(f"/artifacts/{tenant_id}/{run_id}/{artifact_name}")

    async def put(
        self,
        *,
        tenant_id: str,
        run_id: str,
        artifact_name: str,
        content: bytes,
        content_type: "str | None" = None,
        metadata: "Mapping[str, object] | None" = None,
    ) -> StoredObject:
        path = self._path(
            tenant_id=tenant_id, run_id=run_id, artifact_name=artifact_name
        )
        return await self._assets.put(
            path,
            content,
            options=WriteOptions(content_type=content_type, metadata=metadata or {}),
        )

    async def get(
        self, *, tenant_id: str, run_id: str, artifact_name: str
    ) -> "StoredObject | None":
        path = self._path(
            tenant_id=tenant_id, run_id=run_id, artifact_name=artifact_name
        )
        return await self._assets.get(path)

    async def list_for_run(
        self, *, tenant_id: str, run_id: str
    ) -> "tuple[ObjectInfo, ...]":
        prefix = StorageKey(f"/artifacts/{tenant_id}/{run_id}")
        page = await self._assets.list(prefix, depth=Depth.ONE, limit=1000)
        return page.items
