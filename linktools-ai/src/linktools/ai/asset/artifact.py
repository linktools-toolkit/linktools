#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactService: a domain service over AssetStore for run-scoped artifacts.
Owns naming/run-association/content-type/metadata; AssetStore continues to own
content, ETag, version, and idempotency underneath it."""

from typing import Mapping

from .models import Depth, Asset, AssetInfo, WriteOptions
from .path import AssetPath
from .store import AssetStore


class ArtifactService:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def _path(self, *, tenant_id: str, run_id: str, artifact_name: str) -> AssetPath:
        return AssetPath(f"/artifacts/{tenant_id}/{run_id}/{artifact_name}")

    async def put(
        self,
        *,
        tenant_id: str,
        run_id: str,
        artifact_name: str,
        content: bytes,
        content_type: "str | None" = None,
        metadata: "Mapping[str, object] | None" = None,
    ) -> Asset:
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
    ) -> "Asset | None":
        path = self._path(
            tenant_id=tenant_id, run_id=run_id, artifact_name=artifact_name
        )
        return await self._assets.get(path)

    async def list_for_run(
        self, *, tenant_id: str, run_id: str
    ) -> "tuple[AssetInfo, ...]":
        prefix = AssetPath(f"/artifacts/{tenant_id}/{run_id}")
        page = await self._assets.propfind(
            prefix, depth=Depth.ONE, limit=1000, cursor=None
        )
        return page.items
