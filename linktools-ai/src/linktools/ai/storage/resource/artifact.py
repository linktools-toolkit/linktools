#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactService: a domain service over ResourceStore for run-scoped artifacts.
Owns naming/run-association/content-type/metadata; ResourceStore continues to own
content, ETag, version, and idempotency underneath it."""

from typing import Mapping

from .models import Depth, Resource, ResourceInfo, WriteOptions
from .path import ResourcePath
from .store import ResourceStore


class ArtifactService:
    def __init__(self, *, resources: ResourceStore) -> None:
        self._resources = resources

    def _path(self, *, tenant_id: str, run_id: str, artifact_name: str) -> ResourcePath:
        return ResourcePath(f"/artifacts/{tenant_id}/{run_id}/{artifact_name}")

    async def put(
        self,
        *,
        tenant_id: str,
        run_id: str,
        artifact_name: str,
        content: bytes,
        content_type: "str | None" = None,
        metadata: "Mapping[str, object] | None" = None,
    ) -> Resource:
        path = self._path(tenant_id=tenant_id, run_id=run_id, artifact_name=artifact_name)
        return await self._resources.put(
            path, content, options=WriteOptions(content_type=content_type, metadata=metadata or {})
        )

    async def get(self, *, tenant_id: str, run_id: str, artifact_name: str) -> "Resource | None":
        path = self._path(tenant_id=tenant_id, run_id=run_id, artifact_name=artifact_name)
        return await self._resources.get(path)

    async def list_for_run(self, *, tenant_id: str, run_id: str) -> "tuple[ResourceInfo, ...]":
        prefix = ResourcePath(f"/artifacts/{tenant_id}/{run_id}")
        page = await self._resources.propfind(prefix, depth=Depth.ONE, limit=1000, cursor=None)
        return page.items
