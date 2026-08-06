#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Artifact list, authorization and download actions."""

from ...domain.artifact import Artifact
from ...foundation.errors import ErrorCode, LinktoolsAIError


class ListArtifacts:
    def __init__(self, repository: object) -> None: self._repository = repository
    async def execute(self, execution_id: str, cursor: "str | None" = None, limit: int = 100) -> object: return await self._repository.list(execution_id, cursor, min(limit, 200))


class GetArtifact:
    def __init__(self, repository: object, delivery: object, tenant_id: str = "authenticated") -> None:
        self._repository, self._delivery, self._tenant_id = repository, delivery, tenant_id

    async def execute(self, artifact_id: str, expires_at: object) -> object:
        artifact = await self._repository.get(artifact_id)
        if not isinstance(artifact, Artifact) or not artifact.can_download(self._tenant_id):
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED, "artifact is not accessible")
        return await self._delivery.prepare_download(artifact_id, expires_at)


__all__ = ["GetArtifact", "ListArtifacts"]
