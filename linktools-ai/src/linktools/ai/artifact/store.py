"""Artifact backend contract and composed Store."""

from collections.abc import AsyncIterator
from typing import Protocol

from .models import ArtifactRecord


class ArtifactBackend(Protocol):
    async def put(self, *, record: ArtifactRecord, content: AsyncIterator[bytes]) -> ArtifactRecord: ...
    async def get_record(self, artifact_id: str) -> ArtifactRecord | None: ...
    async def open(self, artifact_id: str) -> AsyncIterator[bytes]: ...
    async def delete(self, artifact_id: str) -> None: ...


class ArtifactStore:
    def __init__(self, backend: ArtifactBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> ArtifactBackend:
        return self._backend

    async def initialize_storage(self, *args: object) -> None:
        await self._backend.initialize_storage(*args)

    async def put(self, *, record: ArtifactRecord, content: AsyncIterator[bytes]) -> ArtifactRecord:
        return await self._backend.put(record=record, content=content)

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return await self._backend.get_record(artifact_id)

    async def open(self, artifact_id: str):
        async for chunk in self._backend.open(artifact_id):
            yield chunk

    async def delete(self, artifact_id: str) -> None:
        await self._backend.delete(artifact_id)


__all__ = ["ArtifactBackend", "ArtifactStore"]
