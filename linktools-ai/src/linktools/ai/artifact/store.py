"""Artifact store contract and business facade."""

from collections.abc import AsyncIterator
from typing import Protocol

from .models import ArtifactRecord


class ArtifactPort(Protocol):
    async def put(self, *, record: ArtifactRecord, content: AsyncIterator[bytes]) -> ArtifactRecord: ...
    async def get_record(self, artifact_id: str) -> ArtifactRecord | None: ...
    async def open(self, artifact_id: str) -> AsyncIterator[bytes]: ...
    async def delete(self, artifact_id: str) -> None: ...


class ArtifactStore:
    def __init__(self, backend: ArtifactPort) -> None:
        self.backend = backend

    async def put(self, *, record: ArtifactRecord, content: AsyncIterator[bytes]) -> ArtifactRecord:
        return await self.backend.put(record=record, content=content)

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return await self.backend.get_record(artifact_id)

    async def open(self, artifact_id: str) -> AsyncIterator[bytes]:
        async for chunk in self.backend.open(artifact_id):
            yield chunk

    async def delete(self, artifact_id: str) -> None:
        await self.backend.delete(artifact_id)


class ArtifactService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def put(self, *, record: ArtifactRecord, content: AsyncIterator[bytes]) -> ArtifactRecord:
        return await self._store.put(record=record, content=content)

    async def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return await self._store.get_record(artifact_id)

    async def open(self, artifact_id: str):
        async for chunk in self._store.open(artifact_id):
            yield chunk

    async def delete(self, artifact_id: str) -> None:
        await self._store.delete(artifact_id)


__all__ = ["ArtifactPort", "ArtifactStore", "ArtifactService"]
