"""Capability store composition and public read/write surface."""

from typing import Protocol

from ..storage.cache import ContentCache
from ..storage.revision import MetadataSnapshot, RevisionSource
from .entries import CapabilityEntry, CapabilityEntryChange, CapabilityEntryInfo


class CapabilityPort(Protocol):
    async def get(self, path: str) -> CapabilityEntry | None: ...
    async def stat(self, path: str) -> CapabilityEntryInfo | None: ...
    async def list_info(self, *, kind: str | None = None) -> tuple[CapabilityEntryInfo, ...]: ...
    async def current_revision(self) -> int: ...
    async def put(self, entry: CapabilityEntry) -> CapabilityEntry: ...
    async def delete(self, path: str) -> None: ...
    async def reset(self, entries: tuple[CapabilityEntry, ...]) -> None: ...
    async def list_changes(self, *, after_revision: int, through_revision: int) -> tuple[CapabilityEntryChange, ...]: ...


class CapabilityStore:
    """Compose one backend with optional content cache and revision source."""

    def __init__(
        self,
        backend: CapabilityPort,
        *,
        content_cache: ContentCache | None = None,
        revision_source: RevisionSource | None = None,
    ) -> None:
        self.backend = backend
        self.metadata = MetadataSnapshot(backend, revision_source=revision_source)
        self.content_cache = content_cache

    async def stat(self, path: str) -> CapabilityEntryInfo | None:
        return await self.metadata.get(path)

    async def list_active(self, kind: str | None = None) -> tuple[str, ...]:
        state = await self.metadata.refresh()
        if state is None:
            return ()
        return tuple(sorted(path for path, info in state.entries.items() if info.active and (kind is None or info.kind == kind)))

    async def get(self, path: str) -> CapabilityEntry | None:
        info = await self.metadata.get(path)
        if info is None:
            return await self.backend.get(path)
        key = (path, info.version, info.etag)
        if self.content_cache is not None:
            cached = await self.content_cache.get(key)
            if cached is not None:
                return CapabilityEntry(info, cached)
        entry = await self.backend.get(path)
        if entry is None or entry.info != info:
            return None
        if self.content_cache is not None:
            await self.content_cache.put(key, entry.content)
        return entry

    async def put(self, entry: CapabilityEntry) -> CapabilityEntry:
        return await self.backend.put(entry)

    async def delete(self, path: str) -> None:
        await self.backend.delete(path)

    async def reset(self, entries: tuple[CapabilityEntry, ...]) -> None:
        await self.backend.reset(entries)


__all__ = ["CapabilityPort", "CapabilityStore"]
