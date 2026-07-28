"""Reusable revision-gated metadata snapshots for storage backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Generic, Mapping, Protocol, TypeVar


EntryT = TypeVar("EntryT")
ChangeT = TypeVar("ChangeT")


class SnapshotRequired(Exception):
    """The repository cannot provide a complete revision delta."""


class RevisionSource(Protocol):
    async def revision(self) -> int: ...


class VersionedMetadataRepository(Protocol[EntryT, ChangeT]):
    async def current_revision(self) -> int: ...
    async def list_info(self) -> tuple[EntryT, ...]: ...
    async def list_changes(self, *, after_revision: int, through_revision: int) -> tuple[ChangeT, ...]: ...


@dataclass(frozen=True, slots=True)
class MetadataState(Generic[EntryT]):
    revision: int
    entries: Mapping[str, EntryT]


class MetadataSnapshot(Generic[EntryT, ChangeT]):
    """A stable, revision-gated view shared by storage consumers."""

    def __init__(
        self,
        repository: VersionedMetadataRepository[EntryT, ChangeT],
        *,
        revision_source: RevisionSource | None = None,
        entry_key: Callable[[EntryT], str] = lambda entry: entry.path,
        change_key: Callable[[ChangeT], str] = lambda change: change.path,
        change_value: Callable[[ChangeT], EntryT | None] = lambda change: change.info,
    ) -> None:
        self.repository = repository
        self.revision_source = revision_source
        self.entry_key = entry_key
        self.change_key = change_key
        self.change_value = change_value
        self._state: MetadataState[EntryT] | None = None
        self._refresh_lock = asyncio.Lock()

    async def _revision(self) -> int:
        if self.revision_source is not None:
            return await self.revision_source.revision()
        return await self.repository.current_revision()

    async def _full(self, revision: int) -> MetadataState[EntryT]:
        values = await self.repository.list_info()
        return MetadataState(revision, {self.entry_key(value): value for value in values})

    async def refresh(self) -> MetadataState[EntryT] | None:
        target = await self._revision()
        if self._state is not None and self._state.revision == target:
            return self._state
        async with self._refresh_lock:
            for _attempt in range(3):
                target = await self._revision()
                if self._state is not None and self._state.revision == target:
                    return self._state
                current = self._state
                try:
                    if current is None:
                        candidate = await self._full(target)
                    else:
                        changes = await self.repository.list_changes(
                            after_revision=current.revision,
                            through_revision=target,
                        )
                        if len(changes) > max(1, int(len(current.entries) * 0.25)):
                            candidate = await self._full(target)
                        else:
                            entries = dict(current.entries)
                            for change in changes:
                                key = self.change_key(change)
                                value = self.change_value(change)
                                if value is None:
                                    entries.pop(key, None)
                                else:
                                    entries[key] = value
                            candidate = MetadataState(target, entries)
                except SnapshotRequired:
                    candidate = await self._full(target)
                except Exception:
                    raise
                if await self._revision() == target:
                    self._state = candidate
                    return candidate
            return None

    async def get(self, key: str) -> EntryT | None:
        state = await self.refresh()
        return None if state is None else state.entries.get(key)


__all__ = ["MetadataSnapshot", "MetadataState", "RevisionSource", "SnapshotRequired", "VersionedMetadataRepository"]
