#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-flight metadata refresh for ordered storage layers."""

import asyncio
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from linktools.core import environ

from ._contracts import (
    InfoT,
    InitializableStorage,
    KeyT,
    MetadataLoad,
    MetadataLoadMode,
    ReadableStorageBackend,
    StorageChange,
    StorageMetadataReader,
    StorageRevision,
)
from ._layer import LayerRefreshPolicy

ValueT = TypeVar("ValueT")
_logger = environ.get_logger("ai.storage.revision")


class RevisionSource(Protocol):
    async def head_revision(self) -> 'StorageRevision | None': ...
    async def revision_bumped(self, revision: StorageRevision) -> None: ...


class StorageRevisionSource(Generic[KeyT, ValueT, InfoT]):
    """Live revision source for a metadata backend."""

    def __init__(self, backend: 'ReadableStorageBackend[KeyT, ValueT, InfoT]') -> None:
        self._backend = backend

    async def head_revision(self) -> 'StorageRevision | None':
        return await self._backend.head_revision()

    async def revision_bumped(self, revision: StorageRevision) -> None:
        return None


@dataclass(frozen=True, slots=True)
class MetadataState(Generic[KeyT, InfoT]):
    revision: StorageRevision
    entries: "dict[KeyT, InfoT]"


def apply_metadata_load(
    current: 'MetadataState[KeyT, InfoT] | None',
    load: 'MetadataLoad[KeyT, InfoT]',
) -> 'MetadataState[KeyT, InfoT]':
    entries = {} if load.mode is MetadataLoadMode.REPLACE or current is None else dict(current.entries)
    for change in load.changes:
        if change.info is None:
            entries.pop(change.key, None)
        else:
            entries[change.key] = change.info
    return MetadataState(load.store_revision, entries)


class LayerMetadataView(Generic[KeyT, ValueT, InfoT]):
    """Materialize one backend's metadata without publishing partial state."""

    def __init__(
        self,
        backend: 'ReadableStorageBackend[KeyT, ValueT, InfoT]',
        policy: LayerRefreshPolicy,
        *,
        revision_source: 'RevisionSource | None' = None,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.revision_source = revision_source
        self._state: MetadataState[KeyT, InfoT] | None = None
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[MetadataState[KeyT, InfoT]] | None = None
        self._generation = 0

    async def initialize(self) -> None:
        if isinstance(self.backend, InitializableStorage):
            await self.backend.initialize()

    async def refresh(self) -> 'MetadataState[KeyT, InfoT]':
        if self.policy is LayerRefreshPolicy.STATIC and self._state is not None:
            return self._state
        if self.policy is not LayerRefreshPolicy.ALWAYS and self._state is not None:
            source = self.revision_source
            current = (
                await source.head_revision()
                if source is not None
                else await self.backend.head_revision()
            )
            if current == self._state.revision:
                return self._state
            if source is not None and current is not None:
                head = await self.backend.head_revision()
                if head == self._state.revision:
                    try:
                        await source.revision_bumped(head)
                    except Exception:
                        _logger.warning(
                            "storage revision source correction failed",
                            exc_info=True,
                        )
                    return self._state
        async with self._lock:
            if self.policy is LayerRefreshPolicy.STATIC and self._state is not None:
                return self._state
            if self._refresh_task is None:
                self._refresh_task = asyncio.create_task(self._load())
            task = self._refresh_task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lock:
                    if self._refresh_task is task:
                        self._refresh_task = None

    async def _load(self) -> 'MetadataState[KeyT, InfoT]':
        generation = self._generation
        if self.policy is LayerRefreshPolicy.ALWAYS:
            after = None
        else:
            after = None if self._state is None else self._state.revision
        load = await self.backend.load_metadata(after)
        if generation != self._generation:
            return await self._load()
        state = apply_metadata_load(self._state, load)
        self._state = state
        self._generation += 1
        _logger.debug("metadata refreshed revision=%s entries=%s", state.revision, len(state.entries))
        return state

    async def head_revision(self) -> 'StorageRevision | None':
        if self.policy is LayerRefreshPolicy.ALWAYS:
            return None
        if self.revision_source is not None:
            cached = await self.revision_source.head_revision()
            if cached is not None:
                return cached
        return await self.backend.head_revision()

    def invalidate(self) -> None:
        self._state = None
        self._generation += 1

    def apply_write(self, key: KeyT, info: InfoT, revision: StorageRevision) -> None:
        if self._state is None:
            return
        entries = dict(self._state.entries)
        entries[key] = info
        self._state = MetadataState(revision, entries)
        self._generation += 1


__all__ = [
    "LayerMetadataView",
    "LayerRefreshPolicy",
    "MetadataLoad",
    "MetadataLoadMode",
    "MetadataState",
    "RevisionSource",
    "StorageChange",
    "StorageMetadataReader",
    "StorageRevisionSource",
    "apply_metadata_load",
]
