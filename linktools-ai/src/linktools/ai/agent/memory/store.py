#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory backend contract and composed Store.

Method signatures mirror the optimistic-concurrency shape of the execution
store and task store. update/forget take expected_version because both
backends advertise optimistic_concurrency=True.

``search`` is tenant-scoped: it takes a required :class:`MemoryScope` and no
``scope=None`` global-search path exists. ``category`` is retained as an
optional orthogonal content filter (it carries no authorization weight)."""

from typing import Protocol, runtime_checkable

from .models import MemoryMatch, MemoryRecord
from .scope import MemoryScope

UNSET = (
    object()
)  # sentinel: passing category=None CLEARS the field; omitting leaves unchanged.


@runtime_checkable
class MemoryBackend(Protocol):
    async def get(self, memory_id: str) -> "MemoryRecord | None": ...

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 10,
        category: "str | None" = None,
    ) -> "tuple[MemoryMatch, ...]": ...

    async def remember(self, record: MemoryRecord) -> MemoryRecord: ...

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = UNSET,
        category: object = UNSET,
        confidence: object = UNSET,
        metadata: object = UNSET,
    ) -> MemoryRecord: ...

    async def forget(self, memory_id: str, *, expected_version: int) -> None: ...


class MemoryStore:
    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend

    async def initialize_storage(self, *args: object) -> None:
        await self._backend.initialize_storage(*args)

    async def get(self, memory_id: str) -> MemoryRecord | None:
        return await self._backend.get(memory_id)

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 10,
        category: str | None = None,
    ) -> tuple[MemoryMatch, ...]:
        return await self._backend.search(
            query,
            scope=scope,
            limit=limit,
            category=category,
        )

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        return await self._backend.remember(record)

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = UNSET,
        category: object = UNSET,
        confidence: object = UNSET,
        metadata: object = UNSET,
    ) -> MemoryRecord:
        return await self._backend.update(
            memory_id,
            expected_version=expected_version,
            content=content,
            category=category,
            confidence=confidence,
            metadata=metadata,
        )

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        await self._backend.forget(memory_id, expected_version=expected_version)


__all__ = ["MemoryBackend", "MemoryStore", "UNSET"]
