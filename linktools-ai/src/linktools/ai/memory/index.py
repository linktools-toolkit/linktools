#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryIndex: decoupled ranked-search abstraction over memories. Extension
point for a future vector index; KeywordMemoryIndex adapts a MemoryStore.

``search`` is tenant-scoped (takes a required :class:`MemoryScope`) so a future
vector index can never return a neighbor across the tenant boundary."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import MemoryRecord
from .scope import MemoryScope

if TYPE_CHECKING:
    from .store import MemoryStore


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    memory_id: str
    score: float


class MemoryIndexStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class MemoryIndexEvent:
    memory_id: str
    operation: str
    version: int


@runtime_checkable
class MemoryIndex(Protocol):
    async def index(self, record: MemoryRecord) -> None: ...
    async def remove(self, memory_id: str) -> None: ...
    async def search(
        self, query: str, *, scope: MemoryScope, limit: int = 10
    ) -> "tuple[MemorySearchHit, ...]": ...


class KeywordMemoryIndex:
    """Adapts a MemoryStore into a MemoryIndex: search delegates to the store's
    keyword search and assigns every hit a uniform score=1.0. Owns no storage."""

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    async def index(self, record: MemoryRecord) -> None:
        # no-op: the store is the index of record.
        return None

    async def remove(self, memory_id: str) -> None:
        # no-op.
        return None

    async def search(
        self, query: str, *, scope: MemoryScope, limit: int = 10
    ) -> "tuple[MemorySearchHit, ...]":
        records = await self._store.search(query, scope=scope, limit=limit)
        return tuple(MemorySearchHit(memory_id=r.id, score=1.0) for r in records)

    async def rebuild(self, *, scope: MemoryScope, limit: int = 100_000) -> int:
        """Reconcile the derived index from the authoritative MemoryStore."""
        records = await self._store.search("", scope=scope, limit=limit)
        for record in records:
            await self.index(record)
        return len(records)
