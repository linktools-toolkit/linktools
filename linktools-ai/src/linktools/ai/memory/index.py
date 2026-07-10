#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryIndex: decoupled ranked-search abstraction over memories. Extension
point for a future vector index; KeywordMemoryIndex adapts a MemoryStore."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import MemoryRecord

if TYPE_CHECKING:
    from .store import MemoryStore


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    memory_id: str
    score: float


@runtime_checkable
class MemoryIndex(Protocol):
    async def index(self, record: MemoryRecord) -> None: ...
    async def remove(self, memory_id: str) -> None: ...
    async def search(self, query: str, *, limit: int = 10) -> "tuple[MemorySearchHit, ...]": ...


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

    async def search(self, query: str, *, limit: int = 10) -> "tuple[MemorySearchHit, ...]":
        records = await self._store.search(query, limit=limit)
        return tuple(MemorySearchHit(memory_id=r.id, score=1.0) for r in records)
