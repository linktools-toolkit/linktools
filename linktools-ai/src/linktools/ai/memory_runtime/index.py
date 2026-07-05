#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryIndex: decoupled ranked-search abstraction over memories. Extension
point for a future vector index; KeywordMemoryIndex (Task 2) adapts a MemoryStore."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import MemoryRecord


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    memory_id: str
    score: float


@runtime_checkable
class MemoryIndex(Protocol):
    async def index(self, record: MemoryRecord) -> None: ...
    async def remove(self, memory_id: str) -> None: ...
    async def search(self, query: str, *, limit: int = 10) -> "tuple[MemorySearchHit, ...]": ...
