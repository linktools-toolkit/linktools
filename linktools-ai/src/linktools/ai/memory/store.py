#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryStore Protocol: persistence + search contract for MemoryRecord.
Method signatures mirror the optimistic-concurrency shape of the execution
store and task store. update/forget take expected_version because both
backends advertise optimistic_concurrency=True.

``search`` is tenant-scoped: it takes a required :class:`MemoryScope` and no
``scope=None`` global-search path exists. ``category`` is retained as an
optional orthogonal content filter (it carries no authorization weight)."""

from typing import Protocol, runtime_checkable

from ..storage.features import ComponentCapabilities
from .models import MemoryMatch, MemoryRecord
from .scope import MemoryScope

UNSET = (
    object()
)  # sentinel: passing category=None CLEARS the field; omitting leaves unchanged.


@runtime_checkable
class MemoryPort(Protocol):
    @property
    def capabilities(self) -> ComponentCapabilities: ...

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
    def __init__(self, backend: MemoryPort) -> None:
        self.backend = backend

    def __getattr__(self, name: str):
        return getattr(self.backend, name)
