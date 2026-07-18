#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryStore Protocol: persistence + search contract for MemoryRecord.
Method signatures resolve the spec's `(...)` ellipses, mirroring
the optimistic-concurrency shape of RunStore/SwarmStore. update/forget take
expected_version because both backends advertise optimistic_concurrency=True.

``search`` is tenant-scoped: it takes a required :class:`MemoryScope` and no
``scope=None`` global-search path exists. ``category`` is retained as an
optional orthogonal content filter (it carries no authorization weight)."""

from typing import Protocol, runtime_checkable

from .models import MemoryRecord
from .scope import MemoryScope

_UNSET = (
    object()
)  # sentinel: passing category=None CLEARS the field; omitting leaves unchanged.


@runtime_checkable
class MemoryStore(Protocol):
    async def get(self, memory_id: str) -> "MemoryRecord | None": ...

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 10,
        category: "str | None" = None,
    ) -> "tuple[MemoryRecord, ...]": ...

    async def remember(self, record: MemoryRecord) -> MemoryRecord: ...

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = _UNSET,
        category: object = _UNSET,
        confidence: object = _UNSET,
        metadata: object = _UNSET,
    ) -> MemoryRecord: ...

    async def forget(self, memory_id: str, *, expected_version: int) -> None: ...
