#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryManager: domain facade over MemoryStore (+ optional MemoryIndex).
recall/remember/forget; mints id (uuid4)/version/timestamps on remember."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Mapping

from .._typing import JSONValue
from .models import MemoryRecord

if TYPE_CHECKING:
    from .index import MemoryIndex
    from .store import MemoryStore


@dataclass
class MemoryManager:
    store: "MemoryStore"
    index: "MemoryIndex | None" = None

    async def recall(
        self, owner_id: str, query: str, *, limit: int = 10
    ) -> "tuple[MemoryRecord, ...]":
        return await self.store.search(query, owner_id=owner_id, limit=limit)

    async def remember(
        self,
        owner_id: str,
        content: str,
        *,
        category: "str | None" = None,
        confidence: "float | None" = None,
        metadata: "Mapping[str, JSONValue] | None" = None,
    ) -> MemoryRecord:
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            content=content,
            category=category,
            confidence=confidence,
            version=1,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata) if metadata is not None else {},
        )
        persisted = await self.store.remember(record)
        if self.index is not None:
            await self.index.index(persisted)
        return persisted

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        await self.store.forget(memory_id, expected_version=expected_version)
        if self.index is not None:
            await self.index.remove(memory_id)
