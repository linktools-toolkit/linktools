#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryManager: domain facade over MemoryStore (+ optional MemoryIndex).
recall/remember/forget; mints id (uuid4)/version/timestamps on remember. Every
read/write carries a :class:`MemoryScope` so the tenant boundary is enforced
end to end -- there is no unscoped path."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Mapping

from .._typing import JSONValue
from .models import MemoryRecord
from .scope import MemoryScope

if TYPE_CHECKING:
    from .index import MemoryIndex
    from .store import MemoryStore


@dataclass
class MemoryManager:
    store: "MemoryStore"
    index: "MemoryIndex | None" = None

    async def recall(
        self, scope: MemoryScope, query: str, *, limit: int = 10
    ) -> "tuple[MemoryRecord, ...]":
        return await self.store.search(query, scope=scope, limit=limit)

    async def remember(
        self,
        scope: MemoryScope,
        content: str,
        *,
        owner_id: "str | None" = None,
        category: "str | None" = None,
        confidence: "float | None" = None,
        metadata: "Mapping[str, JSONValue] | None" = None,
    ) -> MemoryRecord:
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            tenant_id=scope.tenant_id,
            owner_id=owner_id if owner_id is not None else scope.user_id or scope.tenant_id,
            content=content,
            category=category,
            confidence=confidence,
            version=1,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata) if metadata is not None else {},
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            session_id=scope.session_id,
        )
        persisted = await self.store.remember(record)
        if self.index is not None:
            await self.index.index(persisted)
        return persisted

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        await self.store.forget(memory_id, expected_version=expected_version)
        if self.index is not None:
            await self.index.remove(memory_id)
