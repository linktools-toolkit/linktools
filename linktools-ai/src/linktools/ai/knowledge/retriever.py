#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retriever Protocol + MemoryRetriever: projects MemoryRecord -> Document."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .document import Document
from .scope import RetrievalScope

if TYPE_CHECKING:
    from ..memory.scope import MemoryScope
    from ..memory.store import MemoryStore


@runtime_checkable
class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int = 10,
    ) -> "tuple[Document, ...]": ...


class MemoryRetriever:
    """Adapts a MemoryStore into a Retriever: projects MemoryRecord -> Document.

    ``scope`` is required on every search and is forwarded to the underlying
    MemoryStore (translated from RetrievalScope to MemoryScope). There is no
    global / unscoped search path."""

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    async def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int = 10,
    ) -> "tuple[Document, ...]":
        from ..memory.scope import MemoryScope

        memory_scope: "MemoryScope" = MemoryScope(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            session_id=scope.session_id,
        )
        records = await self._store.search(query, scope=memory_scope, limit=limit)
        return tuple(
            Document(
                id=r.id,
                content=r.content,
                score=None,
                source="memory",
                metadata=dict(r.metadata),
                trust_level="untrusted",
            )
            for r in records
        )
