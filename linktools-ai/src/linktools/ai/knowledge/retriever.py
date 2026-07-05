#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retriever Protocol + MemoryRetriever: projects MemoryRecord -> Document."""

from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

from .._typing import JSONValue
from .document import Document

if TYPE_CHECKING:
    from ..memory.store import MemoryStore


@runtime_checkable
class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        filters: "Mapping[str, JSONValue] | None" = None,
        limit: int = 10,
    ) -> "tuple[Document, ...]": ...


class MemoryRetriever:
    """Adapts a MemoryStore into a Retriever: projects MemoryRecord -> Document."""

    def __init__(self, store: "MemoryStore", *, owner_id: "str | None" = None) -> None:
        self._store = store
        self._owner_id = owner_id

    async def search(
        self,
        query: str,
        *,
        filters: "Mapping[str, JSONValue] | None" = None,
        limit: int = 10,
    ) -> "tuple[Document, ...]":
        owner_id = self._owner_id
        if filters is not None and "owner_id" in filters:
            owner_id = str(filters["owner_id"])
        records = await self._store.search(query, owner_id=owner_id, limit=limit)
        return tuple(
            Document(
                id=r.id,
                content=r.content,
                score=None,
                source="memory",
                metadata=dict(r.metadata),
            )
            for r in records
        )
