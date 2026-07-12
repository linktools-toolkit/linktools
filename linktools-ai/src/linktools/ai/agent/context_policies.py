#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pluggable context-injection strategies, kept out of AgentRunner so the runner
does not hardcode memory limits, owner-resolution semantics, or prompt section
titles. Default implementations provide the built-in semantics; downstream
systems can substitute any of these Protocols.

- MemoryPolicy: selects which MemoryRecords to inject (and may write new ones).
- RetrievalPolicy: retrieves knowledge items for the run.
- PromptContextFormatter: renders memory/knowledge into prompt sections."""

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from ..knowledge.context import KnowledgeItem
    from ..memory.models import MemoryRecord
    from ..run.context import RunContext
    from ..run.models import RunResult


@runtime_checkable
class MemoryPolicy(Protocol):
    async def select_memories(
        self,
        context: "RunContext",
        query: str,
    ) -> "Sequence[MemoryRecord]": ...

    async def maybe_write_memories(
        self,
        context: "RunContext",
        result: "RunResult",
    ) -> None: ...


@runtime_checkable
class RetrievalPolicy(Protocol):
    async def retrieve(
        self,
        context: "RunContext",
        query: str,
    ) -> "Sequence[KnowledgeItem]": ...


@runtime_checkable
class PromptContextFormatter(Protocol):
    def format_memory(self, records: "Sequence[MemoryRecord]") -> str: ...
    def format_knowledge(self, items: "Sequence[KnowledgeItem]") -> str: ...


class DefaultMemoryPolicy:
    """Searches the memory store scoped to the run owner (user -> tenant ->
    session) with a small limit. ``maybe_write_memories`` is a no-op by
    default; writing is a separate policy decision."""

    def __init__(self, store: Any, *, limit: int = 5) -> None:
        self._store = store
        self._limit = limit

    async def select_memories(self, context, query):
        owner = context.user_id or context.tenant_id or context.session_id
        return await self._store.search(query, owner_id=owner, limit=self._limit)

    async def maybe_write_memories(self, context, result) -> None:
        return None


class DefaultRetrievalPolicy:
    """Retrieves knowledge items via the wired retriever with a small limit."""

    def __init__(self, retriever: Any, *, limit: int = 5) -> None:
        self._retriever = retriever
        self._limit = limit

    async def retrieve(self, context, query):
        return await self._retriever.search(query, limit=self._limit)


class DefaultPromptContextFormatter:
    """Renders memory/knowledge into the prompt sections (the historical shape).
    Substitute this to change titles or ordering."""

    def format_memory(self, records):
        from ..knowledge.context import format_memory

        return format_memory(records)

    def format_knowledge(self, items):
        from ..knowledge.context import KnowledgeContext

        return KnowledgeContext(documents=list(items)).format()


__all__ = [
    "MemoryPolicy",
    "RetrievalPolicy",
    "PromptContextFormatter",
    "DefaultMemoryPolicy",
    "DefaultRetrievalPolicy",
    "DefaultPromptContextFormatter",
]
