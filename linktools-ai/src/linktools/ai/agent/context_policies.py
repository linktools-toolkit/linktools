#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pluggable context-injection strategies, kept out of AgentEngine so the runner
does not hardcode memory limits, owner-resolution semantics, or prompt section
titles. Default implementations provide the built-in semantics; downstream
systems can substitute any of these Protocols.

- MemoryPolicy: selects which MemoryRecords to inject (and may write new ones).
- RetrievalPolicy: retrieves knowledge items for the run.
- PromptContextFormatter: renders memory/knowledge into prompt sections.

Both default policies build a tenant-bound scope from the RunContext and FAIL
CLOSED when the context has no tenant: they return an empty result rather than
searching globally. There is no unscoped / cross-tenant retrieval path."""

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from ..retrieval.context import KnowledgeItem
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


def _workspace_id_from(context: "RunContext") -> "str | None":
    # The spec derives workspace_id from metadata; fall back to the WorkspaceRef
    # id when the run carries one but the metadata key was not set.
    ws = context.metadata.get("workspace_key")
    if ws is not None:
        return str(ws)
    if context.workspace is not None:
        return getattr(context.workspace, "id", None)
    return None


class DefaultMemoryPolicy:
    """Searches the memory store scoped to the run's tenant (narrowed by user /
    workspace / session sub-scopes) with a small limit. A run without a tenant
    gets NO memories -- fail closed, never a global search.
    ``maybe_write_memories`` is a no-op by default; writing is a separate policy
    decision."""

    def __init__(self, store: Any, *, limit: int = 5) -> None:
        self._store = store
        self._limit = limit

    async def select_memories(self, context, query):
        from ..memory.scope import MemoryScope

        if not context.tenant_id:
            # Fail closed: a missing tenant never searches globally.
            return ()
        scope = MemoryScope(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            workspace_id=_workspace_id_from(context),
            session_id=context.session_id,
        )
        return await self._store.search(query, scope=scope, limit=self._limit)

    async def maybe_write_memories(self, context, result) -> None:
        return None


class DefaultRetrievalPolicy:
    """Retrieves knowledge items via the wired retriever, scoped to the run's
    tenant. A run without a tenant retrieves nothing -- fail closed."""

    def __init__(self, retriever: Any, *, limit: int = 5) -> None:
        self._retriever = retriever
        self._limit = limit

    async def retrieve(self, context, query):
        from ..retrieval.scope import RetrievalScope

        if not context.tenant_id:
            return ()
        scope = RetrievalScope(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            workspace_id=_workspace_id_from(context),
            session_id=context.session_id,
        )
        return await self._retriever.search(query, scope=scope, limit=self._limit)


class DefaultPromptContextFormatter:
    """Renders memory/knowledge into the prompt sections (the historical shape).
    Substitute this to change titles or ordering."""

    def format_memory(self, records):
        from ..retrieval.context import format_memory

        return format_memory(records)

    def format_knowledge(self, items):
        from ..retrieval.context import KnowledgeContext

        return KnowledgeContext(documents=list(items)).format()


__all__ = [
    "MemoryPolicy",
    "RetrievalPolicy",
    "PromptContextFormatter",
    "DefaultMemoryPolicy",
    "DefaultRetrievalPolicy",
    "DefaultPromptContextFormatter",
]
