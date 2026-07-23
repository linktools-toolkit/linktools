#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the MemoryManager domain facade. Drives it against an in-process
dict-backed _StubStore implementing the full MemoryStore Protocol (5 methods,
including the _UNSET update sentinel). search returns scored MemoryMatch
results (score=None for the keyword stub)."""

import asyncio
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from linktools.ai.errors import MemoryConflictError
from linktools.ai.memory.manager import MemoryManager
from linktools.ai.memory.models import MemoryMatch, MemoryRecord
from linktools.ai.memory.scope import MemoryScope
from linktools.ai.memory.store import _UNSET


def _scope(*, tenant_id: str = "t1", user_id: "str | None" = "u1") -> MemoryScope:
    return MemoryScope(tenant_id=tenant_id, user_id=user_id)


class _StubStore:
    """Dict-backed in-process MemoryStore: `remember` stores by id (raising on
    duplicate), `search` is tenant-scoped (hard tenant filter + NULL-or-equal
    sub-scope narrowing) and returns MemoryMatch with score=None, mirroring the
    real keyword backends."""

    def __init__(self):
        self._records: "dict[str, MemoryRecord]" = {}

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        return self._records.get(memory_id)

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        category: "str | None" = None,
        limit: int = 10,
    ) -> "tuple[MemoryMatch, ...]":
        out = []
        for r in self._records.values():
            if r.tenant_id != scope.tenant_id:
                continue
            if (
                scope.user_id is not None
                and r.user_id is not None
                and r.user_id != scope.user_id
            ):
                continue
            if (
                scope.workspace_id is not None
                and r.workspace_id is not None
                and r.workspace_id != scope.workspace_id
            ):
                continue
            if (
                scope.session_id is not None
                and r.session_id is not None
                and r.session_id != scope.session_id
            ):
                continue
            if category is not None and r.category != category:
                continue
            if query and query not in r.content:
                continue
            out.append(r)
        return tuple(MemoryMatch(record=r, score=None) for r in out[:limit])

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        if record.id in self._records:
            raise MemoryConflictError(f"memory {record.id} already exists")
        self._records[record.id] = record
        return record

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = _UNSET,
        category: object = _UNSET,
        confidence: object = _UNSET,
        metadata: object = _UNSET,
    ) -> MemoryRecord:
        existing = self._records[memory_id]
        if existing.version != expected_version:
            raise MemoryConflictError("version mismatch")
        fields: "dict[str, object]" = {}
        if content is not _UNSET:
            fields["content"] = content
        if category is not _UNSET:
            fields["category"] = category
        if confidence is not _UNSET:
            fields["confidence"] = confidence
        if metadata is not _UNSET:
            fields["metadata"] = metadata
        fields["version"] = expected_version + 1
        # Replace the stored record with a mutated copy (frozen dataclass).
        updated = replace(existing, **fields)
        self._records[memory_id] = updated
        return updated

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        existing = self._records.get(memory_id)
        if existing is None:
            return
        if existing.version != expected_version:
            raise MemoryConflictError("version mismatch")
        del self._records[memory_id]


# --- MemoryManager.recall / remember ----------------------------------------


def test_remember_then_recall_substring():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        record = await mgr.remember(_scope(), "hello world", category="note")
        assert record.version == 1
        assert record.owner_id == "u1"
        assert record.category == "note"
        # valid uuid4 string
        uuid.UUID(record.id)
        assert record.metadata == {}

        hits = await mgr.recall(_scope(), "hello")
        assert len(hits) == 1
        assert hits[0].record.id == record.id
        assert hits[0].record.content == "hello world"

    asyncio.run(_run())


def test_recall_returns_memory_match_with_none_score():
    # The keyword stub carries no ranking signal: every hit is a MemoryMatch
    # whose score is None (never a fabricated 1.0).
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)
        await mgr.remember(_scope(), "hello world")
        hits = await mgr.recall(_scope(), "hello")
        assert len(hits) == 1
        assert isinstance(hits[0], MemoryMatch)
        assert hits[0].score is None

    asyncio.run(_run())


def test_recall_filters_by_user_subscope():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        # Remembered under user u1 (same tenant); a u2 recall in the same
        # tenant must not see it.
        await mgr.remember(_scope(user_id="u1"), "hello world")
        assert await mgr.recall(_scope(user_id="u2"), "hello") == ()

    asyncio.run(_run())


def test_forget_then_get_is_none():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        record = await mgr.remember(_scope(), "hello world")
        await mgr.forget(record.id, expected_version=1)
        assert await store.get(record.id) is None

    asyncio.run(_run())


def test_remember_round_trips_metadata():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        record = await mgr.remember(_scope(), "hi", metadata={"k": "v"})
        assert record.metadata == {"k": "v"}

    asyncio.run(_run())


def test_manager_has_no_index_parameter():
    # The fake MemoryIndex abstraction is gone: MemoryManager takes only a
    # store, and exposes no index/sync field.
    mgr = MemoryManager(store=_StubStore())
    assert not hasattr(mgr, "index")
