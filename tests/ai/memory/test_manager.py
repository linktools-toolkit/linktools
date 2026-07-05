#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for MemoryManager domain facade and KeywordMemoryIndex adapter.
Drives both against an in-process dict-backed _StubStore implementing the
full MemoryStore Protocol (5 methods, including the _UNSET update sentinel)."""

import asyncio
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from linktools.ai.errors import MemoryConflictError
from linktools.ai.memory.index import KeywordMemoryIndex, MemorySearchHit
from linktools.ai.memory.manager import MemoryManager
from linktools.ai.memory.models import MemoryRecord
from linktools.ai.memory.store import _UNSET


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _StubStore:
    """Dict-backed in-process MemoryStore: `remember` stores by id (raising on
    duplicate), `search` filters by owner_id + substring on content."""

    def __init__(self):
        self._records: "dict[str, MemoryRecord]" = {}

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        return self._records.get(memory_id)

    async def search(
        self,
        query: str,
        *,
        owner_id: "str | None" = None,
        category: "str | None" = None,
        limit: int = 10,
    ) -> "tuple[MemoryRecord, ...]":
        out = []
        for r in self._records.values():
            if owner_id is not None and r.owner_id != owner_id:
                continue
            if category is not None and r.category != category:
                continue
            if query and query not in r.content:
                continue
            out.append(r)
        return tuple(out[:limit])

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

        record = await mgr.remember("u1", "hello world", category="note")
        assert record.version == 1
        assert record.owner_id == "u1"
        assert record.category == "note"
        # valid uuid4 string
        uuid.UUID(record.id)
        assert record.metadata == {}

        hits = await mgr.recall("u1", "hello")
        assert len(hits) == 1
        assert hits[0].id == record.id
        assert hits[0].content == "hello world"

    asyncio.run(_run())


def test_recall_filters_by_owner_id():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        await mgr.remember("u1", "hello world")
        assert await mgr.recall("u2", "hello") == ()

    asyncio.run(_run())


def test_forget_then_get_is_none():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        record = await mgr.remember("u1", "hello world")
        await mgr.forget(record.id, expected_version=1)
        assert await store.get(record.id) is None

    asyncio.run(_run())


def test_remember_round_trips_metadata():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)

        record = await mgr.remember("u1", "hi", metadata={"k": "v"})
        assert record.metadata == {"k": "v"}

    asyncio.run(_run())


# --- KeywordMemoryIndex ------------------------------------------------------


def test_keyword_index_search_returns_uniform_score():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)
        record = await mgr.remember("u1", "hello world")

        idx = KeywordMemoryIndex(store)
        hits = await idx.search("hello")
        assert len(hits) == 1
        assert hits[0] == MemorySearchHit(memory_id=record.id, score=1.0)

    asyncio.run(_run())


def test_keyword_index_search_no_match_returns_empty():
    async def _run():
        store = _StubStore()
        mgr = MemoryManager(store=store)
        await mgr.remember("u1", "hello world")

        idx = KeywordMemoryIndex(store)
        assert await idx.search("nomatch") == ()

    asyncio.run(_run())


def test_keyword_index_index_and_remove_are_noops():
    async def _run():
        store = _StubStore()
        idx = KeywordMemoryIndex(store)

        now = _now()
        record = MemoryRecord(
            id="m-x",
            owner_id="u1",
            content="x",
            category=None,
            confidence=None,
            version=1,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        # Should not raise; both return None (no-op).
        assert await idx.index(record) is None
        assert await idx.remove("m-x") is None

    asyncio.run(_run())


# --- MemoryManager + index integration --------------------------------------


def test_manager_with_index_remember_and_forget():
    async def _run():
        store = _StubStore()
        idx = KeywordMemoryIndex(store)
        mgr = MemoryManager(store=store, index=idx)

        record = await mgr.remember("u1", "hello world")
        # index.search should reflect the remembered record (index.index was called).
        hits = await idx.search("hello")
        assert len(hits) == 1
        assert hits[0].memory_id == record.id

        await mgr.forget(record.id, expected_version=1)
        # After forget, index.remove was called (store-backed index reflects it).
        assert await mgr.recall("u1", "hello") == ()

    asyncio.run(_run())
