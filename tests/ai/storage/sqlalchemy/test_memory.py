#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/test_memory.py — SqlAlchemyMemoryStore contract.
Uses the `def test_x(): asyncio.run(_run())` style (sync test wrapper driving its
own event loop) so no pytest-asyncio mode config is needed. Mirrors how
test_swarm.py bootstraps the in-file aiosqlite engine; mirrors the File test's
5-method contract (round-trip, search, conflict-on-dup, update semantics incl.
clear-via-None, version conflict, not-found, forget) plus one SQL-specific
indexed-category-filter test."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import MemoryConflictError, MemoryNotFoundError
from linktools.ai.memory.models import MemoryRecord
from linktools.ai.storage.sqlalchemy.memory import SqlAlchemyMemoryStore
from linktools.ai.storage.sqlalchemy.models import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record(
    *,
    memory_id: "str | None" = None,
    owner_id: str = "u1",
    content: str = "hello world",
    category: "str | None" = None,
    confidence: "float | None" = None,
    version: int = 1,
    metadata: "dict | None" = None,
) -> MemoryRecord:
    now = _now()
    return MemoryRecord(
        id=memory_id or f"m-{uuid.uuid4().hex}",
        owner_id=owner_id,
        content=content,
        category=category,
        confidence=confidence,
        version=version,
        created_at=now,
        updated_at=now,
        metadata=metadata if metadata is not None else {},
    )


@asynccontextmanager
async def _store_ctx(tmp_path):
    """Build a SqlAlchemyMemoryStore against an in-file SQLite DB. The engine is
    disposed on exit so aiosqlite's background worker threads shut down before
    the per-test event loop closes (otherwise they call into a dead loop)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/mem.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        yield SqlAlchemyMemoryStore(session_factory=session_factory)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# 1. remember -> get round-trips all fields
# ---------------------------------------------------------------------------


def test_remember_then_get_roundtrips_all_fields(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            rec = _record(
                content="hello world",
                category=None,
                confidence=0.5,
                metadata={"k": "v"},
            )
            await store.remember(rec)
            fetched = await store.get(rec.id)
            assert fetched is not None
            assert fetched.id == rec.id
            assert fetched.owner_id == "u1"
            assert fetched.content == "hello world"
            assert fetched.category is None
            assert fetched.confidence == 0.5
            assert fetched.version == 1
            assert fetched.metadata == {"k": "v"}
            # datetime reattached with UTC tzinfo on read (aiosqlite strips it).
            assert fetched.created_at == rec.created_at
            assert fetched.updated_at == rec.updated_at
            assert fetched.created_at.tzinfo is not None

    asyncio.run(_run_case())


def test_get_missing_returns_none(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            assert await store.get("nope") is None

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 2. search filters: owner_id, category, query substring, limit
# ---------------------------------------------------------------------------


def test_search_filters_owner_category_query_and_limit(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            a = _record(owner_id="u1", content="hello world", category="note")
            b = _record(owner_id="u1", content="hello there", category="log")
            c = _record(owner_id="u2", content="hello other", category="note")
            d = _record(owner_id="u1", content="goodbye", category="note")
            for r in (a, b, c, d):
                await store.remember(r)

            # owner_id + query substring (LIKE is case-insensitive on ASCII here,
            # but the substring 'hello' is what binds).
            owned = await store.search("hello", owner_id="u1")
            ids = {r.id for r in owned}
            assert ids == {a.id, b.id}

            # category narrows further
            noted = await store.search("hello", owner_id="u1", category="note")
            assert {r.id for r in noted} == {a.id}

            # limit truncates
            limited = await store.search("hello", limit=1)
            assert len(limited) == 1

            # non-matching query returns empty tuple
            assert await store.search("zzz") == ()

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 3. remember conflict on duplicate id (IntegrityError -> MemoryConflictError)
# ---------------------------------------------------------------------------


def test_remember_duplicate_id_raises_conflict(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            rec = _record(memory_id="dup-1")
            await store.remember(rec)
            with pytest.raises(MemoryConflictError):
                await store.remember(_record(memory_id="dup-1"))

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 4. update applies fields, sentinel clears category, omitted fields unchanged
# ---------------------------------------------------------------------------


def test_update_bumps_version_and_applies_content(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            rec = _record(content="old", version=1)
            await store.remember(rec)
            updated = await store.update(rec.id, expected_version=1, content="new")
            assert updated.version == 2
            assert updated.content == "new"
            assert updated.created_at == rec.created_at
            assert updated.updated_at >= rec.updated_at

    asyncio.run(_run_case())


def test_update_category_none_clears_category_and_omitted_fields_unchanged(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            rec = _record(
                content="keep", category="note", confidence=0.8, metadata={"k": "v"}
            )
            await store.remember(rec)
            # category=None with the sentinel means EXPLICIT clear.
            updated = await store.update(rec.id, expected_version=1, category=None)
            assert updated.category is None
            # Omitted fields are untouched.
            assert updated.content == "keep"
            assert updated.confidence == 0.8
            assert updated.metadata == {"k": "v"}

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 5. update conflict + missing
# ---------------------------------------------------------------------------


def test_update_wrong_expected_version_raises_conflict(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            rec = _record(version=1)
            await store.remember(rec)
            with pytest.raises(MemoryConflictError):
                await store.update(rec.id, expected_version=99, content="x")

    asyncio.run(_run_case())


def test_update_missing_id_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(MemoryNotFoundError):
                await store.update("ghost", expected_version=1, content="x")

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 6. forget removes row, missing / conflict
# ---------------------------------------------------------------------------


def test_forget_removes_row_and_errors(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            rec = _record(version=1)
            await store.remember(rec)
            await store.forget(rec.id, expected_version=1)
            assert await store.get(rec.id) is None
            # forget on missing id
            with pytest.raises(MemoryNotFoundError):
                await store.forget("ghost", expected_version=1)
            # forget with wrong version on a re-created record
            rec2 = _record(memory_id=rec.id, version=1)
            await store.remember(rec2)
            with pytest.raises(MemoryConflictError):
                await store.forget(rec.id, expected_version=99)

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 7. SQL-specific: indexed category filter excludes other categories
# (two records same owner, different category, search(category=...) returns
# only the matching one — exercises the category index.)
# ---------------------------------------------------------------------------


def test_search_category_filter_isolates_categories(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            note_rec = _record(
                memory_id="m-note",
                owner_id="user-x",
                content="same query",
                category="note",
            )
            log_rec = _record(
                memory_id="m-log",
                owner_id="user-x",
                content="same query",
                category="log",
            )
            await store.remember(note_rec)
            await store.remember(log_rec)
            only_note = await store.search("same query", category="note")
            assert {r.id for r in only_note} == {"m-note"}
            only_log = await store.search("same query", category="log")
            assert {r.id for r in only_log} == {"m-log"}

    asyncio.run(_run_case())
