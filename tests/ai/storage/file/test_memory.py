#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/file/test_memory.py — FileMemoryStore contract: JSON-on-disk
persistence for MemoryRecord. Uses the `def test_x(): asyncio.run(_run())`
style (sync test wrapper driving its own event loop) so no pytest-asyncio mode
config is needed."""

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import MemoryConflictError, MemoryNotFoundError
from linktools.ai.memory.models import MemoryRecord
from linktools.ai.memory.scope import MemoryScope
from linktools.ai.storage.file.memory import FileMemoryStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record(
    *,
    memory_id: "str | None" = None,
    tenant_id: str = "t1",
    owner_id: str = "u1",
    content: str = "hello world",
    category: "str | None" = None,
    confidence: "float | None" = None,
    version: int = 1,
    metadata: "dict | None" = None,
    user_id: "str | None" = None,
    workspace_id: "str | None" = None,
    session_id: "str | None" = None,
) -> MemoryRecord:
    now = _now()
    return MemoryRecord(
        id=memory_id or f"m-{uuid.uuid4().hex}",
        tenant_id=tenant_id,
        owner_id=owner_id,
        content=content,
        category=category,
        confidence=confidence,
        version=version,
        created_at=now,
        updated_at=now,
        metadata=metadata if metadata is not None else {},
        user_id=user_id,
        workspace_id=workspace_id,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# 1. remember -> get round-trips all fields
# ---------------------------------------------------------------------------


def test_remember_then_get_roundtrips_all_fields(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
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
        # Datetime precision + tz-awareness preserved across iso round-trip.
        assert fetched.created_at == rec.created_at
        assert fetched.updated_at == rec.updated_at
        assert fetched.created_at.tzinfo is not None

    asyncio.run(_run_case())


def test_get_missing_returns_none(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        assert await store.get("nope") is None

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 2. search filters: owner_id, category, query substring, limit
# ---------------------------------------------------------------------------


def test_search_filters_owner_category_query_and_limit(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        a = _record(owner_id="u1", user_id="u1", content="hello world", category="note")
        b = _record(owner_id="u1", user_id="u1", content="hello there", category="log")
        c = _record(owner_id="u2", user_id="u2", content="hello other", category="note")
        d = _record(owner_id="u1", user_id="u1", content="goodbye", category="note")
        for r in (a, b, c, d):
            await store.remember(r)

        # user sub-scope + query substring (c is a different user -> excluded)
        owned = await store.search(
            "hello", scope=MemoryScope(tenant_id="t1", user_id="u1")
        )
        ids = {r.id for r in owned}
        assert ids == {a.id, b.id}

        # category narrows further
        noted = await store.search(
            "hello", scope=MemoryScope(tenant_id="t1", user_id="u1"), category="note"
        )
        assert {r.id for r in noted} == {a.id}

        # limit truncates (tenant-wide scope sees a, b, c)
        limited = await store.search("hello", scope=MemoryScope(tenant_id="t1"), limit=1)
        assert len(limited) == 1

        # non-matching query returns empty tuple
        assert await store.search("zzz", scope=MemoryScope(tenant_id="t1")) == ()

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 3. remember conflict on duplicate id
# ---------------------------------------------------------------------------


def test_remember_duplicate_id_raises_conflict(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
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
        store = FileMemoryStore(root=tmp_path)
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
        store = FileMemoryStore(root=tmp_path)
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
        store = FileMemoryStore(root=tmp_path)
        rec = _record(version=1)
        await store.remember(rec)
        with pytest.raises(MemoryConflictError):
            await store.update(rec.id, expected_version=99, content="x")

    asyncio.run(_run_case())


def test_update_missing_id_raises_not_found(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        with pytest.raises(MemoryNotFoundError):
            await store.update("ghost", expected_version=1, content="x")

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 6. forget removes file, missing / conflict
# ---------------------------------------------------------------------------


def test_forget_removes_file_and_errors(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
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
# 7. path-traversal in memory_id is rejected
# ---------------------------------------------------------------------------


def test_path_traversal_memory_id_rejected(tmp_path):
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        with pytest.raises(ValueError):
            await store.get("../evil")

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 8. Tenant partitioning + legacy quarantine (§12.8 / §12.10)
# ---------------------------------------------------------------------------


def test_tenant_id_path_traversal_rejected(tmp_path):
    # §12.8: the tenant path segment is validated, so a caller-controlled
    # tenant_id can't escape the store root via "../".
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        with pytest.raises(ValueError):
            await store.search(
                "x", scope=MemoryScope(tenant_id="../evil"), limit=1
            )
        with pytest.raises(ValueError):
            await store.remember(
                _record(memory_id="m-x", tenant_id="../evil")
            )

    asyncio.run(_run_case())


def test_search_isolates_tenants_via_partition(tmp_path):
    # §12.8 / §12.10: search scans ONLY the requesting tenant's subdir, so one
    # tenant can never enumerate another's records (even with a shared owner).
    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        await store.remember(
            _record(memory_id="a1", tenant_id="tenant-a", owner_id="alice", content="hello a")
        )
        await store.remember(
            _record(memory_id="b1", tenant_id="tenant-b", owner_id="alice", content="hello b")
        )
        a_hits = await store.search("hello", scope=MemoryScope(tenant_id="tenant-a"))
        assert {r.id for r in a_hits} == {"a1"}
        b_hits = await store.search("hello", scope=MemoryScope(tenant_id="tenant-b"))
        assert {r.id for r in b_hits} == {"b1"}

    asyncio.run(_run_case())


def test_legacy_flat_record_quarantined_from_real_tenant(tmp_path):
    # §12.9: a pre-tenant record (flat layout, no tenant_id field) is read
    # under the reserved legacy tenant. A real tenant's search must NEVER see
    # it; only an explicit legacy-scope search does (migration quarantine).
    import json as _json

    async def _run_case():
        store = FileMemoryStore(root=tmp_path)
        # Write a legacy flat record directly: root/{id}.json, no tenant_id.
        legacy = {
            "id": "legacy-1",
            "owner_id": "alice",
            "content": "legacy hello secret",
            "category": None,
            "confidence": None,
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        (tmp_path / "legacy-1.json").write_text(_json.dumps(legacy))
        # A real tenant (even "alice's" tenant) does not see the legacy record.
        real_hits = await store.search(
            "legacy", scope=MemoryScope(tenant_id="tenant-a", user_id="alice")
        )
        assert real_hits == ()
        # Only the explicit legacy scope sees it.
        from linktools.ai.memory.scope import LEGACY_TENANT_ID

        legacy_hits = await store.search("legacy", scope=MemoryScope(tenant_id=LEGACY_TENANT_ID))
        assert {r.id for r in legacy_hits} == {"legacy-1"}
        assert legacy_hits[0].tenant_id == LEGACY_TENANT_ID

    asyncio.run(_run_case())
