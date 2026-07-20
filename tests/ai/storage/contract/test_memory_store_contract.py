#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_memory_store_contract.py — runs the same
MemoryStore contract against both FilesystemMemoryStore and SqlAlchemyMemoryStore
(contract backend parity). The parametrized ``store_factory`` fixture is
copied verbatim from ``test_swarm_store_contract.py`` (file + sqlalchemy
branches, including the ``_run_in_new_loop`` helper that bootstraps the SQL
engine off the test loop); ``Base.metadata.create_all`` already covers
``MemoryRow`` since it subclasses the same ``Base``.

Uses the ``def test_x(store_factory):`` + ``asyncio.run(_run())`` style (sync
test wrapper driving its own event loop) — no pytest-asyncio mode config needed."""

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import MemoryConflictError, MemoryNotFoundError
from linktools.ai.memory.models import MemoryRecord
from linktools.ai.memory.scope import MemoryScope
from linktools.ai.storage.filesystem.memory import FilesystemMemoryStore


# ---------------------------------------------------------------------------
# Record builders. Defaults use datetime.now(timezone.utc) (tz-aware) and a
# non-empty metadata mapping so the round-trip test can verify both nullable
# category/confidence and metadata mapping. ``id`` defaults to a stable
# "mem-1" so multi-record tests can pass explicit ids deterministically.
# ---------------------------------------------------------------------------


def make_record(
    memory_id: str = "mem-1",
    tenant_id: str = "t1",
    owner_id: str = "owner-1",
    content: str = "hello world",
    category: "str | None" = "fact",
    confidence: "float | None" = 0.9,
    version: int = 1,
    metadata: "dict | None" = None,
    user_id: "str | None" = None,
    workspace_id: "str | None" = None,
    session_id: "str | None" = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=memory_id,
        tenant_id=tenant_id,
        owner_id=owner_id,
        content=content,
        category=category,
        confidence=confidence,
        version=version,
        created_at=now,
        updated_at=now,
        metadata={"k": "v"} if metadata is None else metadata,
        user_id=user_id,
        workspace_id=workspace_id,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Parametrized store factory. The SQL branch (incl. ``_run_in_new_loop``) is
# copied verbatim from test_swarm_store_contract.py / test_run_store_contract.py.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FilesystemMemoryStore(root=tmp_path / f"mem-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.memory import SqlAlchemyMemoryStore

    counter = {"n": 0}
    engines = []

    def _run_in_new_loop(coro):
        # This factory is called synchronously from inside an already-running
        # pytest-asyncio event loop (the async test function), so we cannot use
        # asyncio.get_event_loop().run_until_complete() here -- that raises
        # "This event loop is already running". Run the setup coroutine to
        # completion on a separate thread with its own fresh event loop instead.
        import asyncio
        import threading

        outcome = {}

        def _runner():
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
                outcome["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def sqlalchemy_factory():
        counter["n"] += 1
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/mem-db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                # MemoryRow subclasses Base, so a single create_all covers
                # every table both backends need.
                await conn.run_sync(Base.metadata.create_all)
            # The connection pool otherwise holds a connection bound to this
            # thread's event loop; dispose it so later operations (running on
            # pytest-asyncio's loop) open fresh connections instead of reusing
            # one tied to a loop that is about to be closed.
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyMemoryStore(session_factory=session_factory)

    def _dispose_engines():
        # The store itself opens fresh connections on pytest-asyncio's loop
        # during the test. Those connections (and aiosqlite's background
        # worker threads) must be disposed before that loop closes at test
        # teardown, otherwise the worker thread tries to call back into an
        # already-closed loop and pytest reports an unraisable exception.
        for engine in engines:
            _run_in_new_loop(engine.dispose())

    request.addfinalizer(_dispose_engines)

    return sqlalchemy_factory


# ---------------------------------------------------------------------------
# 1. remember -> get round-trip (all fields: datetime tz-awareness, metadata
#    mapping, nullable category/confidence).
# ---------------------------------------------------------------------------


def test_remember_then_get_roundtrip(store_factory):
    store = store_factory()

    async def _run():
        record = make_record(
            memory_id="mem-full",
            content="some content",
            category="belief",
            confidence=0.42,
            metadata={"a": 1, "b": "x"},
        )
        created = await store.remember(record)
        fetched = await store.get("mem-full")
        assert fetched is not None
        # Frozen dataclass equality: every field (id, owner_id, content,
        # category, confidence, version, created_at, updated_at, metadata)
        # round-trips identically on both backends.
        assert fetched == created
        # Targeted checks for the load-bearing fields (datetime tz-awareness,
        # metadata mapping, nullable vs. present category/confidence).
        assert fetched.content == "some content"
        assert fetched.category == "belief"
        assert fetched.confidence == 0.42
        assert dict(fetched.metadata) == {"a": 1, "b": "x"}
        assert fetched.created_at.tzinfo is not None
        assert fetched.updated_at.tzinfo is not None
        assert fetched.created_at == created.created_at
        # Nullable category/confidence also round-trip as None on both backends.
        nullable = make_record(
            memory_id="mem-null",
            category=None,
            confidence=None,
        )
        await store.remember(nullable)
        fetched_null = await store.get("mem-null")
        assert fetched_null is not None
        assert fetched_null.category is None
        assert fetched_null.confidence is None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. search: owner_id + category filters, limit, substring; non-matching -> ().
# ---------------------------------------------------------------------------


def test_search_filters_substring_and_limit(store_factory):
    store = store_factory()

    async def _run():
        await store.remember(
            make_record(
                memory_id="m1",
                owner_id="alice",
                user_id="alice",
                content="hello world",
                category="fact",
            )
        )
        await store.remember(
            make_record(
                memory_id="m2",
                owner_id="alice",
                user_id="alice",
                content="goodbye world",
                category="note",
            )
        )
        await store.remember(
            make_record(
                memory_id="m3",
                owner_id="bob",
                user_id="bob",
                content="hello bob",
                category="fact",
            )
        )
        # Substring + user sub-scope narrows to alice's "hello" hit only.
        alice_hello = await store.search(
            "hello", scope=MemoryScope(tenant_id="t1", user_id="alice")
        )
        assert {r.id for r in alice_hello} == {"m1"}
        # Category filter (tenant-wide, no user) covers both users' "fact" rows
        # containing "hello".
        facts = await store.search(
            "hello", scope=MemoryScope(tenant_id="t1"), category="fact"
        )
        assert {r.id for r in facts} == {"m1", "m3"}
        # No matcher -> empty tuple (not None, not list).
        assert await store.search("zzz", scope=MemoryScope(tenant_id="t1")) == ()
        # limit caps results: "world" matches m1 + m2, limit=1 returns one.
        assert (
            len(await store.search("world", scope=MemoryScope(tenant_id="t1"), limit=1))
            == 1
        )
        # Default limit (10) returns every matcher when fewer than limit.
        all_world = await store.search("world", scope=MemoryScope(tenant_id="t1"))
        assert {r.id for r in all_world} == {"m1", "m2"}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. remember duplicate id -> MemoryConflictError.
# ---------------------------------------------------------------------------


def test_remember_duplicate_id_raises_conflict(store_factory):
    store = store_factory()

    async def _run():
        await store.remember(make_record(memory_id="dup-1", content="first"))
        with pytest.raises(MemoryConflictError):
            await store.remember(make_record(memory_id="dup-1", content="second"))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. update: bumps version, applies ONLY provided fields; category=None
#    explicitly CLEARS it (the _UNSET sentinel distinguishes "omit" from
#    "clear").
# ---------------------------------------------------------------------------


def test_update_bumps_version_applies_fields_and_clears_category(store_factory):
    store = store_factory()

    async def _run():
        await store.remember(
            make_record(
                memory_id="u-1",
                content="orig",
                category="fact",
                confidence=0.5,
                metadata={"k": "v"},
            )
        )
        # Pass content/confidence/metadata but NOT category -> category stays.
        updated = await store.update(
            "u-1",
            expected_version=1,
            content="new",
            confidence=0.9,
            metadata={"x": 1},
        )
        assert updated.version == 2
        assert updated.content == "new"
        assert updated.confidence == 0.9
        assert dict(updated.metadata) == {"x": 1}
        # category was NOT passed -> unchanged (sentinel omits the field).
        assert updated.category == "fact"
        # Now explicitly pass category=None -> CLEARS the field (sentinel
        # distinguishes "omit" from "clear").
        cleared = await store.update("u-1", expected_version=2, category=None)
        assert cleared.version == 3
        assert cleared.category is None
        # Other fields are untouched by the clear-only update.
        assert cleared.content == "new"
        assert cleared.confidence == 0.9
        assert dict(cleared.metadata) == {"x": 1}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. update wrong expected_version -> MemoryConflictError; missing id ->
#    MemoryNotFoundError.
# ---------------------------------------------------------------------------


def test_update_wrong_version_and_missing_id_raise(store_factory):
    store = store_factory()

    async def _run():
        await store.remember(make_record(memory_id="u-1"))
        with pytest.raises(MemoryConflictError):
            await store.update("u-1", expected_version=99, content="x")
        with pytest.raises(MemoryNotFoundError):
            await store.update("missing-id", expected_version=1, content="x")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. forget then get -> None; forget missing id -> MemoryNotFoundError.
# ---------------------------------------------------------------------------


def test_forget_then_get_none_and_missing_raises(store_factory):
    store = store_factory()

    async def _run():
        await store.remember(make_record(memory_id="f-1"))
        await store.forget("f-1", expected_version=1)
        assert await store.get("f-1") is None
        with pytest.raises(MemoryNotFoundError):
            await store.forget("missing-id", expected_version=1)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 7. File-only: path-traversal in memory_id -> ValueError. (SQL ids are opaque
#    primary-key strings, not path segments, so this guard is
#    FilesystemMemoryStore-specific — mirrors the file-only path-traversal test in
#    test_swarm_store_contract.py.)
# ---------------------------------------------------------------------------


def test_path_traversal_in_memory_id_is_rejected(tmp_path):
    store = FilesystemMemoryStore(root=tmp_path)

    async def _run():
        with pytest.raises(ValueError):
            await store.get("../evil")
        with pytest.raises(ValueError):
            await store.remember(make_record(memory_id="../evil"))
        with pytest.raises(ValueError):
            await store.update("../evil", expected_version=1, content="x")
        with pytest.raises(ValueError):
            await store.forget("../evil", expected_version=1)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 8. Tenant isolation (§12.10): cross-tenant隔离 with a SHARED owner_id, plus
#    user / workspace / session sub-scope narrowing. owner_id is display-only;
#    tenant_id is the hard boundary. Parametrized over both backends.
# ---------------------------------------------------------------------------


def test_search_isolates_tenants_with_same_owner(store_factory):
    # §12.10: tenant-a/alice and tenant-b/alice share an owner_id but must not
    # see each other's memories. (owner_id is display-only; tenant_id is the
    # authorization boundary.)
    store = store_factory()

    async def _run():
        await store.remember(
            make_record(
                memory_id="a1",
                tenant_id="tenant-a",
                owner_id="alice",
                user_id="alice",
                content="secret tenant-a",
            )
        )
        await store.remember(
            make_record(
                memory_id="b1",
                tenant_id="tenant-b",
                owner_id="alice",
                user_id="alice",
                content="secret tenant-b",
            )
        )
        a_hits = await store.search("secret", scope=MemoryScope(tenant_id="tenant-a"))
        assert {r.id for r in a_hits} == {"a1"}
        b_hits = await store.search("secret", scope=MemoryScope(tenant_id="tenant-b"))
        assert {r.id for r in b_hits} == {"b1"}

    asyncio.run(_run())


def test_search_narrows_by_user_workspace_session_subscope(store_factory):
    # A NULL sub-scope field on the record means "shared at tenant level" and is
    # visible to any user/workspace/session; a NON-NULL field on the scoped axis
    # excludes records whose value on that axis differs. Narrowing is per-axis.
    store = store_factory()

    async def _run():
        await store.remember(
            make_record(memory_id="shared", tenant_id="t1", content="alpha shared")
        )
        # user axis: two distinct users
        await store.remember(
            make_record(memory_id="u1", tenant_id="t1", user_id="u1", content="alpha u1")
        )
        await store.remember(
            make_record(memory_id="u2", tenant_id="t1", user_id="u2", content="alpha u2")
        )
        # workspace axis
        await store.remember(
            make_record(
                memory_id="ws1", tenant_id="t1", workspace_id="ws-1", content="alpha ws1"
            )
        )
        await store.remember(
            make_record(
                memory_id="ws2", tenant_id="t1", workspace_id="ws-2", content="alpha ws2"
            )
        )
        # session axis
        await store.remember(
            make_record(
                memory_id="s1", tenant_id="t1", session_id="sess-1", content="alpha s1"
            )
        )
        await store.remember(
            make_record(
                memory_id="s2", tenant_id="t1", session_id="sess-2", content="alpha s2"
            )
        )
        # tenant-wide (no sub-scope) sees every record.
        all_hits = await store.search("alpha", scope=MemoryScope(tenant_id="t1"))
        assert {r.id for r in all_hits} == {
            "shared",
            "u1",
            "u2",
            "ws1",
            "ws2",
            "s1",
            "s2",
        }
        # user=u1 excludes u2 (records with no user stay visible).
        u1_ids = {
            r.id
            for r in await store.search(
                "alpha", scope=MemoryScope(tenant_id="t1", user_id="u1")
            )
        }
        assert "u1" in u1_ids and "u2" not in u1_ids and "shared" in u1_ids
        # workspace=ws-1 excludes ws2.
        ws_ids = {
            r.id
            for r in await store.search(
                "alpha", scope=MemoryScope(tenant_id="t1", workspace_id="ws-1")
            )
        }
        assert "ws1" in ws_ids and "ws2" not in ws_ids and "shared" in ws_ids
        # session=sess-1 excludes s2.
        sess_ids = {
            r.id
            for r in await store.search(
                "alpha", scope=MemoryScope(tenant_id="t1", session_id="sess-1")
            )
        }
        assert "s1" in sess_ids and "s2" not in sess_ids and "shared" in sess_ids

    asyncio.run(_run())
