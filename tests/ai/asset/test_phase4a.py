#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset storage contract tests:

1. Atomic Asset MOVE -- SqlAlchemy raw_move executes in ONE transaction.
   Observable proof: the revision counter bumps exactly once. A decomposed
   put+delete would bump twice (once for the put, once for the delete), so a
   delta of exactly 1 is structural proof of single-transaction atomicity --
   no intermediate state (target written while source still live, or source
   masked while target missing) is ever committed.

2. stat() metadata-only -- raw_stat returns AssetInfo (which has no content
   field) and, on SqlAlchemy, the issued SELECT does not reference the content
   column. Verified behaviorally (result has no .content attribute) and
   structurally (SQL capture).

3. Cursor pagination -- list with a small limit pages through every item
   exactly once via successive cursors, terminating with cursor=None. Verified
   across all three backends (memory/file/sqlalchemy) since cursor handling
   lives in each backend's raw_list."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.asset.models import Depth, WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.sqlalchemy.asset import SqlAlchemyAssetBackend


# ---- shared backend_factory (mirrors test_store.py's parametrization) ----


def _memory_backend(**kwargs):
    from linktools.ai.asset.memory import MemoryAssetBackend

    return MemoryAssetBackend(**kwargs)


def _file_backend(tmp_path, **kwargs):
    from linktools.ai.asset.file import FileAssetBackend

    return FileAssetBackend(root=tmp_path, **kwargs)


@pytest.fixture(params=["memory", "file", "sqlalchemy"])
def backend_factory(request, tmp_path):
    if request.param == "memory":
        return lambda **kw: _memory_backend(**kw)
    if request.param == "file":
        counter = {"n": 0}

        def file_factory(**kw):
            counter["n"] += 1
            return _file_backend(tmp_path / f"backend-{counter['n']}", **kw)

        return file_factory

    counter = {"n": 0}
    engines = []

    def _run_in_new_loop(coro):
        import asyncio
        import threading

        outcome = {}

        def _runner():
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def sqlalchemy_factory(**kw):
        counter["n"] += 1
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyAssetBackend(session_factory=session_factory, **kw)

    return sqlalchemy_factory


async def _make_sqlalchemy_store(tmp_path, db_name: str = "phase4a.db"):
    """Dedicated SqlAlchemy store for the atomic-MOVE test (needs direct access
    to backend.revision() and the SQL event hook)."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / db_name}",
        connect_args={"timeout": 30.0},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    backend = SqlAlchemyAssetBackend(session_factory=session_factory)
    store = AssetStore(primary=backend)
    return engine, backend, store


# ----------------------------------------------------------------------
# Test 1: Atomic MOVE in ONE transaction (contract)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_atomic_move_bumps_revision_exactly_once(tmp_path):
    """contract atomic-MOVE guard: a MOVE on a primary-resident source must bump
    the revision counter exactly once. A decomposed put+delete (the pre-fix
    orchestration) would bump twice -- once for raw_put, once for raw_delete.
    A delta of exactly 1 is observable proof that all mutations committed
    inside a single transaction, so a concurrent reader cannot observe the
    intermediate state (target written while source still live = duplicate, or
    source masked while target missing = data loss)."""
    engine, backend, store = await _make_sqlalchemy_store(tmp_path)
    try:
        await store.put(
            AssetPath("/src.txt"),
            b"payload",
            options=WriteOptions(metadata={"k": "v"}),
        )
        before = await backend.revision()

        moved = await store.move(AssetPath("/src.txt"), AssetPath("/dst.txt"))

        after = await backend.revision()
        assert after == before + 1, (
            f"atomic MOVE must bump revision exactly once (delta={after - before}); "
            "a delta of 2 means the move decomposed into put+delete"
        )

        # Final state: source masked, target carries the content + metadata.
        assert await store.get(AssetPath("/src.txt")) is None
        target = await store.get(AssetPath("/dst.txt"))
        assert target is not None
        assert target.content == b"payload"
        assert target.info.metadata == {"k": "v"}
        # The moved Asset returned to the caller reflects the same payload.
        assert moved.content == b"payload"
        assert moved.info.path.value == "/dst.txt"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_atomic_move_refuses_overlay_only_source(tmp_path):
    """contract overlay-source MOVE: an overlay-only source cannot be moved
    atomically (the source lives in a different backend; a cross-backend copy +
    whiteout is not an atomic move). AssetStore refuses it with
    AssetMoveNotSupportedError rather than faking the move via put+delete."""
    from linktools.ai.errors import AssetMoveNotSupportedError

    engine, backend, store = await _make_sqlalchemy_store(tmp_path)
    try:
        overlay_engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/overlay.db"
        )
        async with overlay_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        overlay_session_factory = async_sessionmaker(
            overlay_engine, expire_on_commit=False
        )
        overlay = SqlAlchemyAssetBackend(
            session_factory=overlay_session_factory
        )
        await overlay.raw_put(
            AssetPath("/src.md"), b"overlay content", content_type=None, metadata={}
        )
        store_with_overlay = AssetStore(primary=backend, overlays=(overlay,))

        with pytest.raises(AssetMoveNotSupportedError):
            await store_with_overlay.move(
                AssetPath("/src.md"), AssetPath("/dst.md")
            )
        await overlay_engine.dispose()
    finally:
        await engine.dispose()


# ----------------------------------------------------------------------
# Test 2: stat() is metadata-only (contract)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stat_returns_metadata_without_content_field(backend_factory):
    """contract behavioral guard: stat() returns a AssetLookupInfo (= alias of
    AssetInfo), which has no content field. AssetStore.stat() must NOT
    internally call get() (which would load the content blob) when the backend
    exposes raw_stat -- the result type itself proves content was not pulled
    into the returned object."""
    store = AssetStore(primary=backend_factory())
    await store.put(
        AssetPath("/a.txt"),
        b"large-payload",
        options=WriteOptions(metadata={"k": "v"}),
    )

    info = await store.stat(AssetPath("/a.txt"))

    assert info is not None
    assert info.path.value == "/a.txt"
    assert info.metadata == {"k": "v"}
    # AssetInfo is a slotted dataclass with no content field; accessing it
    # must raise AttributeError -- proving stat() returned metadata only.
    assert not hasattr(info, "content")
    with pytest.raises(AttributeError):
        _ = info.content


@pytest.mark.asyncio
async def test_stat_on_sqlalchemy_does_not_select_content_column(tmp_path):
    """contract structural guard: raw_stat on SqlAlchemy must SELECT only metadata
    columns, NOT the content column. Captures every SQL statement issued
    during a stat() call and asserts none of them reference the content column
    (word-boundary match so 'content_type' does not satisfy 'content')."""
    from sqlalchemy import event

    engine, backend, store = await _make_sqlalchemy_store(tmp_path)
    captured: "list[str]" = []
    try:
        await store.put(AssetPath("/a.txt"), b"payload")

        @event.listens_for(engine.sync_engine, "before_cursor_execute")
        def _capture(conn, cursor, statement, parameters, context, executemany):
            captured.append(statement)

        captured.clear()
        info = await store.stat(AssetPath("/a.txt"))
        assert info is not None

        # Every SELECT issued by stat() must omit the bare content column.
        stat_selects = [s for s in captured if s.upper().startswith("SELECT")]
        assert stat_selects, "stat() should have issued at least one SELECT"
        import re

        content_col = re.compile(r"\bcontent\b", re.IGNORECASE)
        for stmt in stat_selects:
            assert not content_col.search(stmt), (
                f"raw_stat SELECT must not reference the content column, got: {stmt!r}"
            )
    finally:
        await engine.dispose()


# ----------------------------------------------------------------------
# Test 3: Cursor pagination covers every item exactly once (contract)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_cursor_pagination_covers_all_items(backend_factory):
    """contract cursor-pagination guard: list with a small limit must page
    through every matching asset exactly once, via successive cursors,
    terminating with cursor=None. Catches the regression where cursor was
    accepted but ignored (the initial implementation always returned cursor=None after
    truncating to limit, dropping items past the first page)."""
    store = AssetStore(primary=backend_factory())
    # Seed 5 assets under /r/. Sorted by path: /r/1.txt ... /r/5.txt.
    for i in range(5):
        await store.put(AssetPath(f"/r/{i}.txt"), f"payload-{i}".encode())

    seen_paths: "list[str]" = []
    cursor: "str | None" = None
    pages = 0
    while True:
        page = await store.list(
            AssetPath("/r"), depth=Depth.ONE, limit=2, cursor=cursor
        )
        pages += 1
        seen_paths.extend(info.path.value for info in page.items)
        cursor = page.cursor
        if cursor is None:
            break
        if pages > 10:
            pytest.fail("pagination did not terminate (cursor never returned None)")

    # Every item covered, no duplicates, sorted order preserved across pages.
    assert seen_paths == [f"/r/{i}.txt" for i in range(5)], (
        f"pagination must cover every item in sorted order, got {seen_paths}"
    )
    assert len(seen_paths) == len(set(seen_paths)), (
        "pagination must not return duplicates"
    )
    # limit=2 over 5 items -> 3 pages (2 + 2 + 1).
    assert pages == 3, f"expected 3 pages for 5 items at limit=2, got {pages}"


@pytest.mark.asyncio
async def test_list_cursor_none_when_results_fit_one_page(backend_factory):
    """contract sanity: when the result fits in one page (fewer items than limit),
    next_cursor must be None -- callers must not loop forever thinking more
    pages remain."""
    store = AssetStore(primary=backend_factory())
    await store.put(AssetPath("/r/a.txt"), b"a")
    await store.put(AssetPath("/r/b.txt"), b"b")

    page = await store.list(
        AssetPath("/r"), depth=Depth.ONE, limit=100, cursor=None
    )
    assert page.cursor is None
    assert {info.path.value for info in page.items} == {"/r/a.txt", "/r/b.txt"}
