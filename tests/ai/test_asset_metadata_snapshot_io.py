#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL Asset metadata binds its revision and entries to one read snapshot."""

from pathlib import Path

import pytest
from sqlalchemy import delete, event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, SqlAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import MetadataLoadMode, StorageChange, StorageOperation, StorageRevision


@pytest.mark.asyncio
@pytest.mark.parametrize("initially_empty", (False, True))
async def test_metadata_revision_and_entries_share_a_snapshot_during_a_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initially_empty: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    reader = SqlAssetBackend(engine, namespace="snapshot")
    writer = SqlAssetBackend(engine, namespace="snapshot")
    await reader.initialize()
    await writer.initialize()
    key = AssetKey("test", "concurrent-write")
    if not initially_empty:
        await writer.put(key, b"old")
    before = await writer.head_revision()
    context_type = type(reader._context)
    run_statement = context_type.run_statement
    wrote = False

    async def commit_after_first_read(context, statement, **kwargs):
        nonlocal wrote
        result = await run_statement(context, statement, **kwargs)
        if context is reader._context and not wrote:
            wrote = True
            await writer.put(key, b"new")
        return result

    monkeypatch.setattr(context_type, "run_statement", commit_after_first_read)
    try:
        loaded = await reader.load_metadata(None)
        assert wrote
        assert loaded.store_revision == before
        if initially_empty:
            assert loaded.changes == ()
        else:
            assert len(loaded.changes) == 1
            assert loaded.changes[0].info.store_revision == loaded.store_revision
        assert (await writer.head_revision()) != loaded.store_revision
    finally:
        await reader.close()
        await writer.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("populated", (False, True))
async def test_metadata_unchanged_revision_avoids_returning_entry_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, populated: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="unchanged")
    await backend.initialize()
    if populated:
        for index in range(3):
            await backend.put(AssetKey("test", str(index)), b"content")
    revision = await backend.head_revision()
    statements = []
    returned_payloads = []
    context_type = type(backend._context)
    run_statement = context_type.run_statement

    def capture(_conn, _cursor, sql, _parameters, _context, _many):
        if sql.lstrip().upper().startswith("SELECT"):
            statements.append(sql)

    async def capture_rows(context, statement, **kwargs):
        result = await run_statement(context, statement, **kwargs)
        if context is backend._context and hasattr(result, "freeze"):
            frozen = result.freeze()
            returned_payloads.extend(
                row["payload_json"] for row in frozen().mappings()
                if "payload_json" in row and row["payload_json"] is not None
            )
            return frozen()
        return result

    monkeypatch.setattr(context_type, "run_statement", capture_rows)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        loaded = await backend.load_metadata(revision)
        assert loaded.mode is MetadataLoadMode.PATCH
        assert loaded.store_revision == revision
        assert loaded.changes == ()
        assert len(statements) == 1
        assert returned_payloads == []
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await backend.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_metadata_replacement_is_one_namespace_scoped_read(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="selected")
    other = SqlAssetBackend(engine, namespace="other")
    await backend.initialize()
    await other.initialize()
    key = AssetKey("test", "visible")
    await backend.put(key, b"selected")
    await other.put(AssetKey("test", "foreign"), b"other")
    statements = []

    def capture(_conn, _cursor, sql, _parameters, _context, _many):
        if sql.lstrip().upper().startswith("SELECT"):
            statements.append(sql)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        loaded = await backend.load_metadata(StorageRevision("0"))
        assert loaded.mode is MetadataLoadMode.REPLACE
        assert loaded.store_revision == StorageRevision("1")
        assert tuple(change.key for change in loaded.changes) == (key,)
        assert len(statements) == 1
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await backend.close()
        await other.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_unchanged_sqlite_metadata_work_does_not_scale_with_entry_count(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    steps = 0
    counting = False

    def progress():
        nonlocal steps
        if counting:
            steps += 1

    def configure(connection, _record):
        connection.run_async(lambda raw: raw.set_progress_handler(progress, 1))

    event.listen(engine.sync_engine, "connect", configure)
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="bounded-head")
    await backend.initialize()

    async def read_steps():
        nonlocal steps, counting
        revision = await backend.head_revision()
        steps, counting = 0, True
        try:
            loaded = await backend.load_metadata(revision)
        finally:
            counting = False
        assert loaded.mode is MetadataLoadMode.PATCH and loaded.changes == ()
        return steps

    try:
        small = await read_steps()
        await backend.apply_batch(tuple(
            StorageChange(StorageOperation.PUT, AssetKey("test", str(index)), b"content", None)
            for index in range(128)
        ))
        large = await read_steps()
        assert small > 0
        assert large <= small * 3, "an unchanged head must not scan every metadata entry"
    finally:
        await backend.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_metadata_missing_namespace_head_is_an_integrity_error(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="missing-head")
    await backend.initialize()
    try:
        async with engine.begin() as connection:
            await connection.execute(delete(backend._metadata.tables["ai_asset_heads"]))
        with pytest.raises(AIError) as failed:
            await backend.load_metadata(None)
        assert failed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await backend.close()
        await engine.dispose()
