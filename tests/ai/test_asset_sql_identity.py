#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact Asset SQL identity retains history without additional key lookups."""

import getpass
from pathlib import Path

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.asset import AssetKey, SqlAssetBackend, build_asset_sql_metadata
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import InMemoryObjectStore, StorageChange, StorageOperation


async def _assert_asset_identity(engine: AsyncEngine) -> None:
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="identity", object_store=InMemoryObjectStore())
    await backend.initialize()
    statements: list[str] = []

    def capture(_connection: object, _cursor: object, sql: str,
                _parameters: object, _context: object, _many: bool) -> None:
        if sql.lstrip().upper().startswith("SELECT"):
            statements.append(sql)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    keys = (AssetKey("resource", "name"), AssetKey("resource", "name "),
            AssetKey("other", "name"), AssetKey("resource", "🪷" * 128))
    try:
        for suffix in (b"old", b"new"):
            changes = tuple(StorageChange(StorageOperation.PUT, key, str(index).encode() + suffix, None)
                            for index, key in enumerate(keys))
            statements.clear()
            result = await backend.apply_batch(changes)
            assert len(statements) <= 2
            assert len(result.results) == len(keys)
        statements.clear()
        assert await backend.get_many(keys) == {key: str(index).encode() + b"new" for index, key in enumerate(keys)}
        assert len(statements) == 1
        versions = await backend.list_versions(keys[0])
        statements.clear()
        assert await backend.get_at_revision(keys[0], versions[0].entry_revision) == b"0old"
        assert len(statements) == 1
        await backend.delete(keys[0])
        await backend.reset(keys[0])
        await backend.put(keys[0], b"recreated")
        assert await backend.get_at_revision(keys[0], versions[0].entry_revision) == b"0old"
        assert len(await backend.list_versions(keys[0])) == 5
        await backend.validate_integrity()

        metadata = build_asset_sql_metadata()
        entries = metadata.tables["ai_asset_entries"]
        history = metadata.tables["ai_asset_changes"]
        async with engine.begin() as connection:
            original = await connection.scalar(select(entries.c.id).where(
                entries.c.namespace == "identity", entries.c.kind == "resource", entries.c.asset_id == b"name",
            ))
            await connection.execute(update(history).where(history.c.entry_id == original).values(entry_id=-1))
        with pytest.raises(AIError) as corrupt:
            await backend.validate_integrity()
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_asset_identity_and_history(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'asset-identity.db'}")
    try:
        await _assert_asset_identity(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_asset_identity_and_history(_server: tuple[str, list[str]]) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url)
    try:
        await _assert_asset_identity(engine)
    finally:
        await engine.dispose()
