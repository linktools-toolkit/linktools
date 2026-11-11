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


@pytest.mark.asyncio
async def test_asset_locator_collision_cannot_read_or_mutate_another_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.asset import _sql
    from linktools.ai.storage import StorageEntryRevision

    monkeypatch.setattr(_sql, "_asset_locator", lambda _key: "a" * 64)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'collision.db'}")
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="collision", object_store=InMemoryObjectStore())
    await backend.initialize()
    first, second = AssetKey("resource", "path/first"), AssetKey("resource", "path/second")
    try:
        await backend.put(first, b"same")
        head = await backend.head_revision()
        for operation in (
            lambda: backend.stat(second),
            lambda: backend.get_many((first, second)),
            lambda: backend.put(second, b"same"),
            lambda: backend.delete(second),
            lambda: backend.reset(second),
            lambda: backend.list_versions(second),
            lambda: backend.get_at_revision(second, StorageEntryRevision(1)),
        ):
            with pytest.raises(AIError) as collision:
                await operation()
            assert collision.value.code is ErrorCode.STORAGE_CONFLICT
        assert await backend.head_revision() == head
        assert await backend.get(first) == b"same"
        assert len(await backend.list_versions(first)) == 1
    finally:
        await backend.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_asset_batch_locator_collision_rolls_back_all_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.asset import _sql

    monkeypatch.setattr(_sql, "_asset_locator", lambda _key: "a" * 64)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batch-collision.db'}")
    await provision_asset_database(engine)
    backend = SqlAssetBackend(engine, namespace="collision", object_store=InMemoryObjectStore())
    await backend.initialize()
    keys = (AssetKey("resource", "first"), AssetKey("resource", "second"))
    try:
        head = await backend.head_revision()
        with pytest.raises(AIError) as collision:
            await backend.apply_batch(tuple(
                StorageChange(StorageOperation.PUT, key, b"same", None) for key in keys
            ), idempotency_key="batch", request_digest="a" * 64)
        assert collision.value.code is ErrorCode.STORAGE_CONFLICT
        assert await backend.head_revision() == head
        assert await backend.batch_result("batch") is None
        assert await backend.get_many(keys) == {}
    finally:
        await backend.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_asset_locator_collision_preserves_only_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from linktools.ai.asset import _sql

    monkeypatch.setattr(_sql, "_asset_locator", lambda _key: "a" * 64)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'race-collision.db'}")
    await provision_asset_database(engine)
    backends = [SqlAssetBackend(engine, namespace="collision", object_store=InMemoryObjectStore())
                for _ in range(2)]
    keys = (AssetKey("resource", "first"), AssetKey("resource", "second"))
    try:
        for backend in backends:
            await backend.initialize()
        results = await asyncio.gather(*(
            backend.put(key, b"same") for backend, key in zip(backends, keys)
        ), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        assert len(failures) == 1
        assert isinstance(failures[0], AIError)
        assert failures[0].code is ErrorCode.STORAGE_CONFLICT
        winner = next(index for index, result in enumerate(results) if not isinstance(result, BaseException))
        assert await backends[winner].get(keys[winner]) == b"same"
        assert len(await backends[winner].list_versions(keys[winner])) == 1
    finally:
        for backend in backends:
            await backend.close()
        await engine.dispose()
