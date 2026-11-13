#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema stamps belong only to explicitly owned complete SQLite schemas."""

import asyncio
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata, provision_database, provision_runtime_database
from linktools.ai.migrate import _database as migration
from linktools.ai.storage import SqlStorageContext, create_sql_storage_context, validate_sql


@pytest.mark.asyncio
async def test_owned_complete_schema_version_skips_reflection_and_bumps_after_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "owned.db"
    with sqlite3.connect(path) as database:
        database.execute(f"PRAGMA application_id={SqlStorageContext.LOCAL_SQLITE_APPLICATION_ID}")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(path)
    statements: list[str] = []
    def capture(connection: object, cursor: object, statement: str, parameters: object, execution_context: object, many: bool) -> None:
        statements.append(statement.lower())
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await asyncio.gather(provision_database(context), provision_database(context))
        async with engine.connect() as connection:
            assert await connection.scalar(text("PRAGMA user_version")) == migration._SQLITE_SCHEMA_VERSION
        statements.clear()
        await provision_database(context)
        assert not any("sqlite_master" in sql or "table_info(" in sql or "index_list(" in sql for sql in statements)
        await validate_sql(engine, build_sql_schema_metadata())
        assert any("table_info(" in sql for sql in statements)
        monkeypatch.setattr(migration, "_SQLITE_SCHEMA_VERSION", migration._SQLITE_SCHEMA_VERSION + 1)
        statements.clear()
        await provision_database(context)
        assert any("table_info(" in sql for sql in statements)
        async with engine.connect() as connection:
            assert await connection.scalar(text("PRAGMA user_version")) == migration._SQLITE_SCHEMA_VERSION
    finally:
        await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", (0, 41))
async def test_shared_and_partial_provision_leave_caller_version_untouched(tmp_path: Path, version: int) -> None:
    path = tmp_path / "shared.db"
    with sqlite3.connect(path) as database:
        database.execute(f"PRAGMA user_version={version}")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_runtime_database(engine)
        await provision_database(engine)
        await provision_database(create_sql_storage_context(engine))
        with sqlite3.connect(path) as database:
            assert database.execute("PRAGMA user_version").fetchone()[0] == version
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_different_owned_version_is_recorded_only_after_valid_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy import Column, Integer, MetaData, Table
    path = tmp_path / "invalid.db"
    with sqlite3.connect(path) as database:
        database.execute(f"PRAGMA application_id={SqlStorageContext.LOCAL_SQLITE_APPLICATION_ID}")
        database.execute("CREATE TABLE invalid (id INTEGER PRIMARY KEY)")
    metadata = MetaData()
    Table("invalid", metadata, Column("id", Integer, primary_key=True), Column("required", Integer))
    monkeypatch.setattr(migration, "build_sql_schema_metadata", lambda: metadata)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(path)
    try:
        with pytest.raises(AIError) as failed:
            await provision_database(context)
        assert failed.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        with sqlite3.connect(path) as database:
            assert database.execute("PRAGMA user_version").fetchone()[0] == 0
        async def future(session: AsyncSession) -> None:
            await session.execute(text("PRAGMA user_version=999"))
        await context.run_mutation(future)
        with pytest.raises(AIError) as future_error:
            await provision_database(context)
        assert future_error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        with sqlite3.connect(path) as database:
            assert database.execute("PRAGMA user_version").fetchone()[0] == 999
        async def repair(session: AsyncSession) -> None:
            await session.execute(text("ALTER TABLE invalid ADD COLUMN required INTEGER"))
        await context.run_mutation(repair)
        await provision_database(context)
        with sqlite3.connect(path) as database:
            assert database.execute("PRAGMA user_version").fetchone()[0] == migration._SQLITE_SCHEMA_VERSION
    finally:
        await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("mysql", "postgresql"))
async def test_server_provision_skips_default_validation_but_explicit_check_remains(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from typing import AsyncIterator
    from sqlalchemy import Column, Integer, MetaData, Table
    from linktools.ai.storage import _database as database
    calls: list[str] = []

    class Connection:
        dialect = SimpleNamespace(name=name)

        async def run_sync(self, callback: object) -> None:
            calls.append("create")

    class Engine:
        dialect = SimpleNamespace(name=name)

        @asynccontextmanager
        async def begin(self) -> AsyncIterator[Connection]:
            yield Connection()

    async def configure(engine: object) -> None:
        calls.append("configure")

    async def validate(engine: object, metadata: object) -> None:
        calls.append("validate")

    monkeypatch.setattr(database, "dialect_for_name", lambda _: SimpleNamespace(configure_engine=configure))
    monkeypatch.setattr(database, "_validate_sql_schema", validate)
    metadata = MetaData()
    Table("example", metadata, Column("id", Integer, primary_key=True))
    engine = Engine()
    await database.provision_sql(engine, metadata)
    assert calls == ["configure", "create"]
    await database.validate_sql(engine, metadata)
    assert calls == ["configure", "create", "configure", "validate"]
