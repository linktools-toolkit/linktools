#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preserve conflict statements and execution semantics across SQL dialects."""

import getpass
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from linktools.ai.storage import InsertResult, PostgreSQLDialect, SQLiteDialect
from sqlalchemy import (
    Column, DateTime, Integer, MetaData, String, Table, UniqueConstraint,
    event, func, select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Dialect, URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.sql import ClauseElement


_OPERATIONS = (
    "insert_ignore_conflict", "insert_ignore_conflict_many", "upsert",
    "upsert_increment", "upsert_many", "upsert_increment_many",
)
_OLD_TIME = datetime(2000, 1, 1)


def _table(audit: bool) -> Table:
    return Table(
        "ai_dialect_entries", MetaData(),
        Column("id", Integer, primary_key=True),
        Column("namespace", String(32), nullable=False),
        Column("key", String(32), nullable=False),
        Column("value", String(32), nullable=False),
        Column("revision", Integer, nullable=False),
        *([Column("updated_at", DateTime, server_default=func.current_timestamp())]
          if audit else []),
        UniqueConstraint("namespace", "key"),
    )


def _rows(audit: bool) -> list[dict[str, str | int | datetime]]:
    return [
        {
            "namespace": "local", "key": key, "value": "incoming",
            "revision": revision, **({"updated_at": _OLD_TIME} if audit else {}),
        }
        for key, revision in (("alpha", 7), ("beta", 11))
    ]


class _Result:
    def first(self) -> tuple[str] | None:
        return ("13",)

    def scalar_one(self) -> str:
        return "17"

    def all(self) -> list[tuple[str | int, str | int]]:
        return [(123, "17"), ("beta", 11)]


class _Session:
    def __init__(self) -> None:
        self.statements: list[ClauseElement] = []

    async def execute(self, statement: ClauseElement) -> _Result:
        self.statements.append(statement)
        return _Result()


async def _execute(
    dialect: SQLiteDialect,
    session: AsyncSession | _Session,
    table: Table,
    operation: str,
    rows: list[dict[str, str | int | datetime]],
) -> InsertResult | int | dict[str, int] | None:
    indexes = ("namespace", "key")
    if operation == "insert_ignore_conflict":
        return await dialect.insert_ignore_conflict(
            session, table=table, values=rows[0], index_elements=indexes,
        )
    if operation == "insert_ignore_conflict_many":
        return await dialect.insert_ignore_conflict_many(
            session, table=table, rows=rows, index_elements=indexes,
        )
    if operation == "upsert":
        return await dialect.upsert(
            session, table=table, values=rows[0], set_values={"value": "explicit"},
            index_elements=indexes,
        )
    if operation == "upsert_increment":
        return await dialect.upsert_increment(
            session, table=table, values=rows[0], column="revision", step=3,
            index_elements=indexes,
        )
    if operation == "upsert_many":
        return await dialect.upsert_many(
            session, table=table, rows=rows, set_columns=("value",),
            index_elements=indexes,
        )
    assert operation == "upsert_increment_many"
    return dict(await dialect.upsert_increment_many(
        session, table=table, rows=rows, column="revision", returning_key="key",
        index_elements=indexes,
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect,compiler", (
    (SQLiteDialect(), sqlite.dialect(paramstyle="named")),
    (PostgreSQLDialect(), postgresql.dialect(paramstyle="named")),
), ids=("sqlite", "postgresql"))
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("audit", (False, True), ids=("no-audit", "audit"))
async def test_conflict_statements_preserve_vendor_sql_and_one_roundtrip(
    dialect: SQLiteDialect, compiler: Dialect, operation: str, audit: bool,
) -> None:
    table, rows, session = _table(audit), _rows(audit), _Session()
    original = [dict(row) for row in rows]
    result = await _execute(dialect, session, table, operation, rows)
    assert rows == original
    assert len(session.statements) == 1
    compiled = session.statements[0].compile(dialect=compiler)
    sql = str(compiled)
    assert 'ON CONFLICT (namespace, key)' in sql.replace('"', '')
    assert compiled.dialect.name == dialect.name
    if operation.startswith("insert_ignore_conflict"):
        assert "DO NOTHING" in sql
    elif operation == "upsert":
        assert "DO UPDATE SET value = :param_1" in sql
        assert compiled.params["param_1"] == "explicit"
    elif operation == "upsert_many":
        assert "DO UPDATE SET value = excluded.value" in sql
    else:
        assert "revision = (ai_dialect_entries.revision + " in sql
        assert ("updated_at = CURRENT_TIMESTAMP" in sql) is audit
        if operation.endswith("_many"):
            assert "revision + excluded.revision" in sql
        else:
            assert compiled.params["revision"] == 3
            assert compiled.params["revision_1"] == 3
    if operation == "insert_ignore_conflict":
        returning = "ai_dialect_entries.id" if dialect.name == "postgresql" else "id"
        assert sql.endswith(f"RETURNING {returning}")
        assert result == InsertResult(True, 13)
    elif operation == "upsert_increment":
        returning = "ai_dialect_entries.revision" if dialect.name == "postgresql" else "revision"
        assert sql.endswith(f"RETURNING {returning}")
        assert result == 17
    elif operation == "upsert_increment_many":
        returning = (
            "ai_dialect_entries.key, ai_dialect_entries.revision"
            if dialect.name == "postgresql" else '"key", revision'
        )
        assert sql.endswith(f"RETURNING {returning}")
        assert result == {"123": 17, "beta": 11}
    else:
        assert result is None
    if operation.endswith("_many"):
        assert compiled.params["revision_m0"] == 7
        assert compiled.params["revision_m1"] == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dialect", (SQLiteDialect(), PostgreSQLDialect()), ids=("sqlite", "postgresql"),
)
async def test_ignored_conflict_returns_no_inserted_id(dialect: SQLiteDialect) -> None:
    class ConflictResult(_Result):
        def first(self) -> None:
            return None

    class ConflictSession(_Session):
        async def execute(self, statement: ClauseElement) -> ConflictResult:
            self.statements.append(statement)
            return ConflictResult()

    result = await _execute(
        dialect, ConflictSession(), _table(False), "insert_ignore_conflict", _rows(False),
    )
    assert result == InsertResult(False, None)


async def _assert_conflict_execution(
    engine: AsyncEngine, dialect: SQLiteDialect, operation: str, audit: bool,
) -> None:
    table, rows = _table(audit), _rows(audit)
    statements: list[str] = []
    event.listen(
        engine.sync_engine, "before_cursor_execute",
        lambda conn, cursor, sql, params, context, many: statements.append(sql),
    )
    try:
        async with engine.begin() as connection:
            await connection.run_sync(table.metadata.create_all)
            await connection.execute(table.insert(), [
                {**rows[0], "value": "old", "revision": 10},
                {**rows[0], "namespace": "other", "value": "other", "revision": 100},
            ])
        async with AsyncSession(engine) as session:
            statements.clear()
            if operation.endswith("_many"):
                result = await _execute(dialect, session, table, operation, rows)
                assert len(statements) == 1
                if operation == "upsert_increment_many":
                    assert result == {"alpha": 17, "beta": 11}
            else:
                existing = await _execute(
                    dialect, session, table, operation, rows,
                )
                inserted = await _execute(
                    dialect, session, table, operation, rows[1:],
                )
                assert len(statements) == 2
                if operation == "insert_ignore_conflict":
                    assert existing == InsertResult(False, None)
                    assert isinstance(inserted, InsertResult)
                    assert inserted.inserted and inserted.row_id is not None
                elif operation == "upsert_increment":
                    assert (existing, inserted) == (13, 3)
            await session.commit()
            stored = {
                (row["namespace"], row["key"]): row
                for row in (await session.execute(select(table))).mappings()
            }
        assert len(stored) == 3
        alpha, beta, other = (
            stored[key]
            for key in (("local", "alpha"), ("local", "beta"), ("other", "alpha"))
        )
        expected_alpha = {
            "upsert": ("explicit", 10),
            "upsert_many": ("incoming", 10),
            "upsert_increment": ("old", 13),
            "upsert_increment_many": ("old", 17),
        }.get(operation, ("old", 10))
        assert (alpha["value"], alpha["revision"]) == expected_alpha
        assert (beta["value"], beta["revision"]) == (
            "incoming", 3 if operation == "upsert_increment" else 11,
        )
        assert (other["value"], other["revision"]) == ("other", 100)
        if audit:
            assert (alpha["updated_at"] > _OLD_TIME) is ("increment" in operation)
            assert beta["updated_at"] == other["updated_at"] == _OLD_TIME
    finally:
        try:
            async with engine.begin() as connection:
                await connection.run_sync(table.metadata.drop_all)
        finally:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("audit", (False, True), ids=("no-audit", "audit"))
async def test_sqlite_conflict_execution_preserves_existing_and_new_rows(
    operation: str, audit: bool,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await _assert_conflict_execution(engine, SQLiteDialect(), operation, audit)


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("postgresql",), indirect=True)
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("audit", (False, True), ids=("no-audit", "audit"))
async def test_postgresql_conflict_execution_preserves_existing_and_new_rows(
    _server: tuple[str, list[str]], operation: str, audit: bool,
) -> None:
    _, command = _server
    url = URL.create(
        "postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
        query={"host": command[command.index("-h") + 1]},
    )
    engine = create_async_engine(url)
    await _assert_conflict_execution(engine, PostgreSQLDialect(), operation, audit)


def test_optional_sql_imports_and_empty_batches_are_lazy(tmp_path: Path) -> None:
    script = """
import asyncio
import importlib.abc
import sys

class NoSqlAlchemy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sqlalchemy" or fullname.startswith("sqlalchemy."):
            raise AssertionError("Unexpected SQLAlchemy import: " + fullname)

sys.meta_path.insert(0, NoSqlAlchemy())
from linktools.ai.storage import MySQLDialect, PostgreSQLDialect, SQLiteDialect

async def check():
    for dialect in (SQLiteDialect(), PostgreSQLDialect(), MySQLDialect()):
        assert await dialect.insert_ignore_conflict_many(
            None, table=None, rows=[], index_elements=("key",),
        ) is None
        assert await dialect.upsert_many(
            None, table=None, rows=[], set_columns=("value",), index_elements=("key",),
        ) is None
        assert await dialect.upsert_increment_many(
            None, table=None, rows=[], column="revision", index_elements=("key",), returning_key="key",
        ) == {}
    assert not any(name == "sqlalchemy" or name.startswith("sqlalchemy.") for name in sys.modules)

asyncio.run(check())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
        env={**os.environ, "LINKTOOLS_PATH": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr
