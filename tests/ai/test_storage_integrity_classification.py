#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL integrity classification follows driver codes, not diagnostic text."""

import sqlite3
from collections.abc import Iterator

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, text
from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_dbapi
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import SqlStateTransaction, StateTransaction
from linktools.ai.storage import (
    IntegrityViolationKind,
    MySQLDialect,
    PostgreSQLDialect,
    SQLiteDialect,
    SqlAlchemyDialect,
    SqlStorageContext,
    classify_integrity_error_by_message,
)


_DIAGNOSTIC_MARKERS = "unique constraint unique failed 1062 foreign key 1451 check constraint 3819"


@pytest.fixture
def sqlite_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
        "CREATE TABLE entries ("
        "id TEXT PRIMARY KEY, token TEXT UNIQUE, value TEXT NOT NULL, "
        "amount INTEGER CHECK (amount > 0), parent_id INTEGER REFERENCES parent(id));"
        "INSERT INTO entries VALUES ('first', 'first', 'value', 1, NULL);"
    )
    try:
        yield connection
    finally:
        connection.close()


def _assert_classification(
    dialect: SqlAlchemyDialect,
    original: BaseException,
    expected: IntegrityViolationKind,
) -> None:
    assert dialect.classify_integrity_error(original) is expected
    for hide_parameters in (False, True):
        for statement in ("INSERT INTO entries VALUES (?)", f"INSERT /* {_DIAGNOSTIC_MARKERS} */ INTO entries VALUES (?)"):
            error = IntegrityError(
                statement, {"value": _DIAGNOSTIC_MARKERS}, original,
                hide_parameters=hide_parameters,
            )
            assert dialect.classify_integrity_error(error) is expected


@pytest.mark.parametrize("statement,expected", (
    ("INSERT INTO entries VALUES ('first', 'other', 'value', 1, NULL)", IntegrityViolationKind.UNIQUE_CONFLICT),
    ("INSERT INTO entries VALUES ('other', 'first', 'value', 1, NULL)", IntegrityViolationKind.UNIQUE_CONFLICT),
    ("INSERT INTO entries (rowid, id, token, value, amount) VALUES (1, 'other', 'other', 'value', 1)", IntegrityViolationKind.UNIQUE_CONFLICT),
    ("INSERT INTO entries VALUES ('other', 'other', 'value', 1, 7)", IntegrityViolationKind.FOREIGN_KEY),
    ("INSERT INTO entries VALUES ('other', 'other', 'value', 0, NULL)", IntegrityViolationKind.CHECK),
    ("INSERT INTO entries VALUES ('other', 'other', NULL, 1, NULL)", IntegrityViolationKind.UNKNOWN),
), ids=("primary-key", "unique", "rowid", "foreign-key", "check", "not-null"))
def test_sql_integrity_classification_uses_actual_sqlite_constraint_codes(
    sqlite_connection: sqlite3.Connection,
    statement: str,
    expected: IntegrityViolationKind,
) -> None:
    with pytest.raises(sqlite3.IntegrityError) as caught:
        sqlite_connection.execute(statement)
    _assert_classification(SQLiteDialect(), caught.value, expected)


@pytest.mark.parametrize("code,expected", (
    (1062, IntegrityViolationKind.UNIQUE_CONFLICT),
    (1215, IntegrityViolationKind.FOREIGN_KEY),
    (1216, IntegrityViolationKind.FOREIGN_KEY),
    (1217, IntegrityViolationKind.FOREIGN_KEY),
    (1451, IntegrityViolationKind.FOREIGN_KEY),
    (1452, IntegrityViolationKind.FOREIGN_KEY),
    (3819, IntegrityViolationKind.CHECK),
    (1048, IntegrityViolationKind.UNKNOWN),
))
def test_sql_integrity_classification_uses_mysql_driver_errno(
    code: int, expected: IntegrityViolationKind,
) -> None:
    errors = pytest.importorskip("asyncmy.errors")
    error_class = errors.OperationalError if code == 3819 else errors.IntegrityError
    original = error_class(code, _DIAGNOSTIC_MARKERS)
    _assert_classification(MySQLDialect(), original, expected)


@pytest.mark.parametrize("name,expected", (
    ("UniqueViolationError", IntegrityViolationKind.UNIQUE_CONFLICT),
    ("ForeignKeyViolationError", IntegrityViolationKind.FOREIGN_KEY),
    ("CheckViolationError", IntegrityViolationKind.CHECK),
    ("NotNullViolationError", IntegrityViolationKind.UNKNOWN),
    ("IntegrityConstraintViolationError", IntegrityViolationKind.UNKNOWN),
))
def test_sql_integrity_classification_uses_postgresql_driver_sqlstate(
    name: str, expected: IntegrityViolationKind,
) -> None:
    errors = pytest.importorskip("asyncpg.exceptions")
    original = getattr(errors, name)(_DIAGNOSTIC_MARKERS)
    _assert_classification(PostgreSQLDialect(), original, expected)
    adapted = AsyncAdapt_asyncpg_dbapi.IntegrityError(str(original))
    adapted.pgcode = adapted.sqlstate = original.sqlstate
    _assert_classification(PostgreSQLDialect(), adapted, expected)


@pytest.mark.parametrize("dialect", (SQLiteDialect(), MySQLDialect(), PostgreSQLDialect()),
                         ids=("sqlite", "mysql", "postgresql"))
def test_sql_integrity_classification_leaves_text_only_errors_unknown(
    dialect: SqlAlchemyDialect,
) -> None:
    _assert_classification(dialect, RuntimeError(_DIAGNOSTIC_MARKERS), IntegrityViolationKind.UNKNOWN)


def test_sql_integrity_classification_keeps_explicit_message_utility() -> None:
    assert classify_integrity_error_by_message(
        RuntimeError("application conflict"), unique_markers=("application conflict",),
    ) is IntegrityViolationKind.UNIQUE_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (1048, 1452, 3819, None))
async def test_mysql_insert_preserves_non_duplicate_integrity_errors(code: int | None) -> None:
    errors = pytest.importorskip("asyncmy.errors")
    original = errors.IntegrityError(code, _DIAGNOSTIC_MARKERS)
    error = IntegrityError("INSERT", {"token": _DIAGNOSTIC_MARKERS}, original)
    table = Table("entries", MetaData(), Column("id", Integer, primary_key=True), Column("token", String(32)))

    class Session:
        async def execute(self, statement: object) -> None:
            raise error

        async def scalar(self, statement: object) -> int:
            pytest.fail("a non-duplicate integrity error must not enter duplicate readback")

    with pytest.raises(IntegrityError) as caught:
        await MySQLDialect().insert_ignore_conflict(
            Session(), table=table, values={"token": _DIAGNOSTIC_MARKERS}, index_elements=("token",),
        )
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate,custom,expected", (
    (True, False, ErrorCode.STORAGE_CONFLICT),
    (False, False, ErrorCode.INTERNAL_ERROR),
    (False, True, ErrorCode.STORAGE_CONFLICT),
), ids=("unique", "not-null", "custom-classifier"))
async def test_sql_integrity_error_mapping_uses_driver_classification_and_override(
    duplicate: bool, custom: bool, expected: ErrorCode,
) -> None:
    class ApplicationDialect(SQLiteDialect):
        def classify_integrity_error(self, error: BaseException) -> IntegrityViolationKind:
            return IntegrityViolationKind.UNIQUE_CONFLICT

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    context = SqlStorageContext(
        engine, async_sessionmaker(engine), ApplicationDialect() if custom else SQLiteDialect(),
    )
    store = SqlStateStore(engine, context=context)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE ai_integrity_entries (id INTEGER PRIMARY KEY, value TEXT NOT NULL, note TEXT)"))
            await connection.execute(text("INSERT INTO ai_integrity_entries VALUES (1, 'first', '')"))
        await store.initialize()

        async def write(session: AsyncSession) -> None:
            await session.execute(
                text(f"INSERT /* {_DIAGNOSTIC_MARKERS} */ INTO ai_integrity_entries VALUES (:id, :value, :note)"),
                {"id": 1 if duplicate else 2, "value": "next" if duplicate else None, "note": _DIAGNOSTIC_MARKERS},
            )

        async def mutate(transaction: StateTransaction) -> None:
            assert isinstance(transaction, SqlStateTransaction)
            await transaction.run_sql(context, write)

        with pytest.raises(AIError) as caught:
            await store.mutate(mutate)
        assert caught.value.code is expected
        assert isinstance(caught.value.__cause__, IntegrityError)
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT id, value FROM ai_integrity_entries"))).all() == [(1, "first")]
    finally:
        await store.close()
        await context.close()
        await engine.dispose()
