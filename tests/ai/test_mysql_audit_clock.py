#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL audit writes use the same UTC database clock as leases."""

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.storage import MySQLDialect, sql_audit_columns, sql_audit_now
from linktools.ai.storage._dialects import column_type_matches


@pytest.mark.parametrize("dialect,expected", [
    (mysql.dialect(), "UTC_TIMESTAMP(6)"),
    (sqlite.dialect(), "CURRENT_TIMESTAMP"),
    (postgresql.dialect(), "CURRENT_TIMESTAMP"),
])
def test_audit_expression_uses_vendor_clock_without_session_settings(dialect: Any, expected: str) -> None:
    assert str(sql_audit_now().compile(dialect=dialect)) == expected


def test_canonical_audit_insert_defaults_do_not_change_server_schema_or_update_guards() -> None:
    columns = sql_audit_columns()
    for column in columns:
        assert str(column.default.arg.compile(dialect=mysql.dialect())) == "UTC_TIMESTAMP(6)"
        assert column.onupdate is None
    assert str(columns[0].server_default.arg.compile(dialect=mysql.dialect())) == (
        "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
    )
    assert str(columns[1].server_default.arg.compile(dialect=mysql.dialect())) == "CURRENT_TIMESTAMP"


class _Result:
    rowcount = 1
    lastrowid = 0

    def all(self) -> list[tuple[str, int]]:
        return [("stream-a", 1)]


class _Session:
    def __init__(self) -> None:
        self.statements: list[Any] = []
        self.parameters: list[Any] = []

    async def execute(self, statement: Any, parameters: Any = None) -> _Result:
        self.statements.append(statement)
        self.parameters.append(parameters)
        return _Result()

    async def scalar(self, statement: Any) -> int:
        self.statements.append(statement)
        return 1


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("identity", [False, True])
async def test_mysql_counter_insert_and_matching_update_write_utc(batch: bool, identity: bool) -> None:
    table = Table("ai_clock_counter", MetaData(),
        Column("store_key", String(64)), Column("stream_key", String(64)),
        Column("value", Integer), *sql_audit_columns())
    session = _Session()
    dialect = MySQLDialect()
    values = {"store_key": "store-a", "stream_key": "stream-a"}
    options = {"table": table, "column": "value", "index_elements": ("store_key",),
               "identity_columns": ("stream_key",) if identity else ()}
    if batch:
        await dialect.upsert_increment_many(
            session, rows=[{**values, "value": 1}], returning_key="stream_key", **options,
        )
    else:
        await dialect.upsert_increment(session, values=values, **options)
    statement = session.statements[0]
    if batch:
        statement = statement.values(session.parameters[0][0])
    sql = str(statement.compile(dialect=mysql.dialect()))
    initial, changed = sql.split(" ON DUPLICATE KEY UPDATE ", 1)
    assert initial.count("UTC_TIMESTAMP(6)") == 2
    assert "UTC_TIMESTAMP(6)" in changed
    assert "CURRENT_TIMESTAMP" not in sql
    assert "created_at =" not in changed
    if identity:
        assert "ELSE ai_clock_counter.updated_at END" in changed
    assert not any("SET time_zone" in str(statement) for statement in session.statements)


@pytest.mark.asyncio
async def test_mysql_batch_counter_preserves_explicit_insert_audit_values() -> None:
    table = Table("ai_clock_counter", MetaData(),
        Column("store_key", String(64)), Column("stream_key", String(64)),
        Column("value", Integer), *sql_audit_columns())
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    row = {"store_key": "store-a", "stream_key": "stream-a", "value": 1,
           "created_at": timestamp, "updated_at": timestamp}
    session = _Session()
    await MySQLDialect().upsert_increment_many(
        session, table=table, rows=[row], column="value",
        index_elements=("store_key",), returning_key="stream_key",
    )
    assert session.parameters[0] == [row]
    compiled = session.statements[0].values(row).compile(dialect=mysql.dialect())
    initial, changed = str(compiled).split(" ON DUPLICATE KEY UPDATE ", 1)
    assert "UTC_TIMESTAMP" not in initial
    assert compiled.params["created_at"] == timestamp
    assert compiled.params["updated_at"] == timestamp
    assert "UTC_TIMESTAMP(6)" in changed


@pytest.mark.parametrize("actual,compatible", [
    (mysql.DATETIME(fsp=6), True),
    (mysql.DATETIME(), False),
    (mysql.DATETIME(fsp=3), False),
    (mysql.TIMESTAMP(fsp=6), False),
    (DateTime(), False),
])
def test_mysql_lease_requires_canonical_datetime_precision(actual: Any, compatible: bool) -> None:
    expected = Column("lease_expires_at", DateTime(timezone=True).with_variant(mysql.DATETIME(fsp=6), "mysql"))
    connection = SimpleNamespace(dialect=mysql.dialect())
    assert column_type_matches(connection, expected=expected, actual={"type": actual}) is compatible


def test_mysql_lease_check_does_not_tighten_unrelated_timestamp_columns() -> None:
    connection = SimpleNamespace(dialect=mysql.dialect())
    assert column_type_matches(
        connection, expected=Column("created_at", mysql.DATETIME()),
        actual={"type": mysql.TIMESTAMP()},
    )
