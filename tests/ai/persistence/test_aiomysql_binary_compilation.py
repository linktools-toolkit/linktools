#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for SDK binary SQL binding without a MySQL server."""

import re

import pytest
from sqlalchemy import bindparam, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.mysql import aiomysql, asyncmy
from sqlalchemy.schema import CreateIndex, CreateTable

from linktools.ai.migrate import build_sql_schema_metadata


def _statement_and_bindings(statement, dialect):
    compiled = statement.compile(dialect=dialect)
    expanded = compiled.construct_expanded_state()
    processors = {**compiled._bind_processors, **expanded.processors}
    bindings = {
        key: processors[key](value) if key in processors else value
        for key, value in expanded.parameters.items()
    }
    return expanded.statement, bindings


def test_aiomysql_hexes_binary_values_but_not_text_or_columns():
    table = build_sql_schema_metadata().tables["ai_state_records"]
    statement = select(table.c.key).where(
        table.c.key.in_((b"\x00\xff", b"00ff")),
        table.c.key_locator == "00ff",
        table.c.parent_key == table.c.key,
    )
    sql, values = _statement_and_bindings(statement, aiomysql.dialect())
    assert re.search(r"\.\x60?key\x60? IN \(UNHEX\(%s\), UNHEX\(%s\)\)", sql, re.I)
    assert re.search(r"\.key_locator = %s", sql)
    assert "parent_key = ai_state_records.`key`" in sql
    assert list(values.values()) == ["00ff", "00ff", "30306666"]
    for dialect in (sqlite.dialect(), postgresql.dialect(), asyncmy.dialect()):
        other_sql, other_values = _statement_and_bindings(statement, dialect)
        assert "unhex" not in other_sql.lower()
        assert list(other_values.values()) == ["00ff", b"\x00\xff", b"00ff"]


@pytest.mark.parametrize(("payload", "wire"), [(None, None), (b"", ""), (b"\x00\xff", "00ff")])
def test_aiomysql_preserves_null_and_empty_binary_binds(payload, wire):
    table = build_sql_schema_metadata().tables["ai_state_records"]
    statement = update(table).where(table.c.key == b"\xfe").values(
        payload_bytes=bindparam("payload", value=payload, type_=table.c.payload_bytes.type),
    )
    sql, values = _statement_and_bindings(statement, aiomysql.dialect())
    assert re.search(r"payload_bytes=UNHEX\(%s\)", sql, re.I)
    assert values["payload"] == wire
    assert "fe" in values.values()


def test_aiomysql_and_asyncmy_share_canonical_ddl():
    metadata = build_sql_schema_metadata()
    for table in metadata.tables.values():
        for ddl in (CreateTable(table), *(CreateIndex(index) for index in table.indexes)):
            assert str(ddl.compile(dialect=aiomysql.dialect())) == str(ddl.compile(dialect=asyncmy.dialect()))


@pytest.mark.asyncio
@pytest.mark.parametrize("no_backslash_escapes", [False, True])
async def test_actual_aiomysql_cursor_receives_one_bounded_insert_per_batch(no_backslash_escapes):
    from types import SimpleNamespace

    pytest.importorskip("aiomysql")
    from aiomysql.connection import Connection
    from aiomysql.cursors import Cursor
    from pymysql.constants import SERVER_STATUS
    from sqlalchemy import insert

    from linktools.ai.storage import sql_insert_many

    dialect = aiomysql.dialect(dbapi=aiomysql.dialect.import_dbapi())
    connection = Connection(charset="utf8mb4")
    connection.server_status = SERVER_STATUS.SERVER_STATUS_NO_BACKSLASH_ESCAPES if no_backslash_escapes else 0
    cursor = Cursor(connection)
    table = build_sql_schema_metadata().tables["ai_object_chunks"]
    packets = []

    class Session:
        def get_bind(self):
            return SimpleNamespace(dialect=dialect)

        async def execute(self, statement, parameters=None):
            # A DBAPI executemany of UNHEX VALUES would execute once per row.
            assert parameters is None
            compiled = statement.compile(dialect=dialect)
            sql, bindings = _statement_and_bindings(statement, dialect)
            wire = cursor.mogrify(sql, tuple(bindings[key] for key in compiled.positiontup)).encode("utf-8")
            assert len(wire) <= cursor.max_stmt_length
            packets.append(wire)
            return SimpleNamespace(rowcount=wire.count(b"UNHEX("))

    payloads = [b"", bytes(range(256)), b"b" * 300_000, b"z" * 300_000]
    rows = [{"object_id": 1, "chunk_index": i, "content": value} for i, value in enumerate(payloads)]
    assert await sql_insert_many(Session(), insert(table), rows) == len(rows)
    assert len(packets) == 2
    for value in payloads:
        literal = b"UNHEX('" + value.hex().encode("ascii") + b"')"
        assert sum(packet.count(literal) for packet in packets) == 1
