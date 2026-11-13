#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed owner omissions retain server types across derived tables and UNIONs."""

import getpass

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence._archive import archive_scope_key
from linktools.ai.persistence._content_reference_sql import content_reference_statement
from linktools.ai.persistence._record_owner import record_owner_rows
from linktools.ai.persistence.schema import build_runtime_sql_metadata
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.state import RecordQuery, record_key
from linktools.ai.persistence._sql import _record_query_statement

from ._metrics_server_test_helpers import _mysql_server, _postgresql_server, _server_database


@pytest.mark.parametrize("kind", ("tool_operation", "agent_run"))
def test_single_typed_owner_nulls_have_explicit_server_types(kind: str) -> None:
    metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
    key = record_key("owner-types", "tenant", "checkpoint", kind, ("owner",))
    rows = record_owner_rows(metadata, keys=(key,))
    for dialect in (postgresql.dialect(), sqlite.dialect(), mysql.dialect()):
        sql = str(select(rows).compile(dialect=dialect, compile_kwargs={"render_postcompile": True}))
        assert "NULL AS payload_bytes" not in sql
        if dialect.name == "mysql" and kind == "agent_run":
            assert "CAST(NULL AS CHAR(64)) AS state" in sql
            assert "CAST(NULL AS CHAR) AS lease_owner" in sql
            assert "COLLATE utf8mb4_bin)" not in sql
        if dialect.name == "postgresql":
            assert "CAST(NULL AS BYTEA) AS payload_bytes" in sql
            if kind == "agent_run":
                assert "CAST(NULL AS BYTEA) AS subject_id" in sql
                assert "CAST(NULL AS TIMESTAMP WITH TIME ZONE) AS lease_expires_at" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("tool_operation", "agent_run"))
@pytest.mark.parametrize("backend", ("postgresql", "mysql"))
@pytest.mark.parametrize("scoped", (False, True))
async def test_server_typed_owner_content_union_prepares(
    request: pytest.FixtureRequest, kind: str, backend: str, scoped: bool,
) -> None:
    server = request.getfixturevalue(f"_{backend}_server")
    with _server_database(server) as (_, command):
        if backend == "postgresql":
            url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                             query={"host": command[command.index("-h") + 1]})
        else:
            socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
            url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                             query={"unix_socket": socket, "charset": "utf8mb4"})
        engine = create_async_engine(url)
        try:
            await provision_database(engine)
            metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
            query = content_reference_statement(metadata, dialect=backend, owner_store_key=b"owner",
                owner_keys=(record_key("owner-types", "tenant", "checkpoint", kind, ("owner",)),),
                content_store_key=b"archive", content_scope_key=archive_scope_key("owner-types", "tenant"),
                content_paths=(("value", "payload", "fields", "arguments_payload", "fields", "content", "$execution_payload"),),
                typed_payloads=True,
                owner_statement=(_record_query_statement(metadata, b"owner",
                    RecordQuery(kind=kind, parent_key=b"parent", limit=2)) if scoped else None))
            async with engine.connect() as connection:
                # Both servers parse casts and resolve UNION column types even
                # with no rows: exercise the actual owner/content JOIN.
                assert not (await connection.execute(query)).all()
        finally:
            await engine.dispose()


@pytest.mark.parametrize("backend", ("postgresql", "mysql", "sqlite"))
def test_scoped_owner_content_join_compiles_with_inner_limit(backend: str) -> None:
    metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
    owner = _record_query_statement(metadata, b"owner",
        RecordQuery(kind="tool_operation", parent_key=b"parent", limit=2))
    query = content_reference_statement(metadata, dialect=backend, owner_store_key=b"owner", owner_keys=(),
        content_store_key=b"archive", content_scope_key=archive_scope_key("owner-types", "tenant"),
        content_paths=(("value", "payload", "fields", "arguments_payload", "fields", "content", "$execution_payload"),),
        typed_payloads=True, owner_statement=owner)
    dialect = {"postgresql": postgresql, "mysql": mysql, "sqlite": sqlite}[backend].dialect()
    sql = str(query.compile(dialect=dialect, compile_kwargs={"render_postcompile": True}))
    assert "LIMIT" in sql and "content_owners" in sql and "parent_key" in sql
    assert "COLLATE utf8mb4_bin)" not in sql
