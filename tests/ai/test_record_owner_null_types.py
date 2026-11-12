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
from linktools.ai.persistence.state import record_key

from ._metrics_server_test_helpers import _postgresql_server, _server_database


@pytest.mark.parametrize("kind", ("tool_operation", "agent_run"))
def test_single_typed_owner_nulls_have_explicit_server_types(kind: str) -> None:
    metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
    key = record_key("owner-types", "tenant", "checkpoint", kind, ("owner",))
    rows = record_owner_rows(metadata, keys=(key,))
    for dialect in (postgresql.dialect(), sqlite.dialect(), mysql.dialect()):
        sql = str(select(rows).compile(dialect=dialect, compile_kwargs={"render_postcompile": True}))
        assert "NULL AS payload_bytes" not in sql
        if dialect.name == "postgresql":
            assert "CAST(NULL AS BYTEA) AS payload_bytes" in sql
            if kind == "agent_run":
                assert "CAST(NULL AS BYTEA) AS subject_id" in sql
                assert "CAST(NULL AS TIMESTAMP WITH TIME ZONE) AS lease_expires_at" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("tool_operation", "agent_run"))
async def test_postgresql_typed_owner_content_union_prepares(_postgresql_server, kind: str) -> None:
    with _server_database(_postgresql_server) as (_, command):
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
        engine = create_async_engine(url)
        try:
            await provision_database(engine)
            metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
            query = content_reference_statement(metadata, dialect="postgresql", owner_store_key=b"owner",
                owner_keys=(record_key("owner-types", "tenant", "checkpoint", kind, ("owner",)),),
                content_store_key=b"archive", content_scope_key=archive_scope_key("owner-types", "tenant"),
                content_paths=(("value", "payload", "fields", "arguments_payload", "fields", "content", "$execution_payload"),),
                typed_payloads=True)
            async with engine.connect() as connection:
                # PostgreSQL resolves UNION column types even with no matching
                # rows. This reproduces the actual server-prepare failure.
                assert not (await connection.execute(query)).all()
        finally:
            await engine.dispose()
