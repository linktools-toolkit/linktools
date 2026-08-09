#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage dialect and raw Asset backend contract checks."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.asset import (
    AssetKey,
    AssetRoot,
    LocalDirectoryAssetBackend,
    SqlAssetBackend,
    StrictConfigReader,
)
from linktools.ai.storage import (
    TABLE_PREFIX,
    MySQLDialect,
    PostgreSQLDialect,
    SQLiteDialect,
    SqlSchemaRegistry,
    resolve_dialect,
)


class _DialectSession:
    def __init__(self, name: str) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=name))
        self.statements = []

    async def execute(self, statement: object) -> None:
        self.statements.append(statement)


class _MappedPathAdapter:
    def to_path(self, key: AssetKey) -> str:
        return f"mapped/{key.kind}/{key.id}.json"

    def from_path(self, path: str) -> "AssetKey | None":
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "mapped" or not parts[2].endswith(".json"):
            return None
        return AssetKey(parts[1], parts[2][:-5])


def test_asset_path_adapter_and_config_are_available() -> None:
    adapter = _MappedPathAdapter()
    key = AssetKey("mcp", "one")
    assert adapter.to_path(key) == "mapped/mcp/one.json"
    assert adapter.from_path("mapped/mcp/one.json") == key
    assert StrictConfigReader({}, context="asset").str_or_bool("missing") is None


@pytest.mark.asyncio
async def test_local_directory_asset_backend_maps_single_files(tmp_path: Path) -> None:
    backend = LocalDirectoryAssetBackend(
        AssetRoot("file:directory", "file", str(tmp_path), "directory"),
        writable=True,
        path_adapter=_MappedPathAdapter(),
    )
    await backend.initialize()
    key = AssetKey("mcp", "one")
    await backend.put(key, b"one")
    assert (tmp_path / "mapped/mcp/one.json").read_bytes() == b"one"
    assert await backend.get(key) == b"one"
    await backend.delete(key)
    assert await backend.get(key) is None


@pytest.mark.asyncio
async def test_sql_dialect_upsert_uses_vendor_statement() -> None:
    from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint
    from sqlalchemy.dialects import mysql, postgresql, sqlite

    table = Table(
        "asset",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("path", String(128), nullable=False),
        Column("value", String(128), nullable=False),
        UniqueConstraint("path"),
    )
    cases = (
        (SQLiteDialect(), sqlite.dialect(), "ON CONFLICT"),
        (PostgreSQLDialect(), postgresql.dialect(), "ON CONFLICT"),
        (MySQLDialect(), mysql.dialect(), "ON DUPLICATE KEY UPDATE"),
    )
    for dialect, compiler_dialect, marker in cases:
        session = _DialectSession(dialect.name)
        assert resolve_dialect(session).name == dialect.name
        await dialect.upsert(
            session,
            table=table,
            values={"path": "sample/one", "value": "one"},
            set_values={"value": "one"},
            index_elements=("path",),
        )
        assert marker in str(session.statements[0].compile(dialect=compiler_dialect))


def test_sql_asset_backend_uses_one_state_table() -> None:
    tables = SqlAssetBackend.register_schema(SqlSchemaRegistry())
    backend = SqlAssetBackend(lambda: None, namespace="test")
    assert backend.root.scheme == "sql"
    assert tables.state.name.startswith(TABLE_PREFIX)


@pytest.mark.asyncio
async def test_sql_dialects_restore_batch_upsert_surface() -> None:
    from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint
    from sqlalchemy.dialects import mysql, postgresql, sqlite

    table = Table(
        "asset_batch",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("path", String(128), nullable=False),
        Column("value", String(128), nullable=False),
        UniqueConstraint("path"),
    )
    cases = (
        (SQLiteDialect(), sqlite.dialect(), "ON CONFLICT"),
        (PostgreSQLDialect(), postgresql.dialect(), "ON CONFLICT"),
        (MySQLDialect(), mysql.dialect(), "ON DUPLICATE KEY UPDATE"),
    )
    for dialect, compiler_dialect, marker in cases:
        session = _DialectSession(dialect.name)
        await dialect.upsert_many(
            session,
            table=table,
            rows=(
                {"path": "sample/one", "value": "one"},
                {"path": "sample/two", "value": "two"},
            ),
            set_columns=("value",),
            index_elements=("path",),
        )
        assert marker in str(session.statements[0].compile(dialect=compiler_dialect))
