#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for restored storage dialects and asset utilities."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.asset import (
    AssetCodec,
    AssetContent,
    AssetContentInfo,
    AssetIndex,
    AssetLoader,
    AssetLoaderSource,
    LocalAssetBackend,
    AssetObjectCache,
    PrefixAssetPathAdapter,
    StrictConfigReader,
    compute_asset_etag,
    parse_json_text,
)
from linktools.ai.asset.sql import SqlAlchemyAssetBackend
from linktools.ai.storage.dialects import (
    MySQLDialect,
    PostgreSQLDialect,
    SQLiteDialect,
    resolve_dialect,
)
from linktools.ai.storage.database import SqlSchemaRegistry
from linktools.ai.storage.names import TABLE_PREFIX, storage_name


@dataclass(frozen=True, slots=True)
class _TextContentStore:
    content: AssetContent

    async def get(self, path: str) -> "AssetContent | None":
        return self.content if path == self.content.info.path else None

    async def list_info(self) -> "tuple[AssetContentInfo, ...]":
        return (self.content.info,)


class _TextCodec(AssetCodec[dict[str, str]]):
    def decode(self, item_id: str, raw: str) -> dict[str, str]:
        value = parse_json_text(raw)
        return {"id": item_id, "value": str(value["value"])}


class _DialectSession:
    def __init__(self, name: str) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=name))
        self.statements = []

    async def execute(self, statement: object) -> None:
        self.statements.append(statement)


def test_asset_loader_index_and_config_are_available() -> None:
    async def run() -> None:
        loader = AssetLoader.from_filesystem(Path("tests/ai/fixtures"))
        source = AssetLoaderSource(loader)
        assert await source.list_ids(".json") == ()

        store = _TextContentStore(
            AssetContent(AssetContentInfo("sample/one.json", "sample", 1, "etag"), b'{"value":"one"}')
        )
        loader = AssetLoader.from_store(store, prefix="sample")
        index = AssetIndex(loader_source := AssetIndex.source_from_loader(loader), _TextCodec(), suffix=".json")
        assert loader_source is not None
        assert await index.list_ids() == ("one",)
        assert await index.get("one") == {"id": "one", "value": "one"}
        assert await index.get("one") == {"id": "one", "value": "one"}

    asyncio.run(run())
    assert StrictConfigReader({}, context="asset").str_or_bool("missing") is None


@pytest.mark.asyncio
async def test_decoded_asset_cache_shares_inflight_work() -> None:
    store = _TextContentStore(
        AssetContent(AssetContentInfo("sample/one.json", "sample", 1, "etag"), b'{"value":"one"}')
    )
    cache = AssetObjectCache(store, _TextCodec(), prefix="sample", suffix=".json")
    values = await asyncio.gather(cache.get("one"), cache.get("one"))
    assert tuple(values) == ({"id": "one", "value": "one"},) * 2
    assert await cache.list_ids() == ("one",)


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


def test_sql_asset_backend_does_not_use_pessimistic_row_locks() -> None:
    source = Path("linktools-ai/src/linktools/ai/asset/sql.py").read_text(encoding="utf-8")
    assert "with_for_update" not in source


def test_sql_asset_backend_keeps_lightweight_asset_surface() -> None:
    backend = SqlAlchemyAssetBackend(lambda: None)
    tables = SqlAlchemyAssetBackend.register_schema(SqlSchemaRegistry())
    assert backend.root.scheme == "sql"
    assert tuple(table.name for table in (tables.root, tables.entry, tables.version, tables.blob)) == (
        storage_name("asset_revision"),
        storage_name("asset_contents"),
        storage_name("asset_changes"),
        storage_name("asset_blobs"),
    )
    assert all(table.name.startswith(TABLE_PREFIX) for table in (tables.root, tables.entry, tables.version, tables.blob))


@pytest.mark.asyncio
async def test_local_content_backend_restores_atomic_path_mapping_and_batch_io(tmp_path: Path) -> None:
    backend = LocalAssetBackend(tmp_path, path_adapter=PrefixAssetPathAdapter({"mcp": "mapped"}))
    await backend.initialize_storage()
    first = AssetContent(
        AssetContentInfo("mcp/one.json", "mcp", 1, compute_asset_etag(b"one")),
        b"one",
    )
    second = AssetContent(
        AssetContentInfo("mcp/two.json", "mcp", 1, compute_asset_etag(b"two")),
        b"two",
    )
    await backend.apply_batch((first, second), ())
    assert (tmp_path / "mapped/one.json").read_bytes() == b"one"
    assert await backend.get_many((first.info.path, second.info.path)) == {
        first.info.path: first,
        second.info.path: second,
    }
    await backend.delete(first.info.path)
    assert await backend.get(first.info.path) is None


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
