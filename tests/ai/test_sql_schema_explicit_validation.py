#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime SQL initialization does not perform deployment schema diagnostics."""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.asset import AssetKey, AssetStore, SqlAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import build_sql_schema_metadata, provision_database
from linktools.ai.observe._sql import SqlMetricStore
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.runtime.state.schema import build_runtime_sql_metadata
from linktools.ai.storage import StorageOverlay, create_sql_storage_context, read_object, validate_sql


@pytest.fixture
def schema_reads() -> Iterator[list[str]]:
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        lowered = statement.lower()
        if any(name in lowered for name in (
            "table_info(", "table_xinfo(", "index_list(", "index_info(",
            "foreign_key_list(", "sqlite_master", "sqlite_temp_master",
        )):
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
async def test_sql_adapters_skip_schema_reflection_but_keep_data_operations(
    tmp_path: Path, schema_reads: list[str],
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'shared.db'}")
    runtime = RuntimeState(RuntimeStorage.sql(engine))
    state = SqlStateStore(engine)
    backend = SqlAssetBackend(engine, namespace="explicit-schema")
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    try:
        await provision_database(engine)
        schema_reads.clear()
        await runtime.initialize(namespace="runtime", tenant_id="tenant")
        await state.initialize()
        await assets.initialize()
        key = AssetKey("resource", "example")
        await assets.put(key, b"first")
        version = (await assets.resolve_versions((key,)))[0]
        await assets.put(key, b"second")
        assert await assets.get(key) == b"second"
        assert await assets.read_versions((version,)) == (b"first",)
        assert await SqlMetricStore(engine).get_definition("metrics", "missing", 1) is None
        assert schema_reads == []

        # The public deployment API still inspects the complete deployed schema.
        await validate_sql(engine, build_sql_schema_metadata())
        assert schema_reads
    finally:
        await assets.close()
        await state.close()
        await runtime.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_external_runtime_initialization_neither_checks_nor_creates_tables(
    tmp_path: Path, schema_reads: list[str],
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'external.db'}")
    runtime = RuntimeState(RuntimeStorage.sql(engine))
    try:
        await runtime.initialize(namespace="external", tenant_id="tenant")
        assert schema_reads == []
        async with engine.connect() as connection:
            tables = await connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type = 'table'")
            assert tables.all() == []
        with pytest.raises(AIError) as error:
            await validate_sql(engine, build_runtime_sql_metadata(frozenset({RuntimeDomain.CONVERSATION})))
        assert error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
    finally:
        await runtime.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_local_sqlite_creation_and_reopen_keep_objects_without_reinspection(
    tmp_path: Path, schema_reads: list[str],
) -> None:
    path = tmp_path / "local.db"
    state = RuntimeState(RuntimeStorage.sqlite(path))
    payload = b"retained object"
    digest = hashlib.sha256(payload).hexdigest()

    async def chunks():
        yield payload

    await state.initialize(namespace="local", tenant_id="tenant")
    try:
        await state.object_store(RuntimeDomain.EXECUTION).put(
            "example", chunks(), expected_digest=digest, expected_size=len(payload),
        )
    finally:
        await state.close()
    assert path.is_file()
    schema_reads.clear()
    reopened = RuntimeState(RuntimeStorage.sqlite(path))
    await reopened.initialize(namespace="local", tenant_id="tenant")
    try:
        assert await read_object(
            reopened.object_store(RuntimeDomain.EXECUTION), "example",
            expected_digest=digest, expected_size=len(payload),
        ) == payload
        assert schema_reads == []
    finally:
        await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ddl,category", (
    ("CREATE TABLE schema_example (id INTEGER PRIMARY KEY)", "column"),
    ("CREATE TABLE schema_example (id INTEGER PRIMARY KEY, token INTEGER NOT NULL, UNIQUE (token))", "type"),
    ("CREATE TABLE schema_example (id INTEGER PRIMARY KEY, token VARCHAR(32) NOT NULL)", "unique"),
))
async def test_explicit_validation_retains_column_type_and_unique_diagnostics(
    tmp_path: Path, ddl: str, category: str, schema_reads: list[str],
) -> None:
    metadata = MetaData()
    Table("schema_example", metadata, Column("id", Integer, primary_key=True),
          Column("token", String(32), nullable=False), UniqueConstraint("token"))
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'invalid.db'}")
    context = create_sql_storage_context(engine)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(ddl)
        await context.initialize()
        assert schema_reads == []
        with pytest.raises(AIError) as error:
            await validate_sql(engine, metadata)
        assert error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        assert error.value.safe_details["table"] == "schema_example"
        assert error.value.safe_details["category"] == category
        # Existing explicit metadata initialization remains a compatible opt-in.
        with pytest.raises(AIError) as error:
            await context.initialize(metadata=metadata)
        assert error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        assert schema_reads
    finally:
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_existing_metric_validation_option_remains_explicit(
    tmp_path: Path, schema_reads: list[str],
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'metrics.db'}")
    try:
        await provision_database(engine)
        schema_reads.clear()
        assert await SqlMetricStore(engine, validate_schema=True).get_definition("metrics", "missing", 1) is None
        assert schema_reads
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_read_only_missing_sqlite_does_not_create_a_database(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    state = RuntimeState(RuntimeStorage.sqlite(path))
    with pytest.raises(AIError) as error:
        await state.initialize(namespace="reader", tenant_id="tenant", read_only=True)
    assert error.value.code is ErrorCode.STORAGE_NOT_FOUND
    assert not path.exists()
