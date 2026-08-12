#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit database provisioning entry-point checks."""

import sqlite3
from pathlib import Path

import pytest
from linktools.ai.adapter import SqlRuntimeSchema, build_step_schema
from linktools.ai.asset import SqlAssetSchema
from linktools.ai.migrate import provision_database
from linktools.ai.storage import SqlSchemaRegistry
from sqlalchemy.ext.asyncio import create_async_engine


def _expected_tables() -> set[str]:
    runtime_registry = SqlSchemaRegistry()
    runtime_manifest = SqlRuntimeSchema.register_schema(runtime_registry)
    asset_registry = SqlSchemaRegistry()
    asset_tables = SqlAssetSchema.register_schema(asset_registry)
    return {
        *(table.name for table in runtime_manifest.tables.values()),
        *build_step_schema().tables,
        *(table.name for table in (asset_tables.entry, asset_tables.change, asset_tables.blob, asset_tables.revision)),
    }


@pytest.mark.asyncio
async def test_provision_database_creates_all_schema_owner_tables(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_database(engine)
        await provision_database(engine)
        with sqlite3.connect(path) as connection:
            actual_tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        assert actual_tables == _expected_tables()
    finally:
        await engine.dispose()
