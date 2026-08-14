#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit database provisioning entry-point checks."""

import sqlite3
from pathlib import Path

import pytest
from linktools.ai.asset import build_asset_sql_metadata
from linktools.ai.migrate import provision_asset_database, provision_runtime_database
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state._schema import build_runtime_sql_metadata
from sqlalchemy.ext.asyncio import create_async_engine


def _expected_tables() -> set[str]:
    runtime = build_runtime_sql_metadata(frozenset(RuntimeDomain))
    assets = build_asset_sql_metadata()
    return set(runtime.tables) | set(assets.tables)


@pytest.mark.asyncio
async def test_provision_schema_owners_creates_each_owner_tables(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await provision_runtime_database(engine)
        await provision_asset_database(engine)
        await provision_runtime_database(engine)
        await provision_asset_database(engine)
        with sqlite3.connect(path) as connection:
            actual_tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        assert actual_tables == _expected_tables()
    finally:
        await engine.dispose()
