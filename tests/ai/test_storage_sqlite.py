#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite storage connection semantics."""

import pytest
from linktools.ai.storage import create_sql_storage_context
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_sqlite_storage_context_configures_checked_out_connections(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'storage.db'}")
    context = create_sql_storage_context(engine)
    try:
        await context.initialize()
        async with engine.connect() as connection:
            foreign_keys = await connection.exec_driver_sql("PRAGMA foreign_keys")
            busy_timeout = await connection.exec_driver_sql("PRAGMA busy_timeout")
            assert foreign_keys.scalar_one() == 1
            assert busy_timeout.scalar_one() == 5000
    finally:
        await context.close()
        await engine.dispose()
