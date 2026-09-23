#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite storage connection semantics."""

import asyncio

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



@pytest.mark.asyncio
async def test_sqlite_mutations_do_not_reserve_writer_before_callback(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'writers.db'}")
    context = create_sql_storage_context(engine)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first(_session) -> None:
        first_entered.set()
        await release_first.wait()

    async def second(_session) -> None:
        second_entered.set()

    try:
        await context.initialize()
        first_task = asyncio.create_task(context.run_mutation(first))
        await first_entered.wait()

        second_task = asyncio.create_task(context.run_mutation(second))
        await asyncio.wait_for(second_entered.wait(), timeout=1)
        await second_task
        assert not first_task.done()

        release_first.set()
        await first_task
    finally:
        release_first.set()
        await context.close()
        await engine.dispose()
