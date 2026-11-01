#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite storage connection semantics."""

import asyncio
import sqlite3

import pytest
from linktools.ai.storage import create_sql_storage_context
from sqlalchemy.exc import OperationalError
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
            await connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            await connection.exec_driver_sql("PRAGMA busy_timeout=1")
        async with engine.connect() as connection:
            foreign_keys = await connection.exec_driver_sql("PRAGMA foreign_keys")
            busy_timeout = await connection.exec_driver_sql("PRAGMA busy_timeout")
            assert foreign_keys.scalar_one() == 1
            assert busy_timeout.scalar_one() == 5000
    finally:
        await context.close()
        await engine.dispose()



@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "error"])
@pytest.mark.parametrize("stage", ["cursor", "execute"])
async def test_sqlite_checkout_releases_configuration_resources_on_failure(tmp_path, monkeypatch, failure, stage) -> None:
    import aiosqlite
    from aiosqlite.context import contextmanager

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'checkout.db'}")
    context = create_sql_storage_context(engine)
    entered = asyncio.Event()
    configuration_cursors = []
    configuration_connections = []
    closed_cursors = []
    closed_connections = []
    execute = aiosqlite.Cursor.execute
    close = aiosqlite.Cursor.close
    connection_close = aiosqlite.Connection.close
    connection_execute = aiosqlite.Connection.execute
    task = None

    async def fail():
        entered.set()
        if failure == "error":
            raise sqlite3.OperationalError("configuration failed")
        await asyncio.Event().wait()

    @contextmanager
    async def allocating(connection, statement, parameters=None):
        cursor = await connection_execute(connection, statement, parameters)
        configuration_cursors.append(cursor)
        configuration_connections.append(connection)
        await fail()
        return cursor

    async def interrupt(cursor, statement, parameters=None):
        result = await execute(cursor, statement, parameters)
        if statement == "PRAGMA busy_timeout=5000":
            configuration_cursors.append(cursor)
            await fail()
        return result

    async def closing(cursor):
        try:
            return await close(cursor)
        finally:
            closed_cursors.append(cursor)

    async def closing_connection(connection):
        try:
            return await connection_close(connection)
        finally:
            closed_connections.append(connection)

    async def borrow():
        async with engine.connect():
            pass

    try:
        await context.initialize()
        async with engine.connect():
            pass
        if stage == "cursor":
            monkeypatch.setattr(aiosqlite.Connection, "execute", allocating)
        else:
            monkeypatch.setattr(aiosqlite.Cursor, "execute", interrupt)
        monkeypatch.setattr(aiosqlite.Cursor, "close", closing)
        monkeypatch.setattr(aiosqlite.Connection, "close", closing_connection)
        task = asyncio.create_task(borrow())
        await asyncio.wait_for(entered.wait(), 3)
        if failure == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(OperationalError, match="configuration failed") as raised:
                await task
            assert isinstance(raised.value.orig, sqlite3.OperationalError)
        assert configuration_cursors
        if stage == "cursor":
            assert all(connection in closed_connections for connection in configuration_connections)
        else:
            assert all(cursor in closed_cursors for cursor in configuration_cursors)
    finally:
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
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
