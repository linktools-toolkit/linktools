#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistent reads preserve borrowed-engine isolation and pool reuse."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage import MySQLDialect


class _InterruptedSetup(RuntimeError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("isolation,skip_rollback", (
    ("READ COMMITTED", False), ("REPEATABLE READ", False),
    ("AUTOCOMMIT", False), ("AUTOCOMMIT", True),
))
async def test_mysql_consistent_reads_preserve_caller_transactions(
    _server, isolation, skip_rollback,
) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
    reader = create_async_engine(url, isolation_level=isolation, skip_autocommit_rollback=skip_rollback,
                                 pool_size=1, max_overflow=0, pool_pre_ping=True)
    writer = create_async_engine(url, isolation_level="READ COMMITTED")
    sessions = async_sessionmaker(reader, expire_on_commit=False)

    async def write(value):
        async with writer.begin() as connection:
            await connection.execute(text("UPDATE read_transaction_probe SET value=:value WHERE id=1"), {"value": value})

    async def pair(*, consistent=False, interrupt=None):
        await write(1)
        try:
            async with sessions() as session, session.begin():
                if consistent:
                    await MySQLDialect().begin_consistent_read(session)
                if interrupt == "setup":
                    raise _InterruptedSetup()
                if interrupt == "cancel":
                    raise asyncio.CancelledError()
                first = await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1"))
                if interrupt == "body":
                    raise _InterruptedSetup()
                await write(2)
                second = await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1"))
                return [first, second]
        except (asyncio.CancelledError, _InterruptedSetup) as error:
            return type(error).__name__

    try:
        async with writer.begin() as connection:
            await connection.execute(text("CREATE TABLE IF NOT EXISTS read_transaction_probe (id INTEGER PRIMARY KEY, value INTEGER NOT NULL) ENGINE=InnoDB"))
            await connection.execute(text("INSERT INTO read_transaction_probe VALUES (1,1) ON DUPLICATE KEY UPDATE value=1"))
        async with reader.connect() as connection:
            settings = tuple((await connection.execute(text("SELECT @@session.transaction_isolation, @@autocommit"))).one())
        expected = [1, 1] if isolation == "REPEATABLE READ" else [1, 2]
        for interruption in (None, "setup", "cancel", "body"):
            result = await pair(consistent=True, interrupt=interruption)
            assert result == ([1, 1] if interruption is None else
                              "CancelledError" if interruption == "cancel" else "_InterruptedSetup")
            assert await pair() == expected
            async with reader.connect() as connection:
                restored = tuple((await connection.execute(text("SELECT @@session.transaction_isolation, @@autocommit"))).one())
            assert restored == settings
    finally:
        await reader.dispose()
        await writer.dispose()
