#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistent reads preserve borrowed-engine isolation and pool reuse."""

import asyncio
import json
from collections import Counter

import pytest
from sqlalchemy import event, text
from sqlalchemy.dialects.mysql.asyncmy import AsyncAdapt_asyncmy_connection, AsyncAdapt_asyncmy_cursor
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
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
    _server, isolation, skip_rollback, monkeypatch, capsys,
) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database="metrics_test", query={"unix_socket": socket})
    reader = create_async_engine(url, isolation_level=isolation, skip_autocommit_rollback=skip_rollback,
                                 pool_size=1, max_overflow=0, pool_pre_ping=True)
    writer = create_async_engine(url, isolation_level="READ COMMITTED")
    sessions = async_sessionmaker(reader, expire_on_commit=False)
    owned = set()
    event.listen(reader.sync_engine, "connect", lambda connection, record: owned.add(id(connection)))
    counters: Counter[str] = Counter()
    enabled = False
    original_execute = AsyncAdapt_asyncmy_cursor.execute

    def execute(cursor, operation, parameters=None):
        if enabled and id(cursor._adapt_connection) in owned:
            words = str(operation).split()
            counters["sql:" + " ".join(words[:2] if words[0] in {"SET", "START"} else words[:1])]+=1
        return original_execute(cursor, operation, parameters)

    monkeypatch.setattr(AsyncAdapt_asyncmy_cursor, "execute", execute)
    for name in ("commit", "rollback", "autocommit", "ping"):
        original = getattr(AsyncAdapt_asyncmy_connection, name)
        def lifecycle(connection, *args, _name=name, _original=original, **kwargs):
            if enabled and id(connection) in owned:
                counters["dbapi:" + _name] += 1
            return _original(connection, *args, **kwargs)
        monkeypatch.setattr(AsyncAdapt_asyncmy_connection, name, lifecycle)

    async def write(value):
        async with writer.begin() as connection:
            await connection.execute(text("UPDATE read_transaction_probe SET value=:value WHERE id=1"), {"value": value})

    async def pair(begin=None, *, interrupt=None):
        await write(1)
        try:
            async with sessions() as session, session.begin():
                if begin is not None:
                    await begin(session, interrupt)
                first = await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1"))
                if interrupt == "body":
                    raise _InterruptedSetup()
                await write(2)
                second = await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1"))
                return [first, second]
        except (asyncio.CancelledError, _InterruptedSetup) as error:
            return type(error).__name__

    async def current(session, interrupt):
        await MySQLDialect().begin_consistent_read(session)
        if interrupt == "setup":
            raise _InterruptedSetup()
        if interrupt == "cancel":
            raise asyncio.CancelledError()

    async def transaction_scoped(session, interrupt):
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
        if interrupt == "setup":
            raise _InterruptedSetup()
        if interrupt == "cancel":
            raise asyncio.CancelledError()
        await session.execute(text("START TRANSACTION"))

    try:
        async with writer.begin() as connection:
            await connection.execute(text("CREATE TABLE IF NOT EXISTS read_transaction_probe (id INTEGER PRIMARY KEY, value INTEGER NOT NULL) ENGINE=InnoDB"))
            await connection.execute(text("INSERT INTO read_transaction_probe VALUES (1,1) ON DUPLICATE KEY UPDATE value=1"))
        async with reader.connect() as connection:
            settings = tuple((await connection.execute(text("SELECT @@session.transaction_isolation, @@autocommit"))).one())
        expected = [1, 1] if isolation == "REPEATABLE READ" else [1, 2]
        observations = {}
        for name, begin in (("production", current), ("transaction_scoped_candidate", transaction_scoped)):
            observed = {}
            for interruption in (None, "setup", "cancel", "body"):
                counters.clear()
                enabled = True
                try:
                    try:
                        result = await pair(begin, interrupt=interruption)
                    except DBAPIError as error:
                        if name == "production":
                            raise
                        result = {"database_error": type(error.orig).__name__,
                                  "driver_error_code": error.orig.args[0]}
                finally:
                    enabled = False
                counts = dict(counters)
                reused = await pair()
                async with reader.connect() as connection:
                    restored = tuple((await connection.execute(text("SELECT @@session.transaction_isolation, @@autocommit"))).one())
                observed[interruption or "success"] = {
                    "result": result, "following_transaction": reused, "settings_restored": restored == settings,
                    "driver_cursor_statements_and_lifecycle_calls": counts,
                }
                if name == "production":
                    assert result == ([1, 1] if interruption is None else
                                      "CancelledError" if interruption == "cancel" else "_InterruptedSetup")
                    assert reused == expected
                    assert restored == settings
                # A probe must not contaminate the next independent experiment.
                await reader.dispose()
            observations[name] = observed
        with capsys.disabled():
            print("MYSQL_READ_TRANSACTION_PROBE " + json.dumps({
                "isolation": isolation, "skip_autocommit_rollback": skip_rollback,
                "expected_following_transaction": expected, "observations": observations,
            }, sort_keys=True))
    finally:
        await reader.dispose()
        await writer.dispose()
