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


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_group_read_reuses_snapshot_and_restores_borrowed_engine(_server) -> None:
    import hashlib
    from dataclasses import replace

    from sqlalchemy import event

    from linktools.ai.migrate import provision_database
    from linktools.ai.runtime.state._sql import SqlStateStore
    from linktools.ai.runtime.state._store import StoredRecord, active_state_scope

    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
    reader = create_async_engine(url, isolation_level="READ COMMITTED", pool_size=1,
                                 max_overflow=0, pool_pre_ping=True, hide_parameters=True)
    writer = create_async_engine(url, isolation_level="READ COMMITTED", hide_parameters=True)
    await provision_database(writer)
    digests = [hashlib.sha256(value.encode()).digest() for value in ("group-a", "group-b", "record")]
    first = SqlStateStore(reader, store_digest=digests[0])
    second = SqlStateStore(reader, store_digest=digests[1], group=first.storage_group)
    writer_first = SqlStateStore(writer, store_digest=digests[0])
    writer_second = SqlStateStore(writer, store_digest=digests[1], group=writer_first.storage_group)
    stores = (first, second, writer_first, writer_second)
    value = StoredRecord(digests[2], None, None, "probe", "record", None,
                         0, None, 0, None, {"value": 1})
    checkouts = []
    event.listen(reader.sync_engine, "checkout", lambda *args: checkouts.append(True))
    try:
        for store in stores:
            await store.initialize()

        async def seed(scope):
            for store in (writer_first, writer_second):
                await scope.transaction(store).insert_record(value)

        await writer_first.storage_group.mutate((writer_first, writer_second), seed)
        checkouts.clear()

        async def capture(scope):
            observed = await first.read(lambda tx: tx.get_record(value.key_digest))
            assert observed == value
            # The writer runs in an independent task, without inheriting reader ownership.
            async def publish():
                async def update(write):
                    for store in (writer_first, writer_second):
                        assert await write.transaction(store).replace_record(
                            replace(value, storage_version=1, data={"value": 2}), expected_storage_version=0,
                        )
                await writer_first.storage_group.mutate((writer_first, writer_second), update)
            import contextvars
            task = contextvars.Context().run(asyncio.create_task, publish())
            await task
            assert await second.read(lambda tx: tx.get_record(value.key_digest)) == value
            assert await scope.transaction(first).get_record(value.key_digest) == value

        await first.storage_group.read_many((first, second), capture)
        assert len(checkouts) == 1
        assert active_state_scope() is None
        assert (await second.read(lambda tx: tx.get_record(value.key_digest))).data["value"] == 2

        for interruption in (_InterruptedSetup, asyncio.CancelledError):
            async def fail(scope):
                assert await scope.transaction(first).get_record(value.key_digest) is not None
                raise interruption()
            with pytest.raises(interruption):
                await first.storage_group.read_many((first, second), fail)
            assert active_state_scope() is None
            async with reader.connect() as connection:
                settings = tuple((await connection.execute(text(
                    "SELECT @@session.transaction_isolation, @@autocommit"
                ))).one())
                assert settings == ("READ-COMMITTED", 0)
            assert (await first.read(lambda tx: tx.get_record(value.key_digest))).data["value"] == 2
    finally:
        for store in reversed(stores):
            await store.close()
        await reader.dispose()
        await writer.dispose()
