#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistent reads preserve borrowed-engine isolation and pool reuse."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import nullcontext

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from linktools.ai.storage import MySQLDialect


class _InterruptedSetup(RuntimeError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("isolation,skip_rollback", (
    ("READ COMMITTED", False), ("REPEATABLE READ", False),
    ("AUTOCOMMIT", False), ("AUTOCOMMIT", True),
))
@pytest.mark.parametrize("explicit_begin", (True, False), ids=("commit", "close-rollback"))
async def test_mysql_consistent_reads_preserve_caller_transactions(
    _server: tuple[str, list[str]], isolation: str, skip_rollback: bool,
    explicit_begin: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asyncmy.connection import Connection
    from asyncmy.constants.COMMAND import COM_PING, COM_QUERY

    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
    reader = create_async_engine(url, isolation_level=isolation, skip_autocommit_rollback=skip_rollback,
                                 pool_size=1, max_overflow=0, pool_pre_ping=True)
    writer = create_async_engine(url, isolation_level="READ COMMITTED")
    sessions = async_sessionmaker(reader, expire_on_commit=False)
    owned = set()
    checkouts = []
    event.listen(reader.sync_engine, "connect", lambda connection, record: owned.add(connection.driver_connection))
    event.listen(reader.sync_engine, "checkout", lambda connection, *args: checkouts.append(connection.driver_connection))
    commands = []
    recording = False
    in_flight = None
    version = 0
    reached = asyncio.Event()
    blocked = asyncio.Event()
    original_command = Connection._execute_command

    async def execute_command(connection, command, sql):
        result = await original_command(connection, command, sql)
        if connection in owned:
            operation = sql.decode("utf-8") if isinstance(sql, bytes) else sql
            operation = " ".join(operation.upper().split())
            if recording:
                commands.append((command, operation))
            if command == COM_QUERY and operation == in_flight:
                # The command has gone to MySQL, but its response is still unread.
                reached.set()
                await blocked.wait()
        return result

    monkeypatch.setattr(Connection, "_execute_command", execute_command)

    async def write(value: int) -> None:
        async with writer.begin() as connection:
            await connection.execute(text("UPDATE read_transaction_probe SET value=:value WHERE id=1"), {"value": value})

    async def pair(
        begin: Callable[[AsyncSession], Awaitable[None]] | None = None,
        *, boundary: str | None = None, failure: str | None = None,
    ) -> list[int] | None:
        nonlocal version
        # Distinct writes also expose a leaked snapshot on a REPEATABLE READ engine.
        version += 2
        initial = version
        await write(initial)

        async def interrupt(stage: str) -> None:
            if stage == boundary:
                if failure == "error":
                    raise _InterruptedSetup()
                reached.set()
                await blocked.wait()

        async with sessions() as session, (session.begin() if explicit_begin else nullcontext()):
            if begin is not None:
                original_execute = session.execute

                async def execute(statement, *args, **kwargs):
                    result = await original_execute(statement, *args, **kwargs)
                    if str(statement) == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ":
                        await interrupt("set")
                    elif str(statement) == "START TRANSACTION":
                        await interrupt("start")
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(session, "execute", execute)
                    await begin(session)
                await interrupt("setup")
            if boundary == "empty":
                return None
            first = await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1"))
            await interrupt("read")
            await write(initial + 1)
            second = await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1"))
            return [first - initial + 1, second - initial + 1]

    async def session_isolation(session: AsyncSession) -> None:
        await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})

    async def cancel(task: asyncio.Task) -> None:
        try:
            await asyncio.wait_for(reached.wait(), timeout=10)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    try:
        async with writer.begin() as connection:
            await connection.execute(text("CREATE TABLE IF NOT EXISTS read_transaction_probe (id INTEGER PRIMARY KEY, value INTEGER NOT NULL) ENGINE=InnoDB"))
            await connection.execute(text("INSERT INTO read_transaction_probe VALUES (1,1) ON DUPLICATE KEY UPDATE value=1"))
        async with reader.connect() as connection:
            settings = tuple((await connection.execute(text("SELECT @@session.transaction_isolation, @@autocommit"))).one())
        expected = [1, 1] if isolation == "REPEATABLE READ" else [1, 2]

        async def borrower(*, previous: Connection, replaced: bool = False) -> None:
            assert await pair() == expected
            assert (checkouts[-1] is not previous) == replaced
            async with reader.connect() as connection:
                restored = tuple((await connection.execute(text("SELECT @@session.transaction_isolation, @@autocommit"))).one())
            assert restored == settings

        traces = {}
        for name, begin in (("session", session_isolation), ("transaction", MySQLDialect().begin_consistent_read)):
            commands.clear()
            recording = True
            try:
                assert await pair(begin) == [1, 1]
            finally:
                recording = False
            traces[name] = tuple(commands)
            await borrower(previous=checkouts[-1])
        actual = traces["transaction"]
        assert actual[0] == (COM_PING, "")
        if skip_rollback:
            assert actual == traces["session"]
            assert any(sql.startswith("SET SESSION") for _, sql in actual)
        else:
            assert [sql for _, sql in actual if sql.startswith(("SET", "START"))] == [
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", "START TRANSACTION",
            ]
            assert sum(command == COM_QUERY and sql == "COMMIT" for command, sql in actual) == int(explicit_begin)
            assert len(actual) < len(traces["session"])

        begin = MySQLDialect().begin_consistent_read
        assert await pair(begin, boundary="empty") is None
        await borrower(previous=checkouts[-1])
        boundaries = ("setup", "read") if skip_rollback else ("set", "start", "read")
        for boundary in boundaries:
            for failure in ("error", "cancel"):
                previous = checkouts[-1]
                if failure == "error":
                    with pytest.raises(_InterruptedSetup):
                        await pair(begin, boundary=boundary, failure=failure)
                else:
                    reached.clear()
                    await cancel(asyncio.create_task(pair(begin, boundary=boundary, failure=failure)))
                await borrower(previous=previous)

        if not skip_rollback:
            for statement in ("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", "START TRANSACTION"):
                previous = checkouts[-1]
                reached.clear()
                in_flight = statement
                try:
                    await cancel(asyncio.create_task(pair(begin)))
                finally:
                    in_flight = None
                await borrower(previous=previous, replaced=True)

            # SET must reject a physical transaction before START could commit it.
            await write(1)
            async with sessions() as session, session.begin():
                if isolation == "AUTOCOMMIT":
                    await session.execute(text("START TRANSACTION"))
                await session.execute(text("UPDATE read_transaction_probe SET value=3 WHERE id=1"))
                with pytest.raises(DBAPIError) as rejected:
                    await begin(session)
                assert rejected.value.orig.args[0] == 1568
                assert await session.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1")) == 3
                async with writer.connect() as connection:
                    assert await connection.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1")) == 1
                await session.rollback()
            async with writer.connect() as connection:
                assert await connection.scalar(text("SELECT value FROM read_transaction_probe WHERE id=1")) == 1
            await borrower(previous=checkouts[-1])
    finally:
        await reader.dispose()
        await writer.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_group_read_reuses_snapshot_and_restores_borrowed_engine(_server) -> None:
    import hashlib
    from dataclasses import replace

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
