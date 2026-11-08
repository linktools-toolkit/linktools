#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SQLite failure cleanup must not block the event loop during GC."""

import asyncio
import gc
import sqlite3
import threading
import traceback
import weakref
from pathlib import Path

import aiosqlite
import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage import create_sql_storage_context, create_sqlite_storage_context
from linktools.ai.storage._sqlite_driver import _OwnedConnection, _OwnedCursor


@pytest.mark.parametrize("operation", ("execute", "executemany", "executescript"))
@pytest.mark.parametrize("convenience", (False, True))
def test_failed_native_cursor_closes_with_original_database_error(operation, convenience):
    created = []

    class Connection(_OwnedConnection):
        def cursor(self, factory=_OwnedCursor):
            cursor = super().cursor(factory)
            created.append(cursor)
            return cursor

    connection = sqlite3.connect(":memory:", factory=Connection)
    cursor = connection.cursor()
    connection.execute("CREATE TABLE sample (value INTEGER UNIQUE)").close()
    connection.execute("INSERT INTO sample VALUES (1)").close()
    created.clear()
    target = connection if convenience else cursor
    arguments = {
        "execute": ("INSERT INTO sample VALUES (?)", (1,)),
        "executemany": ("INSERT INTO sample VALUES (?)", ((1,),)),
        "executescript": ("INSERT INTO sample VALUES (1);",),
    }[operation]
    try:
        with pytest.raises(sqlite3.IntegrityError) as caught:
            getattr(target, operation)(*arguments)
        assert caught.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE
        failed = created[-1] if convenience else cursor
        with pytest.raises(sqlite3.ProgrammingError, match="closed cursor"):
            failed.execute("SELECT 1")
        assert connection.execute("SELECT COUNT(*) FROM sample").fetchone() == (1,)
    finally:
        connection.close()


def test_successful_native_cursor_remains_readable_and_custom_factory_is_honored():
    class CustomCursor(sqlite3.Cursor):
        pass

    with sqlite3.connect(":memory:", factory=_OwnedConnection) as connection:
        cursor = connection.execute("SELECT 1 UNION ALL SELECT 2")
        assert isinstance(cursor, _OwnedCursor)
        assert cursor.fetchone() == (1,)
        assert cursor.fetchall() == [(2,)]
        assert isinstance(connection.cursor(factory=CustomCursor), CustomCursor)
        cursor.close()
    connection.close()


def test_cleanup_failure_does_not_replace_the_original_error():
    connection = sqlite3.connect(":memory:", factory=_OwnedConnection)
    connection.execute("CREATE TABLE sample (value INTEGER)").close()
    cursor = connection.cursor()
    original = RuntimeError("parameter iteration failed")

    def parameters():
        yield (1,)
        connection.close()
        raise original

    with pytest.raises(RuntimeError) as caught:
        cursor.executemany("INSERT INTO sample VALUES (?)", parameters())
    assert caught.value is original
    assert isinstance(caught.value.__cause__, sqlite3.ProgrammingError)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_kind, operation", (
    ("owned", "write"), ("borrowed-statement", "write"),
    ("borrowed-mutation", "write"), ("borrowed-statement", "read"),
))
async def test_failed_cursor_gc_cannot_block_a_later_commit(tmp_path: Path, monkeypatch, engine_kind, operation):
    failed_cursors = []
    original_execute = aiosqlite.Cursor.execute

    async def execute(cursor, *args, **kwargs):
        try:
            return await original_execute(cursor, *args, **kwargs)
        except Exception:
            failed_cursors.append(weakref.ref(cursor))
            raise

    monkeypatch.setattr(aiosqlite.Cursor, "execute", execute)
    path = tmp_path / "state.sqlite"
    context = (create_sqlite_storage_context(path) if engine_kind == "owned" else
               create_sql_storage_context(create_async_engine(f"sqlite+aiosqlite:///{path}")))
    await context.initialize()
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        async with context.engine.connect() as reader, context.engine.connect() as writer:
            if engine_kind == "borrowed-mutation":
                context.sessions = async_sessionmaker(bind=writer, expire_on_commit=False)
            await reader.exec_driver_sql("CREATE TABLE sample (value INTEGER)")
            await reader.exec_driver_sql("INSERT INTO sample VALUES (0)")
            await reader.commit()
            await writer.exec_driver_sql("PRAGMA busy_timeout=100")
            if operation == "read":
                (await writer.exec_driver_sql("SELECT * FROM sample")).all()
                await writer.rollback()
            await reader.exec_driver_sql("BEGIN EXCLUSIVE" if operation == "read" else "BEGIN")
            await reader.exec_driver_sql("UPDATE sample SET value=1")
            with pytest.raises(DBAPIError) as caught:
                if engine_kind == "borrowed-mutation":
                    await context.run_mutation(
                        lambda session: session.execute(text("UPDATE sample SET value=2")), retry_limit=1,
                    )
                elif engine_kind == "borrowed-statement":
                    await context.run_statement(writer.exec_driver_sql(
                        "SELECT * FROM sample" if operation == "read" else "UPDATE sample SET value=2",
                    ))
                else:
                    await writer.exec_driver_sql("UPDATE sample SET value=2")
            assert caught.value.orig.sqlite_errorcode == sqlite3.SQLITE_BUSY
            del caught
            await writer.rollback()
            await reader.rollback()
            assert len(failed_cursors) == 1
            await reader.exec_driver_sql("BEGIN")
            (await reader.exec_driver_sql("SELECT * FROM sample")).all()
            await writer.exec_driver_sql("UPDATE sample SET value=3")
            raw = await writer.get_raw_connection()
            entered = threading.Event()
            await raw.driver_connection.set_trace_callback(
                lambda statement: entered.set() if statement == "COMMIT" else None,
            )

            async def commit():
                try:
                    await writer.commit()
                except DBAPIError as error:
                    return error.orig.sqlite_errorcode
                return None

            pending = asyncio.create_task(commit())
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                # Reclaim this newly created failure cycle, not unrelated
                # test-suite generations whose CPU cost is not this invariant.
                gc.collect(0)
                assert failed_cursors[0]() is None
                await reader.rollback()
                assert await pending is None
            finally:
                await reader.rollback()
                await pending
                await writer.rollback()
    finally:
        if was_enabled:
            gc.enable()
        await context.close()
        if engine_kind != "owned":
            await context.engine.dispose()


@pytest.mark.asyncio
async def test_owned_engine_stream_remains_open_until_consumer_finishes(tmp_path: Path):
    context = create_sqlite_storage_context(tmp_path / "stream.sqlite")
    try:
        async with context.engine.connect() as connection:
            async with connection.stream(text("SELECT 1 UNION ALL SELECT 2")) as rows:
                assert tuple(await rows.fetchone()) == (1,)
                assert [tuple(row) async for row in rows] == [(2,)]
            assert rows.closed
            assert (await connection.exec_driver_sql("SELECT 3")).scalar_one() == 3
    finally:
        await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", (False, True))
@pytest.mark.parametrize("engine_kind", ("owned", "borrowed"))
async def test_native_statement_settles_before_caller_cancellation(tmp_path: Path, fails, engine_kind):
    path = tmp_path / "cancel.sqlite"
    context = (create_sqlite_storage_context(path) if engine_kind == "owned" else
               create_sql_storage_context(create_async_engine(f"sqlite+aiosqlite:///{path}")))
    await context.initialize()
    entered = threading.Event()
    release = threading.Event()

    def native_work():
        entered.set()
        assert release.wait(5)
        if fails:
            raise RuntimeError("native callback failed")
        return 1

    try:
        async with context.engine.connect() as connection:
            raw = await connection.get_raw_connection()
            await raw.driver_connection.create_function("native_work", 0, native_work)
            pending = asyncio.create_task(context.run_statement(connection.exec_driver_sql("SELECT native_work()")))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                pending.cancel("stop native read")
                await asyncio.sleep(0)
                assert not pending.done()
                release.set()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await pending
                assert caught.value.args == ("stop native read",)
                if fails:
                    assert isinstance(caught.value.__cause__, DBAPIError)
                await connection.rollback()
                assert (await connection.exec_driver_sql("SELECT 2")).scalar_one() == 2
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
    finally:
        await context.close()
        if engine_kind != "owned":
            await context.engine.dispose()


@pytest.mark.asyncio
async def test_sqlite_statement_failure_preserves_database_diagnostics(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'diagnostics.sqlite'}")
    context = create_sql_storage_context(engine)
    failures = []

    def record_error(error):
        failures.append(error.sqlalchemy_exception)

    event.listen(engine.sync_engine, "handle_error", record_error)
    try:
        await context.initialize()
        async with engine.begin() as connection:
            await connection.exec_driver_sql("CREATE TABLE sample (value INTEGER UNIQUE)")
            await connection.exec_driver_sql("INSERT INTO sample VALUES (1)")
        statement = "INSERT INTO sample VALUES (?)"
        async with engine.connect() as connection:
            with pytest.raises(DBAPIError) as caught:
                await context.run_statement(connection.exec_driver_sql(statement, (1,)))
            assert caught.value is failures[0]
            assert caught.value.statement == statement
            assert caught.value.params == (1,)
            assert caught.value.orig.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE
            assert caught.value.__cause__ is caught.value.orig
            assert traceback.extract_tb(caught.value.__traceback__)
            assert "UNIQUE constraint failed" in "".join(traceback.format_exception(caught.value))
            await connection.rollback()
            assert (await connection.exec_driver_sql("SELECT value FROM sample")).scalar_one() == 1
    finally:
        await context.close()
        await engine.dispose()
