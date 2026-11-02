#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SQLite failure cleanup must not block the event loop during GC."""

import asyncio
import gc
import sqlite3
import threading
import weakref
from pathlib import Path

import aiosqlite
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from linktools.ai.storage import create_sqlite_storage_context
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
async def test_failed_cursor_gc_cannot_block_a_later_owned_commit(tmp_path: Path, monkeypatch):
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
    context = create_sqlite_storage_context(path)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        async with context.engine.connect() as reader, context.engine.connect() as writer:
            await reader.exec_driver_sql("CREATE TABLE sample (value INTEGER)")
            await reader.exec_driver_sql("INSERT INTO sample VALUES (0)")
            await reader.commit()
            await writer.exec_driver_sql("PRAGMA busy_timeout=100")
            await reader.exec_driver_sql("BEGIN")
            await reader.exec_driver_sql("UPDATE sample SET value=1")
            with pytest.raises(DBAPIError) as caught:
                await writer.exec_driver_sql("UPDATE sample SET value=2")
            assert caught.value.orig.sqlite_errorcode == sqlite3.SQLITE_BUSY
            del caught
            await writer.rollback()
            await reader.rollback()
            assert len(failed_cursors) == 1 and failed_cursors[0]() is not None
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
async def test_owned_native_statement_settles_before_caller_cancellation(tmp_path: Path, fails):
    context = create_sqlite_storage_context(tmp_path / "cancel.sqlite")
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
