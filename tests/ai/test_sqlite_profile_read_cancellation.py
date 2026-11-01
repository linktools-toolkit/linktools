#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancelled SQLite profile reads must release their real buffered cursors."""

import asyncio
import sqlite3
from pathlib import Path

import pytest
from aiosqlite import Cursor
from sqlalchemy import Column, Integer, MetaData, Table, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.storage import create_sql_storage_context, provision_sql


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["permission", "mutation", "inspect", "provision"])
@pytest.mark.parametrize("repeated_cancel", [False, True])
@pytest.mark.parametrize("pragma", ["PRAGMA database_list", "PRAGMA main.application_id"])
async def test_cancelled_profile_read_does_not_retain_shared_source_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    repeated_cancel: bool,
    pragma: str,
) -> None:
    path = tmp_path / "shared.db"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE sentinel (value INTEGER NOT NULL)")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    writer = create_async_engine(f"sqlite+aiosqlite:///{path}")
    context = create_sql_storage_context(engine)
    await context.initialize()
    executing = asyncio.Event()
    release = asyncio.Event()
    cursors: list[Cursor] = []
    callback_calls = []
    buffered = asyncio.Event()
    original_execute, original_fetchall = Cursor.execute, Cursor.fetchall

    async def execute(cursor, sql, *args, **kwargs):
        result = await original_execute(cursor, sql, *args, **kwargs)
        if sql == pragma and not cursors:
            cursors.append(cursor)
        return result

    async def fetchall(cursor):
        if cursors and cursor is cursors[0]:
            executing.set()
            await release.wait()
        result = await original_fetchall(cursor)
        if cursors and cursor is cursors[0]:
            buffered.set()
        return result

    async def mutate(session):
        callback_calls.append(True)
        await session.execute(text("INSERT INTO sentinel VALUES (99)"))

    async def operation():
        if entry == "permission":
            await context.require_write_access()
        elif entry == "mutation":
            await context.run_mutation(mutate)
        elif entry == "inspect":
            await context.is_local_sqlite_source(path)
        else:
            metadata = MetaData()
            Table("not_published", metadata, Column("value", Integer))
            await provision_sql(engine, metadata)

    monkeypatch.setattr(Cursor, "execute", execute)
    monkeypatch.setattr(Cursor, "fetchall", fetchall)
    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(executing.wait(), 5)
        task.cancel("stop profile read")
        await asyncio.sleep(0)
        if repeated_cancel:
            task.cancel("stop profile read again")
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError) as retained:
            await asyncio.wait_for(task, 5)
        assert "stop profile read" in str(retained.value)
        assert callback_calls == []
        # Keep the cancellation traceback alive while another connection writes;
        # a cursor stranded in that traceback must not retain a SQLite read lock.
        async with writer.begin() as connection:
            await connection.exec_driver_sql("INSERT INTO sentinel VALUES (1)")
        async with writer.connect() as connection:
            assert (await connection.exec_driver_sql("SELECT value FROM sentinel")).scalars().all() == [1]
            assert (await connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE name='not_published'"
            )).first() is None
        assert isinstance(retained.value, asyncio.CancelledError)
        assert buffered.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await context.close()
        await engine.dispose()
        await writer.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_owned_profile_io_preserves_result_and_failure_under_cancellation(cancel: bool, fail: bool) -> None:
    from linktools.ai.storage._sqlite_owner import _finish_owned_io

    entered, release = asyncio.Event(), asyncio.Event()
    result = object()
    failure = RuntimeError("synthetic profile I/O failure")

    async def buffered_read():
        entered.set()
        await release.wait()
        if fail:
            raise failure
        return result

    child = asyncio.create_task(buffered_read())
    owner = asyncio.create_task(_finish_owned_io(child))
    try:
        await entered.wait()
        if cancel:
            owner.cancel("stop owner")
            await asyncio.sleep(0)
            assert not child.cancelled() and not child.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError, match="stop owner") as cancelled:
                await owner
            assert cancelled.value.__cause__ is (failure if fail else None)
        elif fail:
            with pytest.raises(RuntimeError) as raised:
                await owner
            assert raised.value is failure
        else:
            assert await owner is result
        assert child.done() and not child.cancelled()
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
