#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default SQLite contention waits do not replace explicit attempt contracts."""

import asyncio
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import (
    SQLiteDialect, SqlStorageContext, SqlTransactionDisposition, create_sql_storage_context,
)
from linktools.ai.storage import _database


@pytest.mark.asyncio
@pytest.mark.parametrize("limit, error_type", (
    (0, ValueError), (-1, ValueError), (1.5, TypeError),
    (float("inf"), TypeError), (float("nan"), TypeError), ("3", TypeError),
))
async def test_sqlite_retry_limit_rejects_invalid_values_before_callback(limit: object, error_type: type) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    context = create_sql_storage_context(engine)
    called = False

    async def mutate(session: AsyncSession) -> None:
        nonlocal called
        called = True

    try:
        with pytest.raises(error_type):
            await context.run_mutation(mutate, retry_limit=limit)
        assert not called
    finally:
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", (None, 1, 3, 8))
async def test_sqlite_busy_default_waits_for_writer_but_explicit_attempts_stay_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int | None,
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'busy.db'}"
    engines = (create_async_engine(url), create_async_engine(url))
    reader, writer = (create_sql_storage_context(engine) for engine in engines)
    reserved, release, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts, pauses, observed = [], [], []
    reader_task = writer_task = None
    sleep = asyncio.sleep

    async def hold(session: AsyncSession) -> None:
        await session.execute(text("UPDATE ai_values SET value = 1"))
        reserved.set()
        await release.wait()

    async def run_writer() -> None:
        await writer.run_mutation(hold)
        committed.set()

    async def increment(session: AsyncSession) -> int:
        attempts.append(session)
        observed.append(await session.scalar(text("SELECT value FROM ai_values")))
        await session.execute(text("UPDATE ai_values SET value = value + 1"))
        return observed[-1] + 1

    async def release_after_eight_aborts(delay, *args, **kwargs):
        if asyncio.current_task() is reader_task and delay > 0:
            assert engines[0].sync_engine.pool.checkedout() == 0
            pauses.append(delay)
            if len(pauses) == 8:
                release.set()
                await asyncio.wait_for(committed.wait(), 5)
            return await sleep(0)
        return await sleep(delay, *args, **kwargs)

    try:
        await reader.initialize()
        await writer.initialize()
        async with engines[0].begin() as connection:
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
            await connection.execute(text("INSERT INTO ai_values (value) VALUES (0)"))
        monkeypatch.setattr(_database.asyncio, "sleep", release_after_eight_aborts)
        writer_task = asyncio.create_task(run_writer())
        await asyncio.wait_for(reserved.wait(), 5)
        options = {} if limit is None else {"retry_limit": limit}
        reader_task = asyncio.create_task(reader.run_mutation(increment, **options))
        if limit is None:
            assert await reader_task == 2
            assert observed == [0] * 8 + [1]
        else:
            with pytest.raises(OperationalError) as caught:
                await reader_task
            assert caught.value.orig.sqlite_errorcode & 255 == sqlite3.SQLITE_BUSY
            assert observed == [0] * limit
            assert not release.is_set()
        assert len(pauses) == len(attempts) - 1
        assert engines[0].sync_engine.pool.checkedout() == 0
        release.set()
        await writer_task
        async with engines[0].connect() as connection:
            assert await connection.scalar(text("SELECT value FROM ai_values")) == (2 if limit is None else 1)
    finally:
        release.set()
        for task in (reader_task, writer_task):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        await reader.close()
        await writer.close()
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("close", "sleep", "cancel"))
async def test_sqlite_busy_retry_admission_keeps_deadline_error_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'deadline.db'}"
    engines = (create_async_engine(url), create_async_engine(url))
    reader, writer = (create_sql_storage_context(engine) for engine in engines)
    reserved, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clock = [0.0]
    seen, pauses = [], []
    worker = competing = None
    sleep, close = asyncio.sleep, AsyncSession.close

    async def hold(session: AsyncSession) -> None:
        await session.execute(text("UPDATE ai_values SET value = 1"))
        reserved.set()
        await release.wait()

    async def mutate(session: AsyncSession) -> None:
        await session.scalar(text("SELECT value FROM ai_values"))
        try:
            await session.execute(text("UPDATE ai_values SET value = value + 1"))
        except OperationalError as error:
            seen.append(error)
            if boundary == "sleep":
                clock[0] = 4.999
            raise

    async def closing(session: AsyncSession) -> None:
        await close(session)
        if asyncio.current_task() is worker and boundary == "close":
            clock[0] = 5.0

    async def pausing(delay, *args, **kwargs):
        if asyncio.current_task() is worker and delay > 0:
            assert engines[0].sync_engine.pool.checkedout() == 0
            pauses.append(delay)
            waiting.set()
            if boundary == "cancel":
                await release.wait()
            else:
                clock[0] = 5.0
            return await sleep(0)
        return await sleep(delay, *args, **kwargs)

    try:
        await reader.initialize()
        await writer.initialize()
        async with engines[0].begin() as connection:
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
            await connection.execute(text("INSERT INTO ai_values (value) VALUES (0)"))
        monkeypatch.setattr(_database, "monotonic", lambda: clock[0])
        monkeypatch.setattr(_database.asyncio, "sleep", pausing)
        monkeypatch.setattr(AsyncSession, "close", closing)
        competing = asyncio.create_task(writer.run_mutation(hold))
        await asyncio.wait_for(reserved.wait(), 5)
        worker = asyncio.create_task(reader.run_mutation(mutate))
        if boundary == "cancel":
            await asyncio.wait_for(waiting.wait(), 5)
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        else:
            with pytest.raises(OperationalError) as caught:
                await worker
            assert caught.value is seen[0]
        assert len(seen) == 1 and engines[0].sync_engine.pool.checkedout() == 0
        if boundary == "close":
            assert not pauses
        elif boundary == "sleep":
            assert pauses == [pytest.approx(0.001)]
        release.set()
        await competing
        async with engines[0].connect() as connection:
            assert await connection.scalar(text("SELECT value FROM ai_values")) == 1
    finally:
        release.set()
        for task in (worker, competing):
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        await reader.close()
        await writer.close()
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("single_writer", (False, True))
async def test_sqlite_busy_default_never_retries_unknown_commit(single_writer: bool) -> None:
    class Dialect(SQLiteDialect):
        @property
        def single_writer(self) -> bool:
            return single_writer

        def classify_transaction_error(self, error, *, phase, connection_invalidated):
            return SqlTransactionDisposition.COMMIT_UNKNOWN

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    context = SqlStorageContext(engine, async_sessionmaker(engine), Dialect())
    original = sqlite3.OperationalError("database is locked")
    original.sqlite_errorcode = sqlite3.SQLITE_BUSY
    error = OperationalError("COMMIT", None, original)
    attempts = []

    def fail(connection) -> None:
        raise error

    async def mutate(session: AsyncSession) -> None:
        attempts.append(session)
        await session.execute(text("INSERT INTO ai_values (value) VALUES (1)"))

    try:
        await context.initialize()
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
        event.listen(engine.sync_engine, "commit", fail)
        with pytest.raises(AIError) as caught:
            await context.run_mutation(mutate, retry_limit=None)
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert caught.value.__cause__ is error and len(attempts) == 1
    finally:
        event.remove(engine.sync_engine, "commit", fail)
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_custom_single_writer_dialect_keeps_default_attempt_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dialect(SQLiteDialect):
        @property
        def name(self) -> str:
            return "application"

        @property
        def single_writer(self) -> bool:
            return True

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'custom.db'}")
    context = SqlStorageContext(engine, async_sessionmaker(engine), Dialect())
    original = sqlite3.OperationalError("database is locked")
    original.sqlite_errorcode = sqlite3.SQLITE_BUSY
    error = OperationalError("UPDATE", None, original)
    attempts, clock, pauses = [], [0.0], []
    sleep = asyncio.sleep

    async def mutate(session: AsyncSession) -> None:
        attempts.append(session)
        raise error

    async def pause(delay, *args, **kwargs):
        if delay > 0:
            assert engine.sync_engine.pool.checkedout() == 0
            pauses.append(delay)
            clock[0] += 1
            return await sleep(0)
        return await sleep(delay, *args, **kwargs)

    try:
        await context.initialize()
        monkeypatch.setattr(_database, "monotonic", lambda: clock[0])
        monkeypatch.setattr(_database.asyncio, "sleep", pause)
        with pytest.raises(OperationalError) as caught:
            await context.run_mutation(mutate)
        assert caught.value is error and len(attempts) == 8 and len(pauses) == 7
    finally:
        await context.close()
        await engine.dispose()
