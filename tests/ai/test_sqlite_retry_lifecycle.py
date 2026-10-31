#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retryable SQLite conflicts yield with neither a snapshot nor a checked-out session."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.storage import create_sql_storage_context
from linktools.ai.storage import _database


@pytest.mark.asyncio
async def test_sqlite_retry_leaves_lock_free_gap_for_independent_writer(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{tmp_path / 'retry.db'}"
    engines = (create_async_engine(url), create_async_engine(url))
    reader, writer = (create_sql_storage_context(engine) for engine in engines)
    reserved, release, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pauses, observed = [], []
    reader_task = writer_task = None
    sleep = asyncio.sleep

    async def competing_write(session: AsyncSession):
        await session.execute(text("UPDATE ai_values SET value = 1"))
        reserved.set()
        await release.wait()

    async def run_writer():
        await writer.run_mutation(competing_write)
        committed.set()

    async def increment(session: AsyncSession):
        value = await session.scalar(text("SELECT value FROM ai_values"))
        observed.append(value)
        await session.execute(text("UPDATE ai_values SET value = value + 1"))
        return value + 1

    async def wait_between_attempts(delay, *args, **kwargs):
        if asyncio.current_task() is reader_task and delay > 0:
            # The competing commit requires the reader's SHARED lock to be gone.
            assert engines[0].sync_engine.pool.checkedout() == 0
            pauses.append(delay)
            release.set()
            await asyncio.wait_for(committed.wait(), 5)
            return await sleep(0)
        return await sleep(delay, *args, **kwargs)

    try:
        await reader.initialize()
        await writer.initialize()
        async with engines[0].begin() as connection:
            assert await connection.scalar(text("PRAGMA journal_mode")) == "delete"
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
            await connection.execute(text("INSERT INTO ai_values (value) VALUES (0)"))
        monkeypatch.setattr(_database.asyncio, "sleep", wait_between_attempts)
        writer_task = asyncio.create_task(run_writer())
        await asyncio.wait_for(reserved.wait(), 5)
        reader_task = asyncio.create_task(reader.run_mutation(increment))
        assert await asyncio.wait_for(reader_task, 5) == 2
        assert observed == [0, 1] and pauses
        await writer_task
        async with engines[0].connect() as connection:
            assert await connection.scalar(text("SELECT value FROM ai_values")) == 2
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
async def test_sqlite_cancellation_during_retry_has_released_transaction(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from linktools.ai.storage import SQLiteDialect, SqlStorageContext, SqlTransactionDisposition

    class RetryBody(Exception):
        pass

    class Dialect(SQLiteDialect):
        def classify_transaction_error(self, error, *, phase, connection_invalidated):
            if isinstance(error, RetryBody):
                return SqlTransactionDisposition.RETRYABLE_ABORTED
            return super().classify_transaction_error(error, phase=phase, connection_invalidated=connection_invalidated)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'cancel.db'}")
    context = SqlStorageContext(engine, async_sessionmaker(engine), Dialect())
    waiting, blocked = asyncio.Event(), asyncio.Event()
    attempts = 0
    worker = None
    sleep = asyncio.sleep

    async def mutate(session):
        nonlocal attempts
        attempts += 1
        await session.execute(text("INSERT INTO ai_values (value) VALUES (1)"))
        raise RetryBody()

    async def wait_between_attempts(delay, *args, **kwargs):
        if asyncio.current_task() is worker and delay > 0:
            assert engine.sync_engine.pool.checkedout() == 0
            waiting.set()
            await blocked.wait()
        return await sleep(delay, *args, **kwargs)

    try:
        await context.initialize()
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
        monkeypatch.setattr(_database.asyncio, "sleep", wait_between_attempts)
        worker = asyncio.create_task(context.run_mutation(mutate))
        await asyncio.wait_for(waiting.wait(), 5)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert attempts == 1 and engine.sync_engine.pool.checkedout() == 0
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_values")) == 0
    finally:
        blocked.set()
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("single_writer", (True, False))
@pytest.mark.parametrize("failure_phase", ("body", "commit_aborted", "commit_unknown", "exhausted"))
async def test_retry_delay_preserves_dialect_dispositions_and_attempt_budget(
    tmp_path, monkeypatch, single_writer, failure_phase,
):
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.storage import SQLiteDialect, SqlStorageContext, SqlTransactionDisposition, SqlTransactionPhase

    class ClassifiedFailure(Exception):
        pass

    class Dialect(SQLiteDialect):
        @property
        def single_writer(self):
            return single_writer

        def classify_transaction_error(self, error, *, phase, connection_invalidated):
            if isinstance(error, ClassifiedFailure):
                if phase is SqlTransactionPhase.COMMIT and failure_phase == "commit_unknown":
                    return SqlTransactionDisposition.COMMIT_UNKNOWN
                return SqlTransactionDisposition.RETRYABLE_ABORTED
            return super().classify_transaction_error(error, phase=phase, connection_invalidated=connection_invalidated)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'dispositions.db'}")
    context = SqlStorageContext(engine, async_sessionmaker(engine), Dialect())
    failure = ClassifiedFailure("classified abort")
    sleep = asyncio.sleep
    owner = asyncio.current_task()
    attempts, commits = 0, 0
    pauses = []

    async def mutate(session):
        nonlocal attempts
        attempts += 1
        await session.execute(text("INSERT INTO ai_values (value) VALUES (1)"))
        if failure_phase == "exhausted" or failure_phase == "body" and attempts == 1:
            raise failure
        return 42

    def commit(connection):
        nonlocal commits
        commits += 1
        if failure_phase.startswith("commit") and commits == 1:
            raise failure

    async def wait_between_attempts(delay, *args, **kwargs):
        if asyncio.current_task() is owner and delay > 0:
            assert engine.sync_engine.pool.checkedout() == 0
            pauses.append(delay)
            return await sleep(0)
        return await sleep(delay, *args, **kwargs)

    try:
        await context.initialize()
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
        event.listen(engine.sync_engine, "commit", commit)
        monkeypatch.setattr(_database.asyncio, "sleep", wait_between_attempts)
        if failure_phase == "commit_unknown":
            with pytest.raises(AIError) as unknown:
                await context.run_mutation(mutate, retry_limit=3)
            assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert unknown.value.__cause__ is failure and attempts == 1 and not pauses
        elif failure_phase == "exhausted":
            with pytest.raises(ClassifiedFailure) as exhausted:
                await context.run_mutation(mutate)
            assert exhausted.value is failure and attempts == 8
        else:
            assert await context.run_mutation(mutate, retry_limit=3) == 42
            assert attempts == 2
        assert len(pauses) == (attempts - 1 if single_writer else 0)
        assert all(0 < delay <= 0.05 for delay in pauses)
        if single_writer and failure_phase == "exhausted":
            assert pauses == [0.005, 0.01, 0.02, 0.04, 0.05, 0.05, 0.05]
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_values")) == (
                0 if failure_phase in {"commit_unknown", "exhausted"} else 1
            )
    finally:
        event.remove(engine.sync_engine, "commit", commit)
        await context.close()
        await engine.dispose()
