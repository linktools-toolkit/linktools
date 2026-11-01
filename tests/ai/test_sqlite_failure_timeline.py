#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure-only lifecycle evidence cannot change or disclose a SQL operation."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, PendingRollbackError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.storage import SqlStorageContext, create_sql_storage_context

from ._sqlite_failure_timeline import _SqliteTimeline, sqlite_failure_timeline


@pytest.mark.asyncio
async def test_sqlite_timeline_preserves_error_and_omits_statement_parameters(tmp_path, capsys):
    failure = RuntimeError("private failure message")
    secret = "private SQL parameter"

    @sqlite_failure_timeline
    async def exercise(*, backend, producer_count):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'timeline.db'}")
        context = create_sql_storage_context(engine)
        try:
            await context.initialize()
            async with context.sessions() as session:
                async with session.begin():
                    result = await context.run_statement(session.execute(text("SELECT :payload"), {"payload": secret}))
                    assert result.scalar_one() == secret
                with pytest.raises(OperationalError):
                    await context.run_statement(session.execute(text("SELECT :payload FROM missing_table"),
                                                                {"payload": secret}))
            raise failure
        finally:
            await context.close()
            await engine.dispose()

    original = SqlStorageContext.run_statement
    with pytest.raises(RuntimeError) as caught:
        await exercise(backend="sqlite", producer_count=2)
    assert caught.value is failure
    notes = "\n".join(failure.__notes__)
    assert secret not in notes and str(failure) not in notes and ":payload" not in notes
    for kind in ("sql.begin", "sql.end", "sql.error", "session.begin", "session.close.begin", "session.close.end",
                 "do_commit.begin", "do_commit.end", "do_rollback.begin", "do_rollback.end", "pool.out", "pool.in"):
        assert kind in notes
    assert "'verb': 'SELECT'" in notes
    assert SqlStorageContext.run_statement is original
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_sqlite_timeline_success_is_silent_and_other_cases_are_untouched(monkeypatch, capsys):
    result = object()

    @sqlite_failure_timeline
    async def exercise(*, backend, producer_count):
        return result

    assert await exercise(backend="sqlite", producer_count=2) is result

    def forbidden(*args):
        raise AssertionError("unselected case installed SQL diagnostics")

    monkeypatch.setattr(_SqliteTimeline, "install", forbidden)
    assert await exercise(backend="sqlite", producer_count=1) is result
    assert await exercise(backend="filesystem", producer_count=1) is result
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_sqlite_timeline_keeps_original_cancellation_and_reports_truncation():
    cancellation = asyncio.CancelledError("caller cancellation")

    @sqlite_failure_timeline
    async def exercise(*, backend, producer_count):
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        await exercise(backend="sqlite", producer_count=2)
    assert caught.value is cancellation and cancellation.args == ("caller cancellation",)
    timeline = _SqliteTimeline(capacity=2)
    for i in range(3):
        timeline.emit("synthetic", connection=i)
    assert len(timeline.events) == 2 and timeline.dropped == 1
    assert "dropped=1" in timeline.format() and "'connection': 0" not in timeline.format()


@pytest.mark.asyncio
async def test_sqlite_timeline_error_never_revalidates_an_invalidated_connection(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'invalidated.db'}")
    timeline = _SqliteTimeline()
    original = sqlite3.OperationalError("private driver message")
    try:
        with pytest.MonkeyPatch.context() as patches:
            timeline.install(patches)
            try:
                failed = next(listener for _target, name, listener in timeline.listeners if name == "handle_error")
                async with engine.connect() as connection:
                    await connection.execute(text("SELECT 1"))
                    await connection.invalidate()
                    with pytest.raises(PendingRollbackError):
                        _ = connection.sync_connection.connection
                    for execution in (SimpleNamespace(_sqlite_failure_connection_id=42), None):
                        failed(SimpleNamespace(dialect=engine.dialect, connection=connection.sync_connection,
                                               execution_context=execution, original_exception=original))
                errors = [row[-1] for row in timeline.events if row[1] == "sql.error"]
                assert errors[-2:] == [
                    {"connection": 42, "error": "OperationalError", "code": None},
                    {"connection": None, "error": "OperationalError", "code": None},
                ]
                assert str(original) not in timeline.format()
            finally:
                timeline.remove()
    finally:
        await engine.dispose()
