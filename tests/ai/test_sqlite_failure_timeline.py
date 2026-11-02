#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure-only lifecycle evidence cannot change or disclose a SQL operation."""

import asyncio
import gc
import sqlite3
from types import SimpleNamespace

import pytest
from aiosqlite import Cursor
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, PendingRollbackError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.storage import SqlStorageContext, create_sql_storage_context

from . import _sqlite_failure_timeline as timeline_module
from ._sqlite_failure_timeline import (
    _SqliteTimeline, sqlite_failure_evidence, sqlite_failure_phase, sqlite_failure_timeline,
)


@pytest.mark.asyncio
async def test_sqlite_timeline_preserves_error_and_omits_statement_parameters(tmp_path, capsys, monkeypatch):
    failure = RuntimeError("private failure message")
    secret = "private SQL parameter"

    @sqlite_failure_timeline
    async def exercise(*, backend, producer_count):
        sqlite_failure_phase("model.entered", producer=1)
        gc.collect(0)
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
    original_cursor_init, original_cursor_close = Cursor.__init__, Cursor.close
    original_gc_callbacks = tuple(gc.callbacks)
    original_format = _SqliteTimeline.format

    def settled_format(timeline):
        assert timeline.gc_callback not in gc.callbacks
        assert timeline.loop_handle.cancelled()
        assert timeline_module._active_timeline.get() is None
        count = len(timeline.events)
        gc.collect(0)
        assert len(timeline.events) == count
        return original_format(timeline)

    monkeypatch.setattr(_SqliteTimeline, "format", settled_format)
    with pytest.raises(RuntimeError) as caught:
        await exercise(backend="sqlite", producer_count=2)
    assert caught.value is failure
    notes = "\n".join(failure.__notes__)
    assert secret not in notes and str(failure) not in notes and ":payload" not in notes
    for kind in ("sql.begin", "sql.end", "sql.error", "session.begin", "session.close.begin", "session.close.end",
                 "cursor.open", "cursor.close.begin", "cursor.close.end", "do_commit.begin", "do_commit.end", "do_rollback.begin", "do_rollback.end", "pool.out", "pool.in"):
        assert kind in notes
    assert "'verb': 'SELECT'" in notes
    assert SqlStorageContext.run_statement is original
    assert (Cursor.__init__, Cursor.close) == (original_cursor_init, original_cursor_close)
    assert tuple(gc.callbacks) == original_gc_callbacks
    assert timeline_module._active_timeline.get() is None
    assert "phase.model.entered" in notes and "'producer': 1" in notes
    assert "gc.start" in notes and "gc.stop" in notes and "thread_cpu" in notes
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
    original_gc_callbacks = tuple(gc.callbacks)

    @sqlite_failure_timeline
    async def exercise(*, backend, producer_count):
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        await exercise(backend="sqlite", producer_count=2)
    assert caught.value is cancellation and cancellation.args == ("caller cancellation",)
    assert tuple(gc.callbacks) == original_gc_callbacks
    assert timeline_module._active_timeline.get() is None
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


@pytest.mark.asyncio
async def test_sqlite_timeline_loop_lag_is_bounded_and_cleanup_cancels_observation(monkeypatch):
    timeline = _SqliteTimeline(capacity=2)
    now = timeline.started + 1
    monkeypatch.setattr(timeline_module, "monotonic", lambda: now)
    timeline._loop_tick(now - 0.25)
    handle = timeline.loop_handle
    assert timeline.events[-1][1] == "loop.lag"
    assert timeline.events[-1][-1]["seconds"] == 0.25
    assert "thread_cpu" in timeline.events[-1][-1]
    timeline.remove()
    assert handle.cancelled()
    sqlite_failure_phase("outside.selected.case", producer=1)
    assert len(timeline.events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_sqlite_failure_evidence_keeps_mutation_error_identity_and_native_code(tmp_path, cancelled, capsys):
    cancellation = asyncio.CancelledError("private cancellation detail")
    observed = []

    @sqlite_failure_evidence
    async def exercise():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'mutation.db'}")
        context = create_sql_storage_context(engine, owns_engine=True)
        try:
            await context.initialize()

            async def operation(session):
                if cancelled:
                    raise cancellation
                return await context.run_statement(session.execute(text("SELECT :value FROM private_missing_table"),
                                                                  {"value": "private parameter"}))

            try:
                await context.run_mutation(operation, domain="diagnostic.test")
            except BaseException as error:
                observed.append(error)
                raise
        finally:
            await context.close()

    original_mutation = SqlStorageContext.run_mutation
    with pytest.raises(asyncio.CancelledError if cancelled else OperationalError) as caught:
        await exercise()
    assert caught.value is observed[0]
    if cancelled:
        assert caught.value is cancellation and cancellation.args == ("private cancellation detail",)
    notes = "\n".join(caught.value.__notes__)
    assert "mutation.begin" in notes and "mutation.error" in notes
    assert "'domain': 'diagnostic.test'" in notes
    assert "'code': None" in notes if cancelled else "'code': 1" in notes
    assert f"'invalidated': {None if cancelled else False}" in notes
    for secret in ("private cancellation detail", "private_missing_table", "private parameter", ":value"):
        assert secret not in notes
    assert SqlStorageContext.run_mutation is original_mutation
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_sqlite_failure_snapshot_reports_open_cursor_without_payload(tmp_path):
    import aiosqlite

    timeline = _SqliteTimeline()
    with pytest.MonkeyPatch.context() as patches:
        timeline.install(patches)
        try:
            async with aiosqlite.connect(tmp_path / 'cursor.db') as connection:
                cursor = await connection.execute("SELECT ?", ("private cursor content",))
                try:
                    timeline.failure_snapshot()
                    snapshot = timeline.events[-1][-1]
                    assert snapshot['cursor_count'] == 1
                    assert snapshot['cursors'][0]['connection'] == id(connection)
                    assert snapshot['cursors'][0]['age'] >= 0
                    assert 'private cursor content' not in timeline.format()
                finally:
                    await cursor.close()
                timeline.failure_snapshot()
                assert timeline.events[-1][-1]['cursor_count'] == 0
        finally:
            timeline.remove()


@pytest.mark.asyncio
async def test_sqlite_failure_snapshot_tolerates_gc_during_summary():
    timeline = _SqliteTimeline()

    class CollectOnRead(dict):
        def get(self, key, default=None):
            gc.collect(0)
            return super().get(key, default)

    with pytest.MonkeyPatch.context() as patches:
        timeline.install(patches)
        try:
            timeline.emit("cursor.open", connection=1, cursor=2)
            timestamp, kind, task, origin, fields = timeline.events[-1]
            timeline.events[-1] = (timestamp, kind, task, origin, CollectOnRead(fields))
            timeline.failure_snapshot()
            assert timeline.events[-1][1] == "failure.observed_unclosed"
            assert timeline.events[-1][-1]["cursor_count"] == 1
            assert any(row[1] == "gc.start" for row in timeline.events)
        finally:
            timeline.remove()


@pytest.mark.asyncio
async def test_sqlite_timeline_format_tolerates_gc_during_rendering():
    timeline = _SqliteTimeline()

    class CollectOnRepr(dict):
        def __repr__(self):
            gc.collect(0)
            return super().__repr__()

    with pytest.MonkeyPatch.context() as patches:
        timeline.install(patches)
        try:
            timeline.emit("cursor.open", connection=1, cursor=2)
            timestamp, kind, task, origin, fields = timeline.events[-1]
            timeline.events[-1] = (timestamp, kind, task, origin, CollectOnRepr(fields))
            rendered = timeline.format()
            assert "cursor.open" in rendered
            assert "'connection': 1" in rendered
            assert any(row[1] == "gc.start" for row in timeline.events)
        finally:
            timeline.remove()
