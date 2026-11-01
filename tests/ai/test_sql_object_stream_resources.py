#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded SQL object reads and ownership of cancelled stream cleanup."""

import asyncio
import errno
import hashlib
import tracemalloc
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiosqlite import Cursor
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncResult, AsyncSession, create_async_engine

from linktools.ai.storage import SqlObjectStore, build_object_sql_metadata, create_sql_storage_context
from linktools.ai.storage import _object_sql

from .test_storage_lifecycle_ownership import _StagingFiles

pytestmark = pytest.mark.asyncio
_MIB = 1024 * 1024


async def _parts(chunk: bytes, count: int) -> AsyncIterator[bytes]:
    for _ in range(count):
        yield chunk


async def _seed(store: SqlObjectStore, key: str, count: int) -> str:
    chunk = b"x" * _MIB
    digest = hashlib.sha256()
    for _ in range(count):
        digest.update(chunk)
    expected = digest.hexdigest()
    await store.put(key, _parts(chunk, count), expected_size=count * _MIB, expected_digest=expected)
    return expected


async def test_sqlite_object_stream_python_allocations_stay_bounded_with_object_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.initialize()
    async with engine.begin() as connection:
        await connection.run_sync(build_object_sql_metadata().create_all)
    store = SqlObjectStore.from_context(context)
    expected = {count: await _seed(store, str(count), count) for count in (8, 64)}
    actual_query = None
    largest_batch = 0
    original_fetchmany = Cursor.fetchmany

    def capture(_connection, _cursor, statement, parameters, _execution_context, _many) -> None:
        nonlocal actual_query
        if actual_query is None and statement.startswith("SELECT") and "ai_object_chunks.content" in statement:
            actual_query = (statement, parameters)

    async def fetchmany(cursor: Cursor, size=None):
        nonlocal largest_batch
        rows = await original_fetchmany(cursor, size)
        names = [column[0] for column in cursor.description or ()]
        if "content" in names:
            position = names.index("content")
            largest_batch = max(largest_batch, sum(len(row[position]) for row in rows if row[position] is not None))
        return rows

    monkeypatch.setattr(Cursor, "fetchmany", fetchmany)
    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    measurements = []
    try:
        warmup = store.open("8")
        try:
            async for chunk in warmup:
                del chunk
        finally:
            await warmup.aclose()
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert actual_query is not None
        async with engine.connect() as connection:
            plan = await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + actual_query[0], actual_query[1])
            print("Object stream query plan:", [row[-1] for row in plan])

        for count in (8, 64):
            largest_batch = 0
            size = 0
            digest = hashlib.sha256()
            tracemalloc.start()
            try:
                stream = store.open(str(count))
                try:
                    async for chunk in stream:
                        size += len(chunk)
                        digest.update(chunk)
                        del chunk
                finally:
                    await stream.aclose()
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert size == count * _MIB and digest.hexdigest() == expected[count]
            measurements.append({"object_bytes": size, "python_peak": peak, "largest_driver_batch_bytes": largest_batch})
        print("Object stream Python allocation measurements:", measurements)
        small, large = measurements
        assert large["python_peak"] <= 16 * _MIB, measurements
        assert large["python_peak"] - small["python_peak"] <= 8 * _MIB, measurements
        assert engine.sync_engine.pool.checkedout() == 0
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", capture):
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await context.close()


@pytest.mark.parametrize(("operation", "phase"), (
    ("open", "result"), ("open", "session"),
    ("stat", "session"), ("list", "session"), ("integrity", "session"),
))
async def test_sqlite_object_cleanup_settles_after_repeated_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    phase: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.initialize()
    async with engine.begin() as connection:
        await connection.run_sync(build_object_sql_metadata().create_all)
    store = SqlObjectStore.from_context(context)
    await _seed(store, "source", 8)
    acquired, release_acquire = asyncio.Event(), asyncio.Event()
    closing, release_close, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sessions: list[AsyncSession] = []
    results: list[AsyncResult] = []
    original_stream = AsyncSession.stream
    original_execute = AsyncSession.execute
    original_result_close = AsyncResult.close
    original_session_close = AsyncSession.close

    async def stream(session: AsyncSession, *args, **kwargs):
        result = await original_stream(session, *args, **kwargs)
        if not acquired.is_set():
            sessions.append(session)
            results.append(result)
            acquired.set()
            await release_acquire.wait()
        return result

    async def execute(session: AsyncSession, *args, **kwargs):
        result = await original_execute(session, *args, **kwargs)
        if not acquired.is_set():
            sessions.append(session)
            acquired.set()
            await release_acquire.wait()
        return result

    async def result_close(result: AsyncResult) -> None:
        selected = phase == "result" and results and result is results[0]
        if selected:
            closing.set()
            await release_close.wait()
        await original_result_close(result)
        if selected:
            closed.set()

    async def session_close(session: AsyncSession) -> None:
        selected = phase == "session" and sessions and session is sessions[0]
        if selected:
            closing.set()
            await release_close.wait()
        await original_session_close(session)
        if selected:
            closed.set()

    monkeypatch.setattr(AsyncSession, "stream", stream)
    monkeypatch.setattr(AsyncSession, "execute", execute)
    monkeypatch.setattr(AsyncResult, "close", result_close)
    monkeypatch.setattr(AsyncSession, "close", session_close)
    source = None
    if operation == "open":
        source = store.open("source")
        pending = anext(source)
    elif operation == "list":
        source = store.list_objects()
        pending = anext(source)
    elif operation == "stat":
        pending = store.stat("source")
    else:
        pending = store.validate_integrity()
    reader = asyncio.create_task(pending)
    try:
        await asyncio.wait_for(acquired.wait(), 5)
        reader.cancel("stop source read")
        release_acquire.set()
        await asyncio.wait_for(closing.wait(), 5)
        reader.cancel("stop source read again")
        done, _ = await asyncio.wait({reader}, timeout=0.05)
        print("Cleanup before release:", {"operation": operation, "phase": phase, "caller_done": reader in done,
                                           "closed": closed.is_set(), "checked_out": engine.sync_engine.pool.checkedout()})
        assert reader not in done, "caller left before its real stream cleanup completed"
        assert not closed.is_set()
        release_close.set()
        with pytest.raises(asyncio.CancelledError) as retained:
            await asyncio.wait_for(reader, 5)
        assert closed.is_set()
        assert engine.sync_engine.pool.checkedout() == 0
        expected = await _seed(store, "writer", 1)
        info = await store.stat("writer")
        assert info is not None and info.digest == expected and info.size == _MIB
        assert isinstance(retained.value, asyncio.CancelledError)
    finally:
        release_acquire.set()
        release_close.set()
        await asyncio.gather(reader, return_exceptions=True)
        if source is not None:
            await source.aclose()
        for session in sessions:
            await original_session_close(session)
        assert engine.sync_engine.pool.checkedout() == 0
        print("Cleanup resources after test:", {"operation": operation, "phase": phase, "checked_out": engine.sync_engine.pool.checkedout()})
        await context.close()


async def test_wal_large_object_staging_allows_other_object_mutations_before_file_write_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.initialize()
    async with engine.connect() as connection:
        mode = (await connection.exec_driver_sql("PRAGMA journal_mode = WAL")).scalar_one()
        timeout = (await connection.exec_driver_sql("PRAGMA busy_timeout")).scalar_one()
    assert mode == "wal"
    async with engine.begin() as connection:
        await connection.run_sync(build_object_sql_metadata().create_all)
    store = SqlObjectStore.from_context(context)
    expected = await _seed(store, "source", 8)
    files = _StagingFiles(monkeypatch)

    async def consume() -> tuple[int, str]:
        source = store.open("source")
        size, digest = 0, hashlib.sha256()
        try:
            async for value in source:
                size += len(value)
                digest.update(value)
        finally:
            await source.aclose()
        return size, digest.hexdigest()

    reader = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(files.entered.wait(), 5)
        value = b"replacement"
        replacement_digest = hashlib.sha256(value).hexdigest()
        await asyncio.wait_for(store.put("writer", _parts(value, 1), expected_size=len(value), expected_digest=replacement_digest), 15)
        assert await asyncio.wait_for(store.delete_object("source", expected_digest=expected), 15)
        await asyncio.wait_for(store.put("source", _parts(value, 1), expected_size=len(value), expected_digest=replacement_digest), 15)
        assert not files.release.is_set() and not reader.done()
        current = await store.stat("source")
        assert current is not None and current.digest == replacement_digest
        print("Paused large reader mutations:", {"journal_mode": mode, "busy_timeout": timeout,
                                                  "put_delete_reput_committed_before_release": True})
        files.release.set()
        assert await asyncio.wait_for(reader, 5) == (8 * _MIB, expected)
        assert b"".join([part async for part in store.open("source")]) == value
        assert all(handle.closed for handle in files.handles)
        assert engine.sync_engine.pool.checkedout() == 0
    finally:
        files.release.set()
        await asyncio.gather(reader, return_exceptions=True)
        await context.close()


async def test_rollback_large_object_commit_finishes_after_repeated_cancelled_stage_write_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.initialize()
    async with engine.connect() as connection:
        mode = (await connection.exec_driver_sql("PRAGMA journal_mode = DELETE")).scalar_one()
        timeout = (await connection.exec_driver_sql("PRAGMA busy_timeout")).scalar_one()
    assert mode == "delete"
    async with engine.begin() as connection:
        await connection.run_sync(build_object_sql_metadata().create_all)
    store = SqlObjectStore.from_context(context)
    await _seed(store, "source", 8)
    files = _StagingFiles(monkeypatch)
    commit_entered = asyncio.Event()
    original_commit = engine.sync_engine.dialect.do_commit

    def commit(connection) -> None:
        commit_entered.set()
        original_commit(connection)

    async def consume() -> None:
        source = store.open("source")
        try:
            async for _value in source:
                pass
        finally:
            await source.aclose()

    monkeypatch.setattr(engine.sync_engine.dialect, "do_commit", commit)
    reader = asyncio.create_task(consume())
    writer = None
    try:
        await asyncio.wait_for(files.entered.wait(), 5)
        value = b"writer"
        digest = hashlib.sha256(value).hexdigest()
        writer = asyncio.create_task(store.put("writer", _parts(value, 1), expected_size=len(value), expected_digest=digest))
        await asyncio.wait_for(commit_entered.wait(), 5)
        reader.cancel("stop staged read")
        await asyncio.sleep(0)
        reader.cancel("stop staged read again")
        done, _ = await asyncio.wait({reader, writer}, timeout=0.05)
        assert reader not in done and writer not in done
        assert all(not handle.closed for handle in files.handles)
        files.release.set()
        with pytest.raises(asyncio.CancelledError) as retained:
            await asyncio.wait_for(reader, 5)
        info = await asyncio.wait_for(writer, 15)
        assert info.digest == digest
        assert b"".join([part async for part in store.open("writer")]) == value
        assert all(handle.closed for handle in files.handles)
        assert engine.sync_engine.pool.checkedout() == 0
        assert isinstance(retained.value, asyncio.CancelledError)
        print("Released large reader mutation:", {"journal_mode": mode, "busy_timeout": timeout,
                                                   "commit_entered_before_release": True, "commit_completed_after_release": True})
    finally:
        files.release.set()
        await asyncio.gather(reader, *(() if writer is None else (writer,)), return_exceptions=True)
        await context.close()


@pytest.mark.parametrize("operation", ("write", "seek", "read", "close"))
async def test_sqlite_object_file_failure_preserves_error_and_releases_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}")
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.initialize()
    async with engine.begin() as connection:
        await connection.run_sync(build_object_sql_metadata().create_all)
    store = SqlObjectStore.from_context(context)
    await _seed(store, "source", 8 if operation == "write" else 1)
    original = _object_sql.tempfile.TemporaryFile
    failure = OSError(errno.ENOSPC if operation == "write" else errno.EIO, "staging file I/O failed")
    handles = []

    class FailingFile:
        def __init__(self, handle) -> None:
            self.handle = handle

        def write(self, value: bytes) -> int:
            if operation == "write":
                raise failure
            return self.handle.write(value)

        def seek(self, offset: int) -> int:
            if operation == "seek":
                raise failure
            return self.handle.seek(offset)

        def read(self, size: int) -> bytes:
            if operation == "read":
                raise failure
            return self.handle.read(size)

        def close(self) -> None:
            self.handle.close()
            if operation == "close":
                raise failure

    def temporary_file() -> FailingFile:
        handle = original()
        handles.append(handle)
        return FailingFile(handle)

    monkeypatch.setattr(_object_sql.tempfile, "TemporaryFile", temporary_file)
    source = store.open("source")
    try:
        with pytest.raises(OSError) as retained:
            async for _part in source:
                pass
        assert retained.value is failure
        assert all(handle.closed for handle in handles)
        assert engine.sync_engine.pool.checkedout() == 0
        expected = await _seed(store, "writer", 1)
        info = await store.stat("writer")
        assert info is not None and info.digest == expected
        print("File failure resources:", {"operation": operation, "checked_out": engine.sync_engine.pool.checkedout(),
                                           "closed_files": all(handle.closed for handle in handles)})
    finally:
        await source.aclose()
        await context.close()
