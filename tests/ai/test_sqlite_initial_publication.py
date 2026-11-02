#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local SQLite initialization never replaces a concurrently published source."""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, Table

from linktools.ai.persistence import _resource_scope


def _metadata() -> MetaData:
    metadata = MetaData()
    Table("initialized", metadata, Column("id", Integer, primary_key=True))
    return metadata


@pytest.mark.asyncio
@pytest.mark.parametrize('local', [False, True])
async def test_initial_publication_preserves_concurrent_sqlite_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local: bool,
) -> None:
    target = tmp_path / "state.db"
    original = _resource_scope._sync_file

    def sync_file(temporary: Path) -> None:
        original(temporary)
        with sqlite3.connect(temporary) as connection:
            assert connection.execute('PRAGMA application_id').fetchone() == (
                _resource_scope.SqlStorageContext.LOCAL_SQLITE_APPLICATION_ID if local else 0,
            )
        with sqlite3.connect(target) as connection:
            connection.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
            connection.execute("INSERT INTO sentinel VALUES ('other-writer')")

    monkeypatch.setattr(_resource_scope, "_sync_file", sync_file)
    cleanups = []
    try:
        with pytest.raises(FileExistsError):
            await _resource_scope._provision_sqlite_database(
                target, _metadata(), register_cleanup=cleanups.append, local=local,
            )
        with sqlite3.connect(target) as connection:
            assert connection.execute("SELECT value FROM sentinel").fetchone() == ("other-writer",)
            assert connection.execute("SELECT name FROM sqlite_master WHERE name='initialized'").fetchone() is None
            assert connection.execute('PRAGMA application_id').fetchone() == (0,)
        assert not tuple(tmp_path.glob(".*.init-*"))
    finally:
        for cleanup in reversed(cleanups):
            await cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize('local', [False, True])
async def test_initial_publication_completes_and_removes_temporary_name(tmp_path: Path, local: bool) -> None:
    target = tmp_path / "state.db"
    cleanups = []
    await _resource_scope._provision_sqlite_database(target, _metadata(), register_cleanup=cleanups.append, local=local)
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM initialized").fetchone() == (0,)
        assert connection.execute('PRAGMA application_id').fetchone() == (
            _resource_scope.SqlStorageContext.LOCAL_SQLITE_APPLICATION_ID if local else 0,
        )
    assert not tuple(tmp_path.glob(".*.init-*"))
    for cleanup in cleanups:
        await cleanup()
    assert target.is_file()


@pytest.mark.asyncio
async def test_initial_publication_file_sync_failure_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.db"
    failure = OSError("file sync failed")

    def fail(path: Path) -> None:
        raise failure

    monkeypatch.setattr(_resource_scope, "_sync_file", fail)
    with pytest.raises(OSError) as caught:
        await _resource_scope._provision_sqlite_database(target, _metadata(), register_cleanup=lambda action: None)
    assert caught.value is failure
    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.init-*"))


@pytest.mark.asyncio
async def test_initial_publication_directory_sync_failure_preserves_published_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    target = tmp_path / "state.db"
    failure = OSError("directory sync failed")

    def fail(path: Path) -> None:
        raise failure

    monkeypatch.setattr(_resource_scope, "sync_directory", fail)
    cleanups = []
    with pytest.raises(AIError) as caught:
        await _resource_scope._provision_sqlite_database(target, _metadata(), register_cleanup=cleanups.append)
    assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert caught.value.__cause__ is failure
    for cleanup in cleanups:
        await cleanup()
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM initialized").fetchone() == (0,)
    assert not tuple(tmp_path.glob(".*.init-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize('local', [False, True])
async def test_initial_publication_temporary_unlink_failure_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local: bool,
) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.storage import create_sql_storage_context

    target = tmp_path / "state.db"
    original = Path.unlink
    blocked = True

    def unlink(path: Path, missing_ok: bool = False) -> None:
        if blocked and ".init-" in path.name:
            raise OSError("temporary unlink blocked")
        original(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    cleanups = []
    context = create_sql_storage_context(create_async_engine(f'sqlite+aiosqlite:///{target}'), owns_engine=True)
    try:
        with pytest.raises(OSError, match="temporary unlink blocked"):
            await _resource_scope._provision_sqlite_database(target, _metadata(), register_cleanup=cleanups.append, local=local)
        assert target.is_file() and tuple(tmp_path.glob(".*.init-*"))
        if local:
            assert target.stat().st_nlink == 2
            await context.acquire_local_sqlite(target)
            with pytest.raises(AIError) as unclassified_alias:
                await context.require_write_access()
            assert unclassified_alias.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        blocked = False
        for cleanup in cleanups:
            await cleanup()
        assert not tuple(tmp_path.glob(".*.init-*"))
        if local:
            await context.require_write_access()
            assert context.exclusive_writer and target.stat().st_nlink == 1
        with sqlite3.connect(target) as connection:
            assert connection.execute("SELECT COUNT(*) FROM initialized").fetchone() == (0,)
    finally:
        blocked = False
        for cleanup in cleanups:
            await cleanup()
        await context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("file_sync", "publication"))
async def test_initial_publication_cancellation_waits_for_owned_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    import asyncio
    import threading

    target = tmp_path / "state.db"
    entered, release = threading.Event(), threading.Event()
    original_sync, original_link = _resource_scope._sync_file, _resource_scope.os.link

    def wait() -> None:
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release publication thread")

    def sync(path: Path) -> None:
        wait()
        original_sync(path)

    def link(source: Path, destination: Path) -> None:
        wait()
        original_link(source, destination)

    monkeypatch.setattr(_resource_scope, "_sync_file", sync if boundary == "file_sync" else original_sync)
    monkeypatch.setattr(_resource_scope.os, "link", link if boundary == "publication" else original_link)
    cleanups = []
    task = asyncio.create_task(_resource_scope._provision_sqlite_database(target, _metadata(), register_cleanup=cleanups.append))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert tuple(tmp_path.glob(".*.init-*"))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert target.exists() is (boundary == "publication")
        if target.exists():
            with sqlite3.connect(target) as connection:
                assert connection.execute("SELECT COUNT(*) FROM initialized").fetchone() == (0,)
        assert not tuple(tmp_path.glob(".*.init-*"))
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for cleanup in cleanups:
            await cleanup()


@pytest.mark.asyncio
async def test_initial_publication_link_failure_never_falls_back_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.db"
    failure = OSError("hard links unavailable")

    def fail(source: Path, destination: Path) -> None:
        raise failure

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("publication must not fall back to replacing a target")

    monkeypatch.setattr(_resource_scope.os, "link", fail)
    monkeypatch.setattr(_resource_scope.os, "replace", forbidden)
    with pytest.raises(OSError) as caught:
        await _resource_scope._provision_sqlite_database(target, _metadata(), register_cleanup=lambda action: None)
    assert caught.value is failure
    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.init-*"))
