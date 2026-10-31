#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedicated local SQLite files cannot be written through an unowned context."""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import create_sql_storage_context, provision_sql

pytestmark = pytest.mark.asyncio
_LOCAL_PROFILE = 0x4C544C53


def _local_database(path: Path) -> None:
    with sqlite3.connect(path) as database:
        database.execute(f'PRAGMA application_id={_LOCAL_PROFILE}')
        database.execute('CREATE TABLE sentinel (value INTEGER NOT NULL)')
        database.execute('INSERT INTO sentinel VALUES (7)')


@pytest.mark.parametrize('entry', ['mutation', 'provision'])
async def test_plain_sql_entry_cannot_write_dedicated_local_file(tmp_path: Path, entry: str) -> None:
    path = tmp_path / 'local.db'
    _local_database(path)
    engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    context = create_sql_storage_context(engine)
    try:
        await context.initialize()
        with pytest.raises(AIError) as rejected:
            if entry == 'mutation':
                await context.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (8)')))
            else:
                metadata = MetaData()
                Table('unowned', metadata, Column('id', Integer, primary_key=True))
                await provision_sql(engine, metadata)
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        async with engine.connect() as connection:
            assert (await connection.execute(text('SELECT value FROM sentinel'))).scalars().all() == [7]
            assert (await connection.execute(text("SELECT name FROM sqlite_master WHERE name='unowned'"))).first() is None
    finally:
        await context.close()
        await engine.dispose()


async def test_dedicated_permission_is_context_owned_not_engine_owned(tmp_path: Path) -> None:
    from linktools.ai.storage._sqlite_owner import _SqliteFileOwner as SqliteFileOwner

    path = tmp_path / 'local.db'
    _local_database(path)
    engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    owned = create_sql_storage_context(engine, owns_engine=True)
    await owned.acquire_local_sqlite(path)
    owner = owned._sqlite_owner
    assert owner is not None
    other = create_sql_storage_context(engine)
    try:
        await owned.initialize()
        await owned.require_write_access()
        assert owner.active
        await owned.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (8)')))
        with pytest.raises(AIError) as rejected:
            await other.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (9)')))
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        async with engine.connect() as connection:
            assert (await connection.execute(text('SELECT value FROM sentinel'))).scalars().all() == [7, 8]
        contender = SqliteFileOwner(path)
        with pytest.raises(AIError) as rejected:
            await contender.acquire()
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        await contender.close()
        await owner.close()
        with pytest.raises(AIError) as rejected:
            await owned.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (10)')))
        assert rejected.value.code is ErrorCode.STORAGE_CLOSED
    finally:
        await owned.close()
        await other.close()
        await engine.dispose()
        await owner.close()


async def test_plain_shared_sqlite_writers_still_share_one_source(tmp_path: Path) -> None:
    path = tmp_path / 'shared.db'
    with sqlite3.connect(path) as database:
        database.execute('CREATE TABLE sentinel (value INTEGER NOT NULL)')
    first_engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    second_engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    first, second = create_sql_storage_context(first_engine), create_sql_storage_context(second_engine)
    try:
        for value, context in enumerate((first, second)):
            await context.initialize()
            await context.require_write_access()
            await context.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (:value)'), {'value': value}))
        async with first_engine.connect() as connection:
            assert (await connection.execute(text('SELECT value FROM sentinel'))).scalars().all() == [0, 1]
    finally:
        await first.close()
        await second.close()
        await first_engine.dispose()
        await second_engine.dispose()


async def test_sqlite_new_connections_revalidate_creator_source(tmp_path: Path) -> None:
    import aiosqlite
    from sqlalchemy.pool import NullPool

    first, second = tmp_path / 'first.db', tmp_path / 'second.db'
    for path in (first, second):
        _local_database(path)
    remaining = iter((first, second))

    async def connect():
        return await aiosqlite.connect(next(remaining))

    engine = create_async_engine('sqlite+aiosqlite:///:memory:', async_creator=connect, poolclass=NullPool)
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(first)
    owner = context._sqlite_owner
    assert owner is not None
    try:
        await context.initialize()
        await context.require_write_access()
        assert owner.active
        with pytest.raises(AIError) as rejected:
            await context.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (8)')))
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        for path in (first, second):
            with sqlite3.connect(path) as database:
                assert database.execute('SELECT value FROM sentinel').fetchall() == [(7,)]
    finally:
        await context.close()
        await engine.dispose()
        await owner.close()


async def test_sqlite_profile_reads_are_per_physical_connection_not_per_mutation(tmp_path: Path) -> None:
    from sqlalchemy import event

    path = tmp_path / 'local.db'
    _local_database(path)
    engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(path)
    owner = context._sqlite_owner
    assert owner is not None
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture)
    try:
        await context.initialize()
        await context.require_write_access()
        assert 'PRAGMA database_list' in statements and 'PRAGMA main.application_id' in statements
        statements.clear()
        await context.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (8)')))
        await context.run_mutation(lambda session: session.execute(text('INSERT INTO sentinel VALUES (9)')))
        assert not any(statement.startswith('PRAGMA') for statement in statements)
    finally:
        await context.close()
        await engine.dispose()
        await owner.close()


@pytest.mark.parametrize('alias', ['symlink', 'hardlink'])
async def test_dedicated_sqlite_aliases_cannot_evade_the_owner(tmp_path: Path, alias: str) -> None:
    import os
    from linktools.ai.storage._sqlite_owner import _SqliteFileOwner as SqliteFileOwner

    path, other_path = tmp_path / 'local.db', tmp_path / 'alias.db'
    _local_database(path)
    if alias == 'symlink':
        other_path.symlink_to(path)
    else:
        os.link(path, other_path)
    engine = create_async_engine(f'sqlite+aiosqlite:///{other_path}')
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(path)
    owner = context._sqlite_owner
    assert owner is not None
    try:
        await context.initialize()
        if alias == 'hardlink':
            with pytest.raises(AIError) as rejected:
                await context.require_write_access()
            assert rejected.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        else:
            await context.require_write_access()
            contender = SqliteFileOwner(other_path)
            with pytest.raises(AIError) as rejected:
                await contender.acquire()
            assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
            await contender.close()
    finally:
        await context.close()
        await engine.dispose()
        await owner.close()


async def test_local_sqlite_cancelled_acquire_settles_before_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import threading
    from linktools.ai.storage._sqlite_owner import _SqliteFileOwner as SqliteFileOwner

    path = tmp_path / 'local.db'
    owner = SqliteFileOwner(path)
    entered, release = threading.Event(), threading.Event()
    original = owner._lock._acquire

    def gated(lock):
        entered.set()
        assert release.wait(5)
        return original(lock)

    monkeypatch.setattr(owner._lock, '_acquire', gated)
    task = asyncio.create_task(owner.acquire())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        contender = SqliteFileOwner(path)
        with pytest.raises(AIError) as rejected:
            await contender.acquire()
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        await contender.close()
        await owner.close()
        following = SqliteFileOwner(path)
        await following.acquire()
        await following.close()
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
        await owner.close()


async def test_local_sqlite_failed_release_keeps_lock_for_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.storage._sqlite_owner import _SqliteFileOwner as SqliteFileOwner

    path = tmp_path / 'local.db'
    owner = SqliteFileOwner(path)
    await owner.acquire()
    lock = owner._lock._lock
    assert lock is not None
    original = lock.release
    failure = OSError('release failed')

    def failed(*args, **kwargs):
        raise failure

    try:
        monkeypatch.setattr(lock, 'release', failed)
        with pytest.raises(OSError) as raised:
            await owner.close()
        assert raised.value is failure
        assert not owner.active
        contender = SqliteFileOwner(path)
        with pytest.raises(AIError) as rejected:
            await contender.acquire()
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        await contender.close()
        monkeypatch.setattr(lock, 'release', original)
        await owner.close()
        following = SqliteFileOwner(path)
        await following.acquire()
        await following.close()
    finally:
        monkeypatch.setattr(lock, 'release', original)
        await owner.close()


async def test_local_context_close_cannot_release_an_inflight_transaction(tmp_path: Path) -> None:
    import asyncio

    path = tmp_path / 'local.db'
    _local_database(path)
    engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(path)
    await context.initialize()
    await context.require_write_access()
    entered, release = asyncio.Event(), asyncio.Event()

    async def write(session):
        await session.execute(text('INSERT INTO sentinel VALUES (8)'))
        entered.set()
        await release.wait()

    writing = asyncio.create_task(context.run_mutation(write))
    contender_engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    contender = create_sql_storage_context(contender_engine, owns_engine=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(AIError) as rejected:
            await context.close()
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert context.exclusive_writer and not context.closed
        with pytest.raises(AIError) as rejected:
            await contender.acquire_local_sqlite(path)
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        release.set()
        await writing
        await context.close()
        await contender.acquire_local_sqlite(path)
        await contender.require_write_access()
        assert contender.exclusive_writer
        async with contender_engine.connect() as connection:
            assert (await connection.execute(text('SELECT value FROM sentinel'))).scalars().all() == [7, 8]
    finally:
        release.set()
        await asyncio.gather(writing, return_exceptions=True)
        await contender.close()
        await context.close()


async def test_local_context_retries_failed_session_cleanup_without_repeating_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'local.db'
    _local_database(path)
    engine = create_async_engine(f'sqlite+aiosqlite:///{path}')
    context = create_sql_storage_context(engine, owns_engine=True)
    await context.acquire_local_sqlite(path)
    await context.initialize()
    sessions = context.sessions
    failure = OSError('session close failed')
    fail = True
    calls = 0

    def created():
        session = sessions()
        original = session.close

        async def close():
            nonlocal fail
            if fail:
                fail = False
                raise failure
            await original()

        monkeypatch.setattr(session, 'close', close)
        return session

    async def write(session):
        nonlocal calls
        calls += 1
        await session.execute(text('INSERT INTO sentinel VALUES (8)'))

    monkeypatch.setattr(context, 'sessions', created)
    try:
        with pytest.raises(OSError) as raised:
            await context.run_mutation(write)
        assert raised.value is failure and calls == 1
        assert context.exclusive_writer
        await context.close()
        assert context.closed and not context.exclusive_writer
        assert calls == 1
        with sqlite3.connect(path) as database:
            assert database.execute('SELECT value FROM sentinel').fetchall() == [(7,), (8,)]
    finally:
        fail = False
        await context.close()


async def test_local_cancelled_release_preserves_cancellation_and_retryable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import threading
    from linktools.ai.storage._sqlite_owner import _SqliteFileOwner

    owner = _SqliteFileOwner(tmp_path / 'local.db')
    await owner.acquire()
    lock = owner._lock._lock
    assert lock is not None
    original = lock.release
    entered, release = threading.Event(), threading.Event()
    failure = OSError('release failed after cancellation')

    def failed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise failure

    monkeypatch.setattr(lock, 'release', failed)
    closing = asyncio.create_task(owner.close())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await closing
        assert cancelled.value.__cause__ is failure
        contender = _SqliteFileOwner(owner.path)
        with pytest.raises(AIError) as rejected:
            await contender.acquire()
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        await contender.close()
        monkeypatch.setattr(lock, 'release', original)
        await owner.close()
    finally:
        release.set()
        if not closing.done():
            await asyncio.gather(closing, return_exceptions=True)
        monkeypatch.setattr(lock, 'release', original)
        await owner.close()
