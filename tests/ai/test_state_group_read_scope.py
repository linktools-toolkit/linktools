#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-store reads share a snapshot without widening task or write authority."""

import asyncio
import hashlib
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._filesystem import FilesystemStateStorageGroup, FilesystemStateStore
from linktools.ai.runtime.state._memory import InMemoryStateStorageGroup, InMemoryStateStore
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.runtime.state._store import (
    StateGroupTransaction, StateStore, StateTransactionNestingError, StoredRecord,
    active_state_scope,
)

pytestmark = pytest.mark.asyncio
_BACKENDS = ("memory", "filesystem", "sql")


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: int) -> StoredRecord:
    return StoredRecord(_digest("record"), None, None, "probe", "record", None,
                        value - 1, None, 0, None, {"value": value})


def _filesystem_stores(root: Path, *, read_only: bool = False) -> tuple[FilesystemStateStore, ...]:
    group = FilesystemStateStorageGroup(
        root, namespace="group-read", tenant_id="tenant", scope_digest="scope", read_only=read_only,
    )
    return tuple(FilesystemStateStore(
        root / domain, namespace="group-read", tenant_id="tenant", runtime_domain=domain, group=group,
    ) for domain in ("conversation", "execution", "step"))


@asynccontextmanager
async def _open_stores(backend: str, root: Path) -> AsyncIterator[tuple[StateStore, ...]]:
    engine = None
    if backend == "memory":
        group = InMemoryStateStorageGroup()
        stores = tuple(InMemoryStateStore(group) for _ in range(3))
    elif backend == "filesystem":
        stores = _filesystem_stores(root)
    else:
        root.mkdir(parents=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'state.db'}", pool_size=1, max_overflow=1)
        await provision_database(engine)
        first = SqlStateStore(engine)
        stores = (first, *(SqlStateStore(
            engine, group=first._storage_group, store_digest=_digest(str(index)),
        ) for index in range(2)))
    for store in stores:
        await store.initialize()
    try:
        yield stores
    finally:
        for store in reversed(stores):
            await store.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_group_read_scope_reuses_enlisted_readonly_transactions(
    backend: str, tmp_path: Path,
) -> None:
    async with _open_stores(backend, tmp_path / backend) as (first, second, omitted):
        group = first.storage_group
        other = InMemoryStateStore()
        await other.initialize()

        async def inspect(scope: StateGroupTransaction) -> str:
            for store in (first, second):
                transaction = scope.transaction(store)

                async def same(actual):
                    assert actual is transaction
                    return await actual.scan_records()

                assert await store.read(same) == ()
                assert await group.read(store, same) == ()
                assert await group.read_many((store, store), lambda nested: same(nested.transaction(store))) == ()
                with pytest.raises(StateTransactionNestingError):
                    await transaction.insert_record(_record(1))
                with pytest.raises(StateTransactionNestingError):
                    await transaction.guard_record(
                        _digest("record"), expected_storage_revision=0,
                    )
                with pytest.raises(StateTransactionNestingError):
                    await transaction.replace_record(
                        _record(2), expected_storage_revision=0,
                    )
                with pytest.raises(StateTransactionNestingError):
                    await transaction.update_record_lease(
                        _digest("record"), expected_storage_revision=0,
                        lease_owner=None, lease_fence=0, lease_expires_at=None,
                    )
                with pytest.raises(StateTransactionNestingError):
                    await transaction.delete_record(
                        _digest("record"), expected_storage_revision=0,
                    )
                with pytest.raises(StateTransactionNestingError):
                    await store.mutate(lambda tx: tx.insert_record(_record(1)))
                with pytest.raises(StateTransactionNestingError):
                    await group.mutate((store,), lambda tx: tx.transaction(store).insert_record(_record(1)))
                with pytest.raises(StateTransactionNestingError, match="child task"):
                    await asyncio.create_task(store.read(same))
                with pytest.raises(StateTransactionNestingError, match="child task"):
                    await asyncio.create_task(group.read_many((store,), inspect))
            with pytest.raises(StateTransactionNestingError, match="not enlisted"):
                await group.read_many((first, omitted), inspect)
            with pytest.raises(StateTransactionNestingError, match="not enlisted"):
                await omitted.read(lambda tx: tx.scan_records())
            with pytest.raises(StateTransactionNestingError, match="two StateStorageGroups"):
                await other.read(lambda tx: tx.scan_records())
            return "result"

        try:
            assert await group.read_many((second, first, first), inspect) == "result"
            assert active_state_scope() is None
            with pytest.raises(ValueError, match="requires a store"):
                await group.read_many((), inspect)
            with pytest.raises(RuntimeError, match="does not belong"):
                await group.read_many((first, other), inspect)
        finally:
            await other.close()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("failure", (RuntimeError, asyncio.CancelledError))
async def test_state_group_read_scope_cleans_up_after_callback_failure(
    backend: str, failure: type[BaseException], tmp_path: Path,
) -> None:
    async with _open_stores(backend, tmp_path / backend) as (first, second, _):
        async def fail(scope: StateGroupTransaction) -> None:
            assert await second.read(lambda tx: tx.scan_records()) == ()
            raise failure()

        with pytest.raises(failure):
            await first.storage_group.read_many((first, second), fail)
        assert active_state_scope() is None
        await asyncio.wait_for(first.mutate(lambda tx: tx.insert_record(_record(1))), 5)
        assert await second.read(lambda tx: tx.scan_records()) == ()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_group_read_scope_reuses_active_mutation(
    backend: str, tmp_path: Path,
) -> None:
    async with _open_stores(backend, tmp_path / backend) as (first, second, _):
        async def mutate(scope: StateGroupTransaction) -> None:
            await scope.transaction(first).insert_record(_record(1))
            await scope.transaction(second).insert_record(_record(2))

            async def inspect(nested: StateGroupTransaction) -> None:
                for store, value in ((first, 1), (second, 2)):
                    assert nested.transaction(store) is scope.transaction(store)
                    assert await store.read(lambda tx: tx.get_record(_digest("record"))) == _record(value)

            await first.storage_group.read_many((second, first), inspect)
            assert active_state_scope().writable

        await first.storage_group.mutate((first, second), mutate)


@pytest.mark.parametrize("backend", (*_BACKENDS, "sql-wal"))
async def test_state_group_read_scope_holds_one_snapshot(
    backend: str, tmp_path: Path,
) -> None:
    async with _open_stores(backend, tmp_path / backend) as (first, second, _):
        group = first.storage_group
        if backend == "sql-wal":
            # Exercise a writer committing while the reader still owns its snapshot.
            async with first.context.engine.connect() as connection:
                await connection.execute(text("PRAGMA journal_mode=WAL"))
        start, attempting = asyncio.Event(), asyncio.Event()
        for store in (first, second):
            await store.mutate(lambda tx: tx.insert_record(_record(1)))

        async def writer() -> None:
            await start.wait()
            attempting.set()

            async def change(scope: StateGroupTransaction) -> None:
                for store in (first, second):
                    assert await scope.transaction(store).replace_record(_record(2), expected_storage_revision=0)

            await group.mutate((first, second), change)

        task = asyncio.create_task(writer())

        async def inspect(scope: StateGroupTransaction) -> None:
            assert await first.read(lambda tx: tx.get_record(_digest("record"))) == _record(1)
            start.set()
            await attempting.wait()
            if backend == "sql-wal":
                await asyncio.wait_for(task, 5)
            else:
                await asyncio.sleep(0)
                assert not task.done()
            assert await second.read(lambda tx: tx.get_record(_digest("record"))) == _record(1)

        try:
            await group.read_many((second, first), inspect)
            await asyncio.wait_for(task, 5)
            assert await second.read(lambda tx: tx.get_record(_digest("record"))) == _record(2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_state_group_read_scope_sql_opens_one_session_and_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _open_stores("sql", tmp_path / "sql") as (first, second, _):
        dialect = first.context.dialect
        begin = dialect.begin_consistent_read
        snapshots, checkouts = [], []

        async def record_begin(session):
            snapshots.append(session)
            await begin(session)

        monkeypatch.setattr(dialect, "begin_consistent_read", record_begin)
        event.listen(first.context.engine.sync_engine, "checkout", lambda *args: checkouts.append(args[0]))

        async def inspect(scope: StateGroupTransaction) -> None:
            for store in (first, second, first):
                assert await store.read(lambda tx: tx.scan_records()) == ()
            assert await first.storage_group.read_many((second,), lambda tx: second.read(
                lambda nested: nested.scan_records()
            )) == ()

        await first.storage_group.read_many((first, second), inspect)
        assert len(snapshots) == len(checkouts) == 1
        assert await second.read(lambda tx: tx.scan_records()) == ()
        assert len(snapshots) == len(checkouts) == 2
        assert checkouts[0] is checkouts[1]


@pytest.mark.parametrize(("writable", "outcome"), (
    (False, "commit"), (False, "rollback"), (False, "cancel"),
    (True, "commit"), (True, "rollback"), (True, "cancel"), (True, "retry"),
))
async def test_sql_group_clock_is_database_sourced_and_transaction_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writable: bool, outcome: str,
) -> None:
    async with _open_stores("sql", tmp_path / "sql") as (first, second, _):
        dialect = first.context.dialect
        database_now = dialect.database_now
        samples = []
        statements = []

        async def sample_database_time(session):
            await database_now(session)
            value = datetime(2001, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=len(samples))
            samples.append(value)
            return value

        monkeypatch.setattr(dialect, "database_now", sample_database_time)
        event.listen(first.context.engine.sync_engine, "before_cursor_execute",
                     lambda conn, cursor, statement, parameters, context, executemany:
                     statements.append(statement))
        group = first.storage_group
        execute = group.mutate if writable else group.read_many

        async def inspect(scope: StateGroupTransaction) -> None:
            expected = await scope.transaction(first).now()
            assert expected == samples[-1]
            assert await scope.transaction(second).now() == expected
            assert await first.read(lambda tx: tx.now()) == expected
            assert await group.read_many((second,), lambda nested:
                                         nested.transaction(second).now()) == expected

        async def finish(scope: StateGroupTransaction) -> None:
            await inspect(scope)
            if outcome == "rollback":
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if outcome == "cancel":
                raise asyncio.CancelledError
            if outcome == "retry" and len(samples) == 1:
                error = sqlite3.OperationalError("database is locked")
                error.sqlite_errorcode = 5
                error.sqlite_errorname = "SQLITE_BUSY"
                raise OperationalError("clock transaction", {}, error)

        if outcome in {"commit", "retry"}:
            await execute((first, second), finish)
        else:
            with pytest.raises(AIError if outcome == "rollback" else asyncio.CancelledError):
                await execute((first, second), finish)
        attempts = 2 if outcome == "retry" else 1
        assert len(samples) == attempts
        clock_statements = tuple(statement for statement in statements if "strftime" in statement)
        assert len(clock_statements) == attempts
        await execute((second, first), inspect)
        assert len(samples) == attempts + 1
        assert samples[0] != samples[1]
        # The pooled connection and a nested scope must not extend timestamp lifetime.
        await first.read(lambda tx: tx.now())
        assert len(samples) == attempts + 2


async def test_state_group_read_scope_filesystem_releases_partial_locks_and_ignores_other_members(
    tmp_path: Path,
) -> None:
    async with _open_stores("filesystem", tmp_path / "fs") as (first, second, unrelated):
        await second._consistency_lock.acquire()
        entered = asyncio.Event()
        acquire = first._consistency_lock.acquire

        async def acquired() -> None:
            await acquire()
            entered.set()

        first._consistency_lock.acquire = acquired
        task = asyncio.create_task(first.storage_group.read_many(
            (second, first), lambda scope: first.read(lambda tx: tx.scan_records())
        ))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not first._consistency_lock.locked()
        finally:
            second._consistency_lock.release()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await unrelated._consistency_lock.acquire()
        try:
            assert await asyncio.wait_for(first.storage_group.read_many(
                (second, first), lambda scope: first.read(lambda tx: tx.scan_records())
            ), 5) == ()
        finally:
            unrelated._consistency_lock.release()


@pytest.mark.parametrize("failure", (None, RuntimeError, asyncio.CancelledError))
async def test_state_group_read_scope_filesystem_live_reader_holds_all_publication_fences(
    tmp_path: Path, failure: type[BaseException] | None,
) -> None:
    root = tmp_path / "fs"
    async with _open_stores("filesystem", root) as writers:
        for store in writers[:2]:
            await store.mutate(lambda tx: tx.insert_record(_record(1)))
        readers = _filesystem_stores(root, read_only=True)
        for store in readers:
            await store.initialize()
        start, attempting = asyncio.Event(), asyncio.Event()

        async def writer() -> None:
            await start.wait()
            attempting.set()

            async def change(scope: StateGroupTransaction) -> None:
                for store in writers[:2]:
                    assert await scope.transaction(store).replace_record(_record(2), expected_storage_revision=0)

            await writers[0].storage_group.mutate(writers[:2], change)

        task = asyncio.create_task(writer())

        async def inspect(scope: StateGroupTransaction) -> None:
            assert await readers[0].read(lambda tx: tx.get_record(_digest("record"))) == _record(1)
            start.set()
            await attempting.wait()
            await asyncio.sleep(0)
            assert not task.done()
            assert await readers[1].read(lambda tx: tx.get_record(_digest("record"))) == _record(1)
            if failure is not None:
                raise failure()

        try:
            if failure is None:
                await readers[0].storage_group.read_many(readers[:2], inspect)
            else:
                with pytest.raises(failure):
                    await readers[0].storage_group.read_many(readers[:2], inspect)
            await asyncio.wait_for(task, 5)
            assert await readers[1].read(lambda tx: tx.get_record(_digest("record"))) == _record(2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            for store in reversed(readers):
                await store.close()
