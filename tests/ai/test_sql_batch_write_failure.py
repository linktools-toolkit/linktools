#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Caught interrupted record writes cannot commit a SQL storage group."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import AIError
from linktools.ai.migrate import provision_database
from linktools.ai.storage import SQLiteDialect, SqlStorageContext
from linktools.ai.runtime.state._sql import SqlStateStore, _SqlTransaction
from linktools.ai.runtime.state._store import RecordReplacement
from tests.ai.test_state_batch_delete_backends import _digest, _record, _seed


@pytest.mark.asyncio
@pytest.mark.parametrize("executemany", (False, True))
@pytest.mark.parametrize("failure,size", (("conflict", 2), ("conflict", 65),
                                         ("cancel", 65), ("database", 65)))
async def test_caught_replace_failure_rolls_back_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, size: int, executemany: bool,
) -> None:
    class Dialect(SQLiteDialect):
        @property
        def prefer_update_executemany(self) -> bool:
            return executemany

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'replace.db'}")
    await provision_database(engine)
    context = SqlStorageContext(engine, async_sessionmaker(engine), Dialect())
    store = SqlStateStore(engine, context=context)
    other = SqlStateStore(engine, group=store._storage_group, store_digest=_digest("other"))
    await store.initialize()
    await other.initialize()
    records = tuple(sorted((_record(index) for index in range(size)), key=lambda r: r.key_digest))
    await _seed(store, records)
    entered, release = asyncio.Event(), asyncio.Event()
    caught = []
    execute = _SqlTransaction._execute
    writes = 0

    async def interrupt(transaction, statement, parameters=None):
        nonlocal writes
        if (transaction._store_digest == store.store_digest
                and getattr(statement, "is_update", False)
                and statement.table.name == "ai_state_records"):
            writes += 1
            if writes == 2 and failure == "cancel":
                entered.set()
                await release.wait()
            elif writes == 2 and failure == "database":
                return await execute(transaction, text("SELECT * FROM missing_replace_probe_table"))
        return await execute(transaction, statement, parameters)

    async def callback(group):
        transaction = group.transaction(store)
        if failure == "conflict":
            await transaction.get_records(tuple(record.key_digest for record in records))
            table = transaction._table("ai_state_records")
            # Preserve stale cached versions to model a competing CAS winner.
            await execute(transaction, update(table).where(
                table.c.store_digest == store.store_digest.hex(),
                table.c.key_digest == records[-1].key_digest.hex(),
            ).values(storage_version=1))
        try:
            await transaction.replace_records(tuple(
                RecordReplacement(replace(record, storage_version=1, data={"changed": True}), 0)
                for record in records
            ))
        except (asyncio.CancelledError, AIError, SQLAlchemyError) as error:
            caught.append(error)
        await group.transaction(other).insert_record(_record(1000))
        return "caught"

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "_execute", interrupt)
            task = asyncio.create_task(store.storage_group.mutate((store, other), callback))
            try:
                if failure == "cancel":
                    await asyncio.wait_for(entered.wait(), 5)
                    task.cancel()
                with pytest.raises((AIError, asyncio.CancelledError)):
                    await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        assert len(caught) == 1
        assert await store.read(lambda tx: tx.get_records(tuple(r.key_digest for r in records))) == {
            record.key_digest: record for record in records
        }
        assert await other.read(lambda tx: tx.scan_records()) == ()
        await store.validate_integrity()
        await other.validate_integrity()
        # Failure ownership is local to the discarded transaction.
        assert await store.mutate(lambda tx: tx.replace_record(
            replace(records[0], storage_version=1), expected_storage_version=0,
        ))
    finally:
        await other.close()
        await store.close()
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_lost_scalar_cas_does_not_abort_unrelated_write(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scalar.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    record = _record(1)
    await store.mutate(lambda tx: tx.insert_record(record))

    async def callback(transaction):
        await transaction.get_record(record.key_digest)
        table = transaction._table("ai_state_records")
        await transaction._execute(update(table).where(
            table.c.store_digest == store.store_digest.hex(),
            table.c.key_digest == record.key_digest.hex(),
        ).values(storage_version=1))
        assert not await transaction.replace_record(
            replace(record, storage_version=1), expected_storage_version=0,
        )
        await transaction.insert_record(_record(2))

    try:
        await store.mutate(callback)
        assert len(await store.read(lambda tx: tx.scan_records())) == 2
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_caught_cancellation_after_insert_statement_rolls_back_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'insert.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    entered, release = asyncio.Event(), asyncio.Event()
    execute = AsyncSession.execute
    records = (_record(1), _record(2))

    async def pause_after_write(session, statement, *args, **kwargs):
        result = await execute(session, statement, *args, **kwargs)
        if (getattr(statement, "is_insert", False)
                and statement.table.name == "ai_state_records"):
            entered.set()
            await release.wait()
        return result

    async def callback(transaction):
        try:
            await transaction.insert_records(records)
        except asyncio.CancelledError:
            return "caught"

    try:
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "execute", pause_after_write)
            task = asyncio.create_task(store.mutate(callback))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        assert await store.read(lambda tx: tx.scan_records()) == ()
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_large_replacements_preserve_individual_record_capacity(
    _server: tuple[str, list[str]],
) -> None:
    from sqlalchemy.engine import URL

    _name, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    admin = create_async_engine(url, hide_parameters=True)
    engine = None
    store = None
    try:
        async with admin.connect() as connection:
            original_limit = int(await connection.scalar(text("SELECT @@GLOBAL.max_allowed_packet")))
            try:
                await connection.execute(text("SET GLOBAL max_allowed_packet = 1048576"))
                # A fresh physical connection inherits the temporary server limit.
                engine = create_async_engine(url, hide_parameters=True, pool_size=1, max_overflow=0)
                await provision_database(engine)
                async with engine.connect() as probe:
                    assert await probe.scalar(text("SELECT @@SESSION.max_allowed_packet")) == 1048576
                store = SqlStateStore(engine)
                await store.initialize()
                records = tuple(_record(index) for index in range(8))
                await store.mutate(lambda tx: tx.insert_records(records))
                replacements = tuple(replace(record, storage_version=1, data={"body": "x" * (128 * 1024)})
                                     for record in records)

                async def replace_individually(transaction):
                    for record in replacements:
                        assert await transaction.replace_record(record, expected_storage_version=0)

                await store.mutate(replace_individually)
                final = tuple(replace(record, storage_version=2, state="replaced") for record in replacements)
                await store.mutate(lambda tx: tx.replace_records(tuple(
                    RecordReplacement(record, 1) for record in final
                )))
                assert await store.read(lambda tx: tx.get_records(tuple(r.key_digest for r in records))) == {
                    record.key_digest: record for record in final
                }
            finally:
                try:
                    if store is not None:
                        await store.close()
                finally:
                    try:
                        if engine is not None:
                            await engine.dispose()
                    finally:
                        await connection.execute(text(f"SET GLOBAL max_allowed_packet = {original_limit}"))
    finally:
        await admin.dispose()
