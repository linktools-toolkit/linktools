#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A caught partial owner deletion cannot commit an inconsistent SQL group."""

import asyncio
import getpass
from pathlib import Path

import pytest
from sqlalchemy import select, text, update
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
from tests.ai.test_state_batch_delete_backends import _digest, _record, _seed


async def _assert_caught_failure(
    engine: AsyncEngine, failure: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    await provision_database(engine)
    store = SqlStateStore(engine, store_digest=_digest(f"delete-failure:{failure}"))
    other = SqlStateStore(engine, group=store._storage_group,
                          store_digest=_digest(f"delete-other:{failure}"))
    await store.initialize()
    await other.initialize()
    records = tuple(_record(index) for index in range(260 if failure == "conflict" else 2))
    await _seed(store, records)
    await _seed(other, (records[0],))
    entered, release = asyncio.Event(), asyncio.Event()
    caught = []
    execute = _SqlTransaction._execute

    async def interrupt(transaction, statement, parameters=None):
        if (transaction._store_digest == store.store_digest
                and getattr(statement, "is_delete", False)
                and statement.table.name == "ai_state_aliases"):
            if failure == "cancel":
                entered.set()
                await release.wait()
            elif failure == "database":
                return await execute(transaction, text("SELECT * FROM missing_delete_probe_table"))
        return await execute(transaction, statement, parameters)

    async def remove(transaction):
        keys = tuple(record.key_digest for record in records)
        if failure == "conflict":
            await transaction.get_records(keys)
            table = transaction._table("ai_state_records")
            # The stale cached version models losing a CAS after an earlier batch.
            await transaction._execute(update(table).where(
                table.c.store_digest == store.store_digest.hex(),
                table.c.key_digest == records[-1].key_digest.hex(),
            ).values(storage_revision=1))
        await transaction.delete_records((*keys, keys[0], _digest("missing")))

    async def callback(group):
        try:
            await store.mutate(remove)
        except (asyncio.CancelledError, SQLAlchemyError, AIError) as error:
            caught.append(error)
        if failure != "database":
            # A caller can continue after catching cancellation/CAS, but cannot commit.
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
                with pytest.raises((asyncio.CancelledError, AIError)) as raised:
                    await task
                assert len(caught) == 1
                if failure == "database":
                    assert raised.value.__cause__ is caught[0]
                else:
                    original: BaseException | None = raised.value
                    seen: set[int] = set()
                    while (original is not caught[0] and isinstance(original, asyncio.CancelledError)
                           and id(original) not in seen):
                        seen.add(id(original))
                        original = original.__cause__ or original.__context__
                    assert original is caught[0]
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        assert await store.read(lambda tx: tx.get_records(tuple(record.key_digest for record in records))) == {
            record.key_digest: record for record in records
        }
        assert len(await store.read(lambda tx: tx.scan_aliases())) == len(records)
        assert len(await store.read(lambda tx: tx.scan_facts())) == len(records)
        assert await other.read(lambda tx: tx.scan_records()) == (records[0],)
        await store.validate_integrity()
        await other.validate_integrity()
        async with engine.connect() as connection:
            assert await connection.scalar(select(1)) == 1
        # Failure ownership ends with the transaction; a fresh delete still succeeds.
        await store.mutate(lambda tx: tx.delete_records((records[0].key_digest,
                            _digest("missing"), records[0].key_digest)))
        assert await store.read(lambda tx: tx.get_record(records[0].key_digest)) is None
        await store.validate_integrity()
    finally:
        await other.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("cancel", "database", "conflict"))
async def test_sqlite_caught_batch_delete_failure_rolls_back_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'delete.db'}",
                                 pool_size=1, max_overflow=0)
    try:
        await _assert_caught_failure(engine, failure, monkeypatch)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_caught_batch_delete_failures_roll_back_group(
    _server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url, pool_size=1, max_overflow=0)
    try:
        for failure in ("cancel", "database", "conflict"):
            await _assert_caught_failure(engine, failure, monkeypatch)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_caught_delete_failures_retain_first_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'first.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    records = (_record(1), _record(2))
    await _seed(store, records)
    first, second = AIError(ErrorCode.STORAGE_CONFLICT), AIError(ErrorCode.STORAGE_UNAVAILABLE)
    errors = iter((first, second))
    execute = _SqlTransaction._execute

    async def fail(transaction, statement, parameters=None):
        if getattr(statement, "is_delete", False) and statement.table.name == "ai_state_aliases":
            raise next(errors)
        return await execute(transaction, statement, parameters)

    async def callback(transaction):
        for record in records:
            try:
                await transaction.delete_records((record.key_digest,))
            except AIError:
                pass

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "_execute", fail)
            with pytest.raises(AIError) as raised:
                await store.mutate(callback)
        assert raised.value is first
        assert len(await store.read(lambda tx: tx.scan_records())) == 2
        await store.validate_integrity()
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before", ("ai_state_aliases", "ai_state_facts"))
async def test_caught_scalar_delete_cancel_preserves_owner_and_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_before: str,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scalar.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    record = _record(1)
    await _seed(store, (record,))
    execute = _SqlTransaction._execute
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause(transaction, statement, parameters=None):
        if getattr(statement, "is_delete", False) and statement.table.name == cancel_before:
            entered.set()
            await release.wait()
        return await execute(transaction, statement, parameters)

    async def callback(transaction):
        try:
            await transaction.delete_record(record.key_digest)
        except asyncio.CancelledError:
            await transaction.insert_record(_record(2))

    try:
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "_execute", pause)
            task = asyncio.create_task(store.mutate(callback))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
                if cancel_before == "ai_state_aliases":
                    await task
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await task
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
        assert len(await store.read(lambda tx: tx.scan_records())) == (
            2 if cancel_before == "ai_state_aliases" else 1
        )
        assert len(await store.read(lambda tx: tx.scan_aliases())) == 1
        assert len(await store.read(lambda tx: tx.scan_facts())) == 1
        await store.validate_integrity()
    finally:
        await store.close()
        await engine.dispose()
