#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequence reservations retain transaction truth across scalar and batch SQL."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.sql import Executable

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._filesystem import FilesystemStateStore
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._schema import build_runtime_sql_metadata
from linktools.ai.runtime.state._sql import SqlStateStore, _SqlTransaction
from linktools.ai.runtime.state._store import StateStore, StateTransaction
from linktools.ai.storage import MySQLDialect, SqlStorageContext


async def _assert_reservations(store: StateStore) -> None:
    first, second, concurrent = (value * 32 for value in (b"a", b"b", b"c"))

    async def reserve(transaction: StateTransaction) -> None:
        assert await transaction.reserve_sequences({first: 5}) == {first: 5}
        assert await transaction.get_sequence(first) == 5
        assert await transaction.reserve_sequences({first: 3}) == {first: 8}
        assert await transaction.get_sequence(first) == 8
        assert await transaction.reserve_sequences({first: 2, second: 7}) == {
            first: 10, second: 7,
        }
        assert await transaction.get_sequences((first, second)) == {first: 10, second: 7}

    await store.mutate(reserve)

    async def rollback(transaction: StateTransaction) -> None:
        assert await transaction.reserve_sequence(first, 4) == 14
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    with pytest.raises(AIError, match="STORAGE_CONFLICT"):
        await store.mutate(rollback)
    assert await store.read(lambda transaction: transaction.get_sequence(first)) == 10
    values = await asyncio.gather(*(
        store.mutate(lambda transaction: transaction.reserve_sequence(concurrent, 1))
        for _ in range(6)
    ))
    assert sorted(values) == list(range(1, 7))
    assert await store.read(lambda transaction: transaction.get_sequence(concurrent)) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_sequence_reservations_preserve_batch_rollback_and_concurrent_truth(
    tmp_path: Path, backend: str,
) -> None:
    engine = None
    if backend == "memory":
        store = InMemoryStateStore()
    elif backend == "filesystem":
        store = FilesystemStateStore(
            tmp_path / "state", namespace="sequence", tenant_id="tenant",
            runtime_domain="conversation",
        )
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
    await store.initialize()
    try:
        await _assert_reservations(store)
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rowcount,lastrowid,expected", ((1, 41, 5), (2, 12, 12)))
async def test_mysql_single_sequence_uses_insert_or_increment_result(
    rowcount: int, lastrowid: int, expected: int,
) -> None:
    statements: list[Executable] = []

    class Session:
        async def execute(self, statement: Executable) -> SimpleNamespace:
            statements.append(statement)
            return SimpleNamespace(rowcount=rowcount, lastrowid=lastrowid)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    context = SqlStorageContext(engine, async_sessionmaker(engine), MySQLDialect())
    transaction = _SqlTransaction(Session(), build_runtime_sql_metadata(frozenset(RuntimeDomain)), context, b"s" * 32)
    key = b"k" * 32
    try:
        assert await transaction.reserve_sequences({key: 5}) == {key: expected}
        assert await transaction.get_sequence(key) == expected
        assert len(statements) == 1
        compiled = statements[0].compile(dialect=mysql.dialect())
        assert "last_insert_id(ai_state_sequences.value + " in str(compiled)
        assert compiled.params["value"] == 5
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_mysql_single_sequence_keeps_result_readback_when_driver_omits_id() -> None:
    statements: list[Executable] = []

    class Session:
        async def execute(self, statement: Executable) -> SimpleNamespace:
            statements.append(statement)
            return SimpleNamespace(rowcount=2, lastrowid=None)

        async def scalar(self, statement: Executable) -> int:
            statements.append(statement)
            return 12

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    context = SqlStorageContext(engine, async_sessionmaker(engine), MySQLDialect())
    transaction = _SqlTransaction(Session(), build_runtime_sql_metadata(frozenset(RuntimeDomain)), context, b"s" * 32)
    try:
        assert await transaction.reserve_sequences({b"k" * 32: 5}) == {b"k" * 32: 12}
        assert len(statements) == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_sequence_execution_preserves_batch_rollback_and_concurrency(
    _server: tuple[str, list[str]],
) -> None:
    from sqlalchemy.engine import URL

    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    engine = create_async_engine(URL.create(
        "mysql+asyncmy", username="root", database="metrics_test", query={"unix_socket": socket},
    ))
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    try:
        await _assert_reservations(store)
    finally:
        await store.close()
        await engine.dispose()
