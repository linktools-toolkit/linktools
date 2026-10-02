#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A failed SQL statement cannot restart a cancelled mutation callback."""

import asyncio
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pytest
from aiosqlite import Cursor
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state._sql import SqlStateStore
from linktools.ai.runtime.state._store import StateTransaction


@pytest.mark.asyncio
async def test_cancelled_statement_failure_does_not_retry_or_commit_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    key = b"s" * 32
    executing = asyncio.Event()
    release = asyncio.Event()
    continued = asyncio.Event()
    original_execute = Cursor.execute

    async def fail_statement(
        cursor: Cursor,
        sql: str,
        parameters: Iterable[object] | None = None,
    ) -> Cursor:
        if (
            not executing.is_set()
            and sql.startswith("SELECT")
            and "ai_state_sequences" in sql
        ):
            executing.set()
            await release.wait()
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = 5
            error.sqlite_errorname = "SQLITE_BUSY"
            raise error
        return await original_execute(cursor, sql, parameters)

    async def mutate(transaction: StateTransaction) -> int:
        await transaction.get_sequence(key)
        continued.set()
        return await transaction.reserve_sequence(key, 1)

    monkeypatch.setattr(Cursor, "execute", fail_statement)
    mutation = asyncio.create_task(store.mutate(mutate))
    try:
        await asyncio.wait_for(executing.wait(), 5)
        mutation.cancel("stop mutation")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(mutation, 5)
        cancellation: BaseException | None = raised.value
        while cancellation is not None and not cancellation.args:
            cancellation = cancellation.__cause__ or cancellation.__context__
        assert isinstance(cancellation, asyncio.CancelledError)
        assert cancellation.args == ("stop mutation",)
        assert not continued.is_set()
        assert await store.read(lambda transaction: transaction.get_sequence(key)) == 0
        assert await store.mutate(
            lambda transaction: transaction.reserve_sequence(key, 1)
        ) == 1
    finally:
        release.set()
        await asyncio.gather(mutation, return_exceptions=True)
        await store.close()
        await engine.dispose()
