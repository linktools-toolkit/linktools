#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-before-write mutations keep a snapshot and retry stale write upgrades."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.storage import create_sql_storage_context


@pytest.mark.asyncio
async def test_sqlite_mutation_retries_whole_callback_after_snapshot_upgrade(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'snapshot.db'}"
    engines = (create_async_engine(url), create_async_engine(url))
    first, second = (create_sql_storage_context(engine) for engine in engines)
    observed: list[tuple[int, int]] = []

    async def concurrent_write(session: AsyncSession) -> None:
        await session.execute(text("UPDATE ai_values SET value = 1"))

    async def mutate(session: AsyncSession) -> int:
        before = await session.scalar(text("SELECT value FROM ai_values"))
        if not observed:
            await second.run_mutation(concurrent_write)
        after = await session.scalar(text("SELECT value FROM ai_values"))
        observed.append((before, after))
        assert before == after
        await session.execute(text("UPDATE ai_values SET value = value + 1"))
        return after + 1

    try:
        await first.initialize()
        await second.initialize()
        async with engines[0].begin() as connection:
            await connection.execute(text("PRAGMA journal_mode=WAL"))
            await connection.execute(text("CREATE TABLE ai_values (value INTEGER NOT NULL)"))
            await connection.execute(text("INSERT INTO ai_values (value) VALUES (0)"))

        assert await first.run_mutation(mutate) == 2
        assert observed == [(0, 0), (1, 1)]
        async with engines[1].connect() as connection:
            assert await connection.scalar(text("SELECT value FROM ai_values")) == 2
    finally:
        await first.close()
        await second.close()
        for engine in engines:
            await engine.dispose()
