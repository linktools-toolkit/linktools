#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""upsert_increment dialect contract.

Verifies the vendor seam that self-seeds a singleton counter row and
atomically increments it: ``INSERT ... ON CONFLICT (pk) DO UPDATE SET col =
col + step`` (SQLite/PostgreSQL) / ``INSERT ... ON DUPLICATE KEY UPDATE col =
col + step`` (MySQL). The row is created on the first call with ``col = step``
(the first-increment value) and advanced on every subsequent call.

Uses the ``def test_x(): asyncio.run(_run())`` style (sync wrapper driving its
own event loop) so no pytest-asyncio mode config is needed."""

import asyncio
from contextlib import asynccontextmanager

from sqlalchemy import BigInteger, Column, Integer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.sqlalchemy.base import Base
from linktools.ai.storage.sqlalchemy.dialects import SqliteDialect


class _Counter(Base):
    __tablename__ = "dialect_test_counter"
    id = Column(BigInteger, primary_key=True)
    revision = Column(Integer, nullable=False, default=0)


@asynccontextmanager
async def _engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/dialect.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        yield engine, session_factory
    finally:
        await engine.dispose()


def test_upsert_increment_seeds_singleton_on_first_call(tmp_path):
    # No row is pre-seeded: the first call creates it with col = step and
    # returns step (1).
    async def _run():
        from sqlalchemy import select

        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    result = await dialect.upsert_increment(
                        session,
                        model=_Counter,
                        pk=1,
                        column="revision",
                    )
            assert result == 1
            async with session_factory() as session:
                row = await session.get(_Counter, 1)
            assert row is not None
            assert row.revision == 1

    asyncio.run(_run())


def test_upsert_increment_advances_existing_value(tmp_path):
    # After the first call seeds (value 1), a second call advances to 2.
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    first = await dialect.upsert_increment(
                        session, model=_Counter, pk=1, column="revision"
                    )
                    second = await dialect.upsert_increment(
                        session, model=_Counter, pk=1, column="revision"
                    )
            assert first == 1
            assert second == 2

    asyncio.run(_run())


def test_upsert_increment_honors_step(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    first = await dialect.upsert_increment(
                        session, model=_Counter, pk=1, column="revision", step=5
                    )
                    second = await dialect.upsert_increment(
                        session, model=_Counter, pk=1, column="revision", step=5
                    )
            assert first == 5  # seed value == step
            assert second == 10

    asyncio.run(_run())


def test_upsert_increment_is_atomic_under_concurrency(tmp_path):
    # Concurrent calls with no pre-seeded row: the first seeds, the rest
    # increment, and no update is lost -- the 20 returned values are exactly
    # 1..20 in some order.
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()

            async def bump():
                async with session_factory() as session:
                    async with session.begin():
                        return await dialect.upsert_increment(
                            session, model=_Counter, pk=1, column="revision"
                        )

            results = await asyncio.gather(*(bump() for _ in range(20)))
            assert sorted(results) == list(range(1, 21))

    asyncio.run(_run())
