#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""delete_returning / update_returning dialect contract.

Verifies the vendor seam that DELETEs/UPDATEs rows and returns their column
values in one logical operation: ``DELETE ... RETURNING`` /
``UPDATE ... RETURNING`` on SQLite/PostgreSQL, and a same-transaction
SELECT-then-DELETE / UPDATE-then-SELECT on MySQL (no portable RETURNING)."""

import asyncio
from contextlib import asynccontextmanager

from sqlalchemy import BigInteger, Column, String
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.sqlalchemy.base import Base
from linktools.ai.storage.sqlalchemy.dialects import SqliteDialect


class _Doc(Base):
    __tablename__ = "dialect_test_doc"
    id = Column(BigInteger, primary_key=True)
    path = Column(String(128), unique=True)
    kind = Column(String(32))
    etag = Column(String(64))


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


def _seed(session, docs):
    session.add_all(
        _Doc(id=i, path=p, kind=k, etag=e) for i, (p, k, e) in enumerate(docs, 1)
    )


def test_delete_returning_returns_pre_delete_values(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    _seed(session, [("a", "agent", "v1"), ("b", "tool", "v2")])
                    deleted = await dialect.delete_returning(
                        session,
                        model=_Doc,
                        where=_Doc.path == "a",
                        returning=("path", "kind", "etag"),
                    )
            assert len(deleted) == 1
            assert (deleted[0].path, deleted[0].kind, deleted[0].etag) == (
                "a",
                "agent",
                "v1",
            )

    asyncio.run(_run())


def test_delete_returning_empty_when_no_match(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    deleted = await dialect.delete_returning(
                        session,
                        model=_Doc,
                        where=_Doc.path == "absent",
                        returning=("path",),
                    )
            assert deleted == ()

    asyncio.run(_run())


def test_delete_returning_deletes_all_matching_rows(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    _seed(
                        session,
                        [
                            ("a", "agent", "v1"),
                            ("b", "agent", "v2"),
                            ("c", "tool", "v3"),
                        ],
                    )
                    deleted = await dialect.delete_returning(
                        session,
                        model=_Doc,
                        where=_Doc.kind == "agent",
                        returning=("path",),
                    )
            assert {d.path for d in deleted} == {"a", "b"}

    asyncio.run(_run())


def test_update_returning_returns_post_update_values(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    _seed(session, [("a", "agent", "v1")])
                    rows = await dialect.update_returning(
                        session,
                        model=_Doc,
                        where=_Doc.path == "a",
                        values={"etag": "v2"},
                        returning=("path", "etag"),
                    )
            assert len(rows) == 1
            row = rows[0]
            assert row.path == "a"
            assert row.etag == "v2"

    asyncio.run(_run())


def test_update_returning_empty_when_no_match(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    rows = await dialect.update_returning(
                        session,
                        model=_Doc,
                        where=_Doc.path == "absent",
                        values={"etag": "v2"},
                        returning=("path", "etag"),
                    )
            assert rows == ()

    asyncio.run(_run())
