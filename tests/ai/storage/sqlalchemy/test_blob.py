#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-addressed blob write/read helpers.

Verifies the generic ``put_blob``/``read_blob`` storage helpers that dedup
content by its sha256 (insert-ignore on the unique sha256 key) and read a blob
back by its sha256 hex. The row class is passed in -- these helpers enforce the
content-addressed convention without owning any table, so the test supplies an
inline model shaped like a backend's blob row (sha256 unique + content bytes).

Uses the ``def test_x(): asyncio.run(_run())`` style (sync wrapper driving its
own event loop) so no pytest-asyncio mode config is needed."""

import asyncio
import hashlib
from contextlib import asynccontextmanager

from sqlalchemy import LargeBinary, String
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Mapped, mapped_column

from linktools.ai.storage.sqlalchemy.base import Base
from linktools.ai.storage.sqlalchemy.blob import put_blob, read_blob
from linktools.ai.storage.sqlalchemy.conventions import (
    TABLE_PREFIX,
    timestamp_indexes,
)
from linktools.ai.storage.sqlalchemy.dialects import SqliteDialect


class _BlobRow(Base):
    # Mirrors a backend's content-addressed blob row: sha256 unique + content.
    __tablename__ = f"{TABLE_PREFIX}blob_test_blobs"
    __table_args__ = (*timestamp_indexes(),)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    content: Mapped[bytes] = mapped_column(LargeBinary)


@asynccontextmanager
async def _engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/blob.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        yield engine, session_factory
    finally:
        await engine.dispose()


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_put_blob_returns_sha256_hex_matching_hashlib(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    sha = await put_blob(
                        session, dialect, _BlobRow, b"hello"
                    )
            assert sha == _sha(b"hello")

    asyncio.run(_run())


def test_put_blob_dedups_identical_content(tmp_path):
    async def _run():
        from sqlalchemy import select

        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    first = await put_blob(session, dialect, _BlobRow, b"same")
                    second = await put_blob(session, dialect, _BlobRow, b"same")
            assert first == second
            async with session_factory() as session:
                rows = (await session.scalars(select(_BlobRow))).all()
            assert len(rows) == 1  # identical content -> one row

    asyncio.run(_run())


def test_put_blob_distinguishes_different_content(tmp_path):
    async def _run():
        from sqlalchemy import select

        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    a = await put_blob(session, dialect, _BlobRow, b"one")
                    b = await put_blob(session, dialect, _BlobRow, b"two")
            assert a != b
            async with session_factory() as session:
                rows = (await session.scalars(select(_BlobRow))).all()
            assert len(rows) == 2

    asyncio.run(_run())


def test_put_blob_is_idempotent_on_repeat(tmp_path):
    # Repeated puts of the same content neither raise nor grow rows.
    async def _run():
        from sqlalchemy import select

        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    for _ in range(3):
                        await put_blob(session, dialect, _BlobRow, b"again")
            async with session_factory() as session:
                rows = (await session.scalars(select(_BlobRow))).all()
            assert len(rows) == 1

    asyncio.run(_run())


def test_read_blob_returns_content(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            dialect = SqliteDialect()
            async with session_factory() as session:
                async with session.begin():
                    sha = await put_blob(session, dialect, _BlobRow, b"payload")
            async with session_factory() as session:
                content = await read_blob(session, _BlobRow, sha)
            assert content == b"payload"

    asyncio.run(_run())


def test_read_blob_returns_none_for_unknown(tmp_path):
    async def _run():
        async with _engine(tmp_path) as (_, session_factory):
            async with session_factory() as session:
                content = await read_blob(session, _BlobRow, "f" * 64)
            assert content is None

    asyncio.run(_run())
