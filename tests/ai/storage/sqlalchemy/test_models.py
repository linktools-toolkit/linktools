#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/test_models.py"""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from linktools.ai.storage.sqlalchemy.asset import _idempotency_key_hash, asset_path_hash
from linktools.ai.storage.sqlalchemy.models import (
    Base,
    AssetRow,
    AssetIdempotencyRow,
    AssetRevisionRow,
)
from linktools.ai.asset.path import AssetPath


@pytest.mark.asyncio
async def test_create_all_and_insert_resource_row(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine) as session:
        session.add(
            AssetRow(
                path="/a.txt",
                path_hash=asset_path_hash(AssetPath("/a.txt")),
                kind="file",
                etag="e1",
                version=1,
                content_type="text/plain",
                size=5,
                content=b"hello",
                modified_at=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ),
                metadata_json="{}",
                deleted_at=None,
                whiteout_version=None,
            )
        )
        await session.commit()

    async with AsyncSession(engine) as session:
        from sqlalchemy import select

        row = (
            await session.execute(
                select(AssetRow).where(AssetRow.path == "/a.txt")
            )
        ).scalar_one()
        assert row.content == b"hello"
        assert row.version == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_idempotency_row_unique_key(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test2.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as session:
        session.add(
            AssetIdempotencyRow(
                key="put:k1",
                key_hash=_idempotency_key_hash("put:k1"),
                request_hash="h1",
                result_json=None,
            )
        )
        await session.commit()
    async with AsyncSession(engine) as session:
        from sqlalchemy import select

        row = (
            await session.execute(
                select(AssetIdempotencyRow).where(AssetIdempotencyRow.key == "put:k1")
            )
        ).scalar_one()
        assert row.request_hash == "h1"
    await engine.dispose()


@pytest.mark.asyncio
async def test_revision_row_single_counter(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test3.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as session:
        session.add(AssetRevisionRow(id=1, value=0))
        await session.commit()
    async with AsyncSession(engine) as session:
        from sqlalchemy import select

        row = (
            await session.execute(select(AssetRevisionRow).where(AssetRevisionRow.id == 1))
        ).scalar_one()
        assert row.value == 0
    await engine.dispose()
