#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL Asset metadata never duplicates its owned file bytes."""

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, SqlAssetBackend, build_asset_sql_metadata
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import (
    InMemoryObjectStore,
    ObjectStat,
    StorageChange,
    StorageOperation,
    build_object_sql_metadata,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", (b"", b"\x00small-file\xff", b"x" * (1024 * 1024 + 1)),
                         ids=("empty", "small-binary", "multiple-chunks"))
async def test_sql_asset_files_have_one_byte_owner_across_metadata_and_receipts(
    tmp_path: Path, content: bytes,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    backend = SqlAssetBackend(engine, namespace="files")
    metadata = build_asset_sql_metadata()
    build_object_sql_metadata(metadata=metadata)
    key = AssetKey("file", "one")
    change = StorageChange(StorageOperation.PUT, key, content, None)
    try:
        await provision_asset_database(engine)
        await backend.initialize()
        committed = await backend.apply_batch((change,), idempotency_key="upload")
        info = await backend.stat(key)
        assert info is not None and info.content is not None and info.content.ref is not None
        assert info.content.kind == "object"
        assert (info.digest, info.size) == (hashlib.sha256(content).hexdigest(), len(content))
        descriptor = info.content.to_json()
        assert "value" not in descriptor

        async with engine.connect() as connection:
            for name in ("ai_asset_entries", "ai_asset_changes"):
                table = metadata.tables[name]
                payload = await connection.scalar(select(table.c.payload_json))
                assert payload["value"]["content"] == descriptor
            receipt = metadata.tables["ai_asset_batch_receipts"]
            payload = await connection.scalar(select(receipt.c.payload_json))
            assert payload["results"][0]["info"]["content"] == descriptor
            objects = metadata.tables["ai_objects"]
            assert (await connection.execute(select(objects.c.content_digest, objects.c.size))).all() == [
                (info.digest, info.size),
            ]
            chunks = metadata.tables["ai_object_chunks"]
            bodies = (await connection.execute(select(chunks.c.content).order_by(chunks.c.chunk_index))).scalars().all()
            assert all(isinstance(body, bytes) for body in bodies)
            assert b"".join(bodies) == content

        assert await backend.get(key) == content
        assert await backend.apply_batch((change,), idempotency_key="upload") == committed
        await backend.delete(key)
        assert await backend.get(key) is None
        assert await backend.get_at_revision(key, info.revision) == content
        await backend.validate_integrity()
    finally:
        await backend.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_small_asset_object_failure_does_not_publish_metadata(tmp_path: Path) -> None:
    class UnavailableObjects(InMemoryObjectStore):
        async def put(
            self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str,
        ) -> ObjectStat:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    backend = SqlAssetBackend(engine, namespace="unavailable", object_store=UnavailableObjects())
    key = AssetKey("file", "small")
    try:
        await provision_asset_database(engine)
        await backend.initialize()
        before = await backend.head_revision()
        with pytest.raises(AIError) as caught:
            await backend.apply_batch(
                (StorageChange(StorageOperation.PUT, key, b"small", None),), idempotency_key="upload",
            )
        assert caught.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert await backend.head_revision() == before
        assert await backend.stat(key) is None
        assert await backend.batch_result("upload") is None
    finally:
        await backend.close()
        await engine.dispose()
