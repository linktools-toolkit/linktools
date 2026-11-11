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
from linktools.ai.asset._store import _batch_request_digest
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
        committed = await backend.apply_batch((change,), idempotency_key="upload",
                                              request_digest=_batch_request_digest((change,), None))
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
            assert payload["idempotency_key"] == "upload"
            assert "idempotency_key_digest" not in payload
            objects = metadata.tables["ai_objects"]
            assert (await connection.execute(select(objects.c.content_digest, objects.c.size))).all() == [
                (info.digest, info.size),
            ]
            chunks = metadata.tables["ai_object_chunks"]
            bodies = (await connection.execute(select(chunks.c.content).order_by(chunks.c.chunk_index))).scalars().all()
            assert all(isinstance(body, bytes) for body in bodies)
            assert b"".join(bodies) == content

        assert await backend.get(key) == content
        assert await backend.apply_batch((change,), idempotency_key="upload",
                                              request_digest=_batch_request_digest((change,), None)) == committed
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
        change = StorageChange(StorageOperation.PUT, key, b"small", None)
        with pytest.raises(AIError) as caught:
            await backend.apply_batch(
                (change,), idempotency_key="upload", request_digest=_batch_request_digest((change,), None),
            )
        assert caught.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert await backend.head_revision() == before
        assert await backend.stat(key) is None
        assert await backend.batch_result("upload") is None
    finally:
        await backend.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_asset_namespace_and_content_identity_need_no_nested_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock
    from linktools.ai.asset import _sql

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'namespace.db'}")
    backend = SqlAssetBackend(engine, namespace="files/中文")
    other = SqlAssetBackend(engine, namespace="files/other")
    content = b"one immutable payload"
    key = AssetKey("file", "same")
    try:
        await provision_asset_database(engine)
        await backend.initialize()
        await other.initialize()
        await backend.put(key, b"previous")
        expected = hashlib.sha256(content).hexdigest()
        change = StorageChange(StorageOperation.PUT, key, content, None)
        request_digest = _batch_request_digest((change,), None)
        tracked = Mock(wraps=hashlib.sha256)
        with monkeypatch.context() as patch:
            patch.setattr(_sql.hashlib, "sha256", tracked)
            committed = await backend.apply_batch(
                (change,), idempotency_key="same-receipt", request_digest=request_digest,
            )
        assert sum(call.args == (content,) for call in tracked.call_args_list) == 1
        assert await backend.batch_result("same-receipt") == committed
        other_change = StorageChange(StorageOperation.PUT, key, b"other namespace", None)
        other_digest = _batch_request_digest((other_change,), None)
        await other.apply_batch(
            (other_change,), idempotency_key="same-receipt", request_digest=other_digest,
        )
        with pytest.raises(AIError) as conflict:
            await backend.apply_batch(
                (other_change,), idempotency_key="same-receipt", request_digest=other_digest,
            )
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        info = await backend.stat(key)
        assert info is not None and info.content is not None and info.content.ref is not None
        assert info.digest == expected
        assert info.content.ref.key == f"asset/files/中文/{expected}"
        metadata = build_asset_sql_metadata()
        async with engine.connect() as connection:
            heads = metadata.tables["ai_asset_heads"]
            assert "namespace_digest" not in heads.c
            assert set((await connection.execute(select(heads.c.namespace))).scalars()) == {
                "files/中文", "files/other",
            }
        assert await backend.get(key) == content
        assert await other.get(key) == b"other namespace"
        await backend.validate_integrity()
        await other.validate_integrity()
    finally:
        await backend.close()
        await other.close()
        await engine.dispose()
