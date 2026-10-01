#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Asset mutation, replay, and immutable metadata contracts."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.asset import AssetKey, AssetStore, FilesystemAssetBackend, InMemoryAssetBackend, SqlAssetBackend
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, StorageChange, StorageOperation, StorageOverlay


@pytest_asyncio.fixture(params=("memory", "filesystem", "sqlite-inline", "sqlite-object"))
async def asset_store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[AssetStore]:
    engine = None
    if request.param == "memory":
        backend = InMemoryAssetBackend()
    elif request.param == "filesystem":
        backend = FilesystemAssetBackend(tmp_path / "assets")
    else:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
        await provision_asset_database(engine)
        backend = SqlAssetBackend(
            engine, namespace="conformance",
            payload_policy=PayloadPolicy(0) if request.param == "sqlite-object" else None,
        )
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_asset_batch_replay_keeps_original_revision_precondition(asset_store: AssetStore) -> None:
    key = AssetKey("resource", "one")
    changes = (StorageChange(StorageOperation.PUT, key, b"one", None),)
    revision = await asset_store.current_revision()
    committed = await asset_store.apply_batch(changes, expected_revision=revision, idempotency_key="batch")
    await asset_store.put(key, b"two")
    current = await asset_store.current_revision()

    assert await asset_store.apply_batch(
        changes, expected_revision=revision, idempotency_key="batch"
    ) == committed
    assert await asset_store.current_revision() == current
    assert await asset_store.get(key) == b"two"
    assert len(await asset_store.list_versions(key)) == 2
    with pytest.raises(AIError) as conflict:
        await asset_store.apply_batch(
            (StorageChange(StorageOperation.PUT, key, b"different", None),),
            expected_revision=revision,
            idempotency_key="batch",
        )
    assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_asset_metadata_reads_cannot_mutate_stored_versions(asset_store: AssetStore) -> None:
    key = AssetKey("resource", "one")
    info = await asset_store.put(key, b"one", metadata={"nested": {"mode": 1}})
    revision = await asset_store.current_revision()
    detached = info.metadata["nested"]
    assert isinstance(detached, dict)
    detached["mode"] = 2
    with pytest.raises(TypeError):
        info.metadata["extra"] = True

    current = await asset_store.stat(key)
    assert current is not None
    assert current.metadata == {"nested": {"mode": 1}}
    assert (await asset_store.list_versions(key))[0].metadata == {"nested": {"mode": 1}}
    assert await asset_store.current_revision() == revision


@pytest.mark.asyncio
async def test_asset_mixed_batch_only_records_changed_entries(asset_store: AssetStore) -> None:
    stable = AssetKey("resource", "stable")
    changed = AssetKey("resource", "changed")
    missing_delete = AssetKey("resource", "missing-delete")
    missing_reset = AssetKey("resource", "missing-reset")
    original = await asset_store.put(stable, b"same")
    result = await asset_store.apply_batch((
        StorageChange(StorageOperation.PUT, stable, b"same", None),
        StorageChange(StorageOperation.PUT, changed, b"new", None),
        StorageChange(StorageOperation.DELETE, missing_delete, None, None),
        StorageChange(StorageOperation.RESET, missing_reset, None, None),
    ), idempotency_key="mixed")
    assert await asset_store.batch_result("mixed") == result

    assert result.results[0].changed is False
    assert result.results[0].info == original
    assert result.results[1].changed is True
    assert result.results[2].deleted is False
    assert result.results[3].reset is False
    assert len(await asset_store.list_versions(stable)) == 1
    assert await asset_store.stat(missing_delete) is None
    assert await asset_store.stat(missing_reset) is None


@pytest.mark.asyncio
async def test_empty_asset_bytes_round_trip_through_versions_and_snapshot(asset_store: AssetStore) -> None:
    key = AssetKey("resource", "empty")
    info = await asset_store.put(key, b"")
    assert info.size == 0
    assert await asset_store.get(key) == b""
    versions = await asset_store.resolve_versions((key,))
    assert await asset_store.read_versions(versions) == (b"",)
    objects = InMemoryObjectStore()
    reference = await asset_store.snapshot((key,), object_store=objects)
    restored = AssetStore.from_snapshot(reference, object_store=objects)
    await restored.initialize()
    try:
        assert await restored.get(key) == b""
    finally:
        await restored.close()
