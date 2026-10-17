#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Asset mutation, replay, and immutable metadata contracts."""

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.asset import AssetKey, AssetStore, AssetVersionRef, FilesystemAssetBackend, InMemoryAssetBackend, SqlAssetBackend
from linktools.ai.capability import CapabilityLoadContext, CapabilityLoadEntry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, StorageChange, StorageOperation, StorageOverlay, read_object


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
    first = (await asset_store.resolve_versions((key,)))[0]
    await asset_store.put(key, b"two")
    second = (await asset_store.resolve_versions((key,)))[0]
    current = await asset_store.current_revision()

    assert await asset_store.apply_batch(
        changes, expected_revision=revision, idempotency_key="batch"
    ) == committed
    assert await asset_store.current_revision() == current
    assert await asset_store.get(key) == b"two"
    assert len(await asset_store._storage.primary.list_versions(key)) == 2
    assert await asset_store.read_versions((first, second)) == (b"one", b"two")
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
    assert (await asset_store._storage.primary.list_versions(key))[0].metadata == {"nested": {"mode": 1}}
    assert await asset_store.current_revision() == revision


@pytest.mark.asyncio
async def test_asset_stat_refreshes_an_outdated_revision_source(asset_store: AssetStore) -> None:
    key = AssetKey("resource", "external-change")
    await asset_store.put(key, b"old")
    assert await asset_store.stat(key) is not None
    backend = asset_store._storage.primary
    revision = await backend.head_revision()
    class StaleRevisionSource:
        async def head_revision(self):
            return revision
        async def revision_bumped(self, value):
            pass
    asset_store._storage.primary_view.revision_source = StaleRevisionSource()
    await backend.put(key, b"new")
    assert await asset_store.stat(key) == await backend.stat(key)


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
    assert len(await asset_store._storage.primary.list_versions(stable)) == 1
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
        restored_versions = await restored.resolve_versions((key,))
        assert restored_versions[0].revision == info.revision
        assert await restored.read_versions(restored_versions) == (b"",)
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_asset_digest_binds_metadata_receipts_versions_and_snapshot(asset_store: AssetStore) -> None:
    key = AssetKey("resource", "digest-contract")
    value = b"\x00asset-content\xff"
    digest = hashlib.sha256(value).hexdigest()
    result = await asset_store.apply_batch(
        (StorageChange(StorageOperation.PUT, key, value, None),),
        idempotency_key="digest-contract",
    )
    info = await asset_store.stat(key)
    assert info is not None and info.digest == digest
    receipt = await asset_store.batch_result("digest-contract")
    assert receipt == result
    assert receipt.results[0].info.digest == digest

    ref = (await asset_store.resolve_versions((key,)))[0]
    payload = {
        "version": 1,
        "kind": key.kind,
        "id": key.id,
        "layer_id": ref.layer_id,
        "revision": info.revision.value,
        "digest": digest,
        "size": len(value),
    }
    assert ref.to_payload() == payload
    assert AssetVersionRef.from_payload(payload) == ref
    assert AssetVersionRef(
        key=key, layer_id=ref.layer_id, revision=info.revision,
        digest=digest, size=len(value),
    ).matches_info(info)
    context = await CapabilityLoadContext.capture("application", asset_store)
    assert context.list() == (CapabilityLoadEntry(key=key, digest=digest, size=len(value)),)
    assert await context.read(key) == value

    objects = InMemoryObjectStore()
    snapshot = await asset_store.snapshot((key,), object_store=objects)
    manifest_bytes = await read_object(
        objects, snapshot.key, expected_digest=snapshot.digest, expected_size=snapshot.size,
    )
    assert snapshot.digest == hashlib.sha256(manifest_bytes).hexdigest()
    assert snapshot.key == f"v1/asset-snapshot/{snapshot.digest}"
    manifest = json.loads(manifest_bytes)
    entry = manifest["entries"][0]
    assert entry["digest"] == digest
    assert entry["content"]["digest"] == digest
    assert entry["content"]["key"] == f"v1/asset-content/{digest}"

    await asset_store.put(key, b"changed")
    assert await asset_store.read_versions((ref,)) == (value,)
    restored = AssetStore.from_snapshot(snapshot, object_store=objects)
    await restored.initialize()
    try:
        restored_info = await restored.stat(key)
        assert restored_info is not None and restored_info.digest == digest
        assert await restored.get(key) == value
    finally:
        await restored.close()
