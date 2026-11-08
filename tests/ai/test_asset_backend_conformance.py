#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Asset mutation, replay, and immutable metadata contracts."""

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from linktools.ai.asset import AssetKey, AssetStore, AssetVersionRef, InMemoryAssetBackend, SqlAssetBackend
from linktools.ai.capability import CapabilityLoadContext, CapabilityLoadEntry
from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, ObjectRef, PayloadPolicy, StorageChange, StorageOperation, StorageOverlay, read_object


@pytest_asyncio.fixture(params=("memory", "sqlite-inline", "sqlite-object", "sqlite-filesystem"))
async def asset_store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[AssetStore]:
    engine = None
    if request.param == "memory":
        backend = InMemoryAssetBackend()
    else:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
        await provision_asset_database(engine)
        backend = SqlAssetBackend(
            engine, namespace="conformance",
            object_store=FilesystemObjectStore(tmp_path / "objects") if request.param == "sqlite-filesystem" else None,
            payload_policy=None if request.param == "sqlite-inline" else PayloadPolicy(0),
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
    context = await CapabilityLoadContext.read_current("application", asset_store)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", (
    "fractional-size", "bool-size", "key-kind", "key-id", "entry-revision",
    "bool-entry-revision", "store-revision", "captured-revision", "content-size",
    "content-key", "metadata-omitted", "extra-fields",
))
async def test_asset_snapshot_preserves_known_field_types_and_open_metadata(corruption: str) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    objects = InMemoryObjectStore("snapshot")
    key = AssetKey("test", "123")
    await store.initialize()
    try:
        await store.put(key, b"x")
        original = await store.snapshot((key,), object_store=objects)
    finally:
        await store.close()
    manifest = json.loads(await read_object(
        objects, original.key, expected_digest=original.digest, expected_size=original.size,
    ))
    entry = manifest["entries"][0]
    if corruption == "fractional-size":
        entry["size"] = 1.5
    elif corruption == "bool-size":
        entry["size"] = True
    elif corruption == "key-kind":
        entry["key"]["kind"] = 123
    elif corruption == "key-id":
        entry["key"]["id"] = 123
    elif corruption == "entry-revision":
        entry["entry_revision"] = 1.5
    elif corruption == "bool-entry-revision":
        entry["entry_revision"] = True
    elif corruption == "store-revision":
        entry["store_revision"] = 123
    elif corruption == "captured-revision":
        manifest["captured_revision"] = 123
    elif corruption == "content-size":
        entry["content"]["size"] = True
    elif corruption == "content-key":
        content = b"x"

        async def content_chunks() -> AsyncIterator[bytes]:
            yield content

        await objects.put("123", content_chunks(), expected_size=1,
                          expected_digest=hashlib.sha256(content).hexdigest())
        entry["content"]["key"] = 123
    elif corruption == "metadata-omitted":
        del entry["metadata"]
    else:
        manifest["ordinary_extension"] = {"value": True}
        entry["ordinary_extension"] = {"value": 1}
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        yield payload

    await objects.put("modified-manifest", chunks(), expected_size=len(payload), expected_digest=digest)
    restored = AssetStore.from_snapshot(
        ObjectRef(objects.store_id, "modified-manifest", digest, len(payload)), object_store=objects,
    )
    try:
        if corruption in {"metadata-omitted", "extra-fields"}:
            await restored.initialize()
            info = await restored.stat(key)
            assert info is not None and info.size == 1 and info.metadata == {}
            assert await restored.get(key) == b"x"
        else:
            with pytest.raises(AIError) as rejected:
                await restored.initialize()
            assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await restored.close()
