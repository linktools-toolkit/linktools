#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset snapshot utility tests: pinning a live object into an immutable
Artifact so retries/replay read the exact pinned bytes."""

import asyncio
import hashlib

import pytest

from linktools.ai.artifact import ArtifactStore
from linktools.ai.storage.coordination.process_local import InProcessKeyedCoordinator
from linktools.ai.storage.backends.memory.object import MemoryObjectBackend
from linktools.ai.storage.object.models import StorageKey, WriteOptions
from linktools.ai.storage.object.store import ObjectStore
from linktools.ai.jobs.snapshot import AssetSnapshotError, snapshot_asset
from linktools.ai.storage.filesystem.artifact import FilesystemArtifactBlobStore
from linktools.ai.artifact.persistence.filesystem import FilesystemArtifactRecordStore


def _stores(tmp_path) -> "tuple[ObjectStore, ArtifactStore]":
    assets = ObjectStore(primary=MemoryObjectBackend())
    artifacts = ArtifactStore(
        FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs"),
        FilesystemArtifactRecordStore(records_root=tmp_path / "records"),
        InProcessKeyedCoordinator(),
    )
    return assets, artifacts


def test_snapshot_asset_pins_content_and_metadata(tmp_path) -> None:
    assets, artifacts = _stores(tmp_path)

    async def run() -> None:
        await assets.put(
            StorageKey("/data/file.txt"),
            b"snapshot-me",
            options=WriteOptions(content_type="text/plain"),
        )
        snap = await snapshot_asset(
            assets, artifacts, "/data/file.txt", tenant_id="t1"
        )
        assert snap.path == "/data/file.txt"
        assert snap.version >= 1
        assert snap.etag
        assert snap.sha256 == hashlib.sha256(b"snapshot-me").hexdigest()
        # The artifact id is a per-write record id (UUID), distinct from the
        # content sha256 (the blob id).
        assert snap.artifact_id != snap.sha256
        assert snap.artifact_id.startswith("art-")
        # The pinned artifact is retrievable and integrity-verified.
        content = await artifacts.get(artifact_id=snap.artifact_id, tenant_id="t1")
        assert content == b"snapshot-me"

    asyncio.run(run())


def test_snapshot_missing_resource_raises(tmp_path) -> None:
    assets, artifacts = _stores(tmp_path)

    async def run() -> None:
        with pytest.raises(AssetSnapshotError):
            await snapshot_asset(assets, artifacts, "/missing", tenant_id="t1")

    asyncio.run(run())
