#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resource snapshot utility tests: pinning a live resource into an immutable
Artifact so retries/replay read the exact pinned bytes."""

import asyncio
import hashlib

import pytest

from linktools.ai.artifact import ArtifactStore
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.models import WriteOptions
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.resource.store import ResourceStore
from linktools.ai.task.snapshot import ResourceSnapshotError, snapshot_resource


def _stores() -> "tuple[ResourceStore, ArtifactStore]":
    resources = ResourceStore(primary=MemoryResourceBackend())
    return resources, ArtifactStore(resources)


def test_snapshot_resource_pins_content_and_metadata() -> None:
    resources, artifacts = _stores()

    async def run() -> None:
        await resources.put(
            ResourcePath("/data/file.txt"),
            b"snapshot-me",
            options=WriteOptions(content_type="text/plain"),
        )
        snap = await snapshot_resource(
            resources, artifacts, "/data/file.txt", tenant_id="t1"
        )
        assert snap.path == "/data/file.txt"
        assert snap.version >= 1
        assert snap.etag
        assert snap.sha256 == hashlib.sha256(b"snapshot-me").hexdigest()
        assert snap.artifact_id == snap.sha256
        # The pinned artifact is retrievable and integrity-verified.
        content = await artifacts.get(snap.artifact_id, tenant_id="t1")
        assert content == b"snapshot-me"

    asyncio.run(run())


def test_snapshot_missing_resource_raises() -> None:
    resources, artifacts = _stores()

    async def run() -> None:
        with pytest.raises(ResourceSnapshotError):
            await snapshot_resource(resources, artifacts, "/missing", tenant_id="t1")

    asyncio.run(run())
