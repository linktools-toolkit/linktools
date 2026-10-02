#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage overlay boundary regressions."""

from collections.abc import Sequence

import pytest
from linktools.ai.asset import AssetInfo, AssetKey, AssetRoot, InMemoryAssetBackend
from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import MetadataLoad, StorageChange, StorageOperation, StorageOverlay, StorageRevision


class _PartiallyMissingBatchBackend(InMemoryAssetBackend):
    def __init__(self, root: AssetRoot, missing: AssetKey) -> None:
        super().__init__(root)
        self.missing = missing
        self.metadata_loads = 0
        self.batch_reads = 0

    async def load_metadata(
        self,
        after_revision: "StorageRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo]":
        self.metadata_loads += 1
        return await super().load_metadata(after_revision)

    async def get_many(self, keys: Sequence[AssetKey]) -> "dict[AssetKey, bytes]":
        self.batch_reads += 1
        values = dict(await super().get_many(keys))
        values.pop(self.missing, None)
        return values


@pytest.mark.asyncio
async def test_batch_origin_mismatch_reports_the_still_missing_key() -> None:
    missing = AssetKey("sample", "missing")
    good = AssetKey("sample", "good")
    backend = _PartiallyMissingBatchBackend(
        AssetRoot("memory", "batch-mismatch"),
        missing,
    )
    await backend.put(good, b"good")
    await backend.put(missing, b"missing")
    storage = StorageOverlay(backend)

    with pytest.raises(AIError) as raised:
        await storage.get_many((good, missing))

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert raised.value.safe_details["storage_key_digest"] == canonical_sha256(str(missing))
    assert raised.value.safe_details["storage_key_digest"] != canonical_sha256(str(good))
    assert backend.metadata_loads == 2
    assert backend.batch_reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize("content", (b"old", b"new"))
async def test_overlay_write_keeps_intervening_external_entries(batch: bool, content: bytes) -> None:
    backend = InMemoryAssetBackend()
    storage = StorageOverlay(backend, writer=backend)
    own = AssetKey("resource", "own")
    external = AssetKey("resource", "external")
    await storage.initialize()
    try:
        await storage.put(own, b"old")
        assert await storage.get(own) == b"old"
        await backend.put(external, b"external")
        if batch:
            await storage.apply_batch((StorageChange(StorageOperation.PUT, own, content, None),))
        else:
            await storage.put(own, content)
        assert await storage.get(external) == b"external"
        assert await storage.get(own) == content
    finally:
        await storage.close()


class _RevisionSource:
    def __init__(self) -> None:
        self.revision = StorageRevision("0")

    async def head_revision(self) -> StorageRevision:
        return self.revision

    async def revision_bumped(self, revision: StorageRevision) -> None:
        self.revision = revision


@pytest.mark.asyncio
async def test_batch_replay_refreshes_metadata_after_unobserved_commit() -> None:
    backend = InMemoryAssetBackend()
    source = _RevisionSource()
    storage = StorageOverlay(backend, writer=backend, revision_source=source)
    key = AssetKey("resource", "committed")
    changes = (StorageChange(StorageOperation.PUT, key, b"value", None),)
    await storage.initialize()
    try:
        assert await storage.get(key) is None
        receipt = await backend.apply_batch(changes, idempotency_key="batch", request_digest="a" * 64)
        assert await storage.apply_batch(
            changes, idempotency_key="batch", request_digest="a" * 64
        ) == receipt
        assert await storage.get(key) == b"value"
    finally:
        await storage.close()
