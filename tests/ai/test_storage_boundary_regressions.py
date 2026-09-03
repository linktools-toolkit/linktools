#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage overlay boundary regressions."""

from collections.abc import Sequence

import pytest
from linktools.ai.asset import AssetInfo, AssetKey, AssetRoot, InMemoryAssetBackend
from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import MetadataLoad, StorageOverlay, StorageRevision


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
        AssetRoot("memory:batch-mismatch", "memory", "batch-mismatch", "batch-mismatch"),
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
