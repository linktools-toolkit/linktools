#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage overlay boundary regressions."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

import pytest
from linktools.ai.asset import AssetInfo, AssetKey, AssetRoot, InMemoryAssetBackend
from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import MetadataLoad, StorageChange, StorageLayer, StorageOperation, StorageOverlay, StorageRevision


from tests.ai._storage_stream_helpers import collect_values


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

    async def get_many(self, keys: Sequence[AssetKey]) -> "AsyncIterator[bytes | None]":
        self.batch_reads += 1
        values = await collect_values(super().get_many(keys))
        for key, value in zip(keys, values, strict=True):
            yield None if key == self.missing else value


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
        await collect_values(storage.get_many((good, missing)))

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
@pytest.mark.parametrize("operation", ("put", "delete", "reset", "reset_fallback"))
async def test_stat_refreshes_changed_authoritative_metadata(operation: str) -> None:
    primary, fallback = InMemoryAssetBackend(), InMemoryAssetBackend()
    source = _RevisionSource()
    key = AssetKey("resource", "changed")
    if operation == "reset_fallback":
        await fallback.put(key, b"fallback")
    storage = StorageOverlay(primary, writer=primary, revision_source=source,
                             layers=(StorageLayer("fallback", fallback),))
    await storage.initialize()
    try:
        await storage.put(key, b"old")
        assert await storage.stat(key) == await primary.stat(key)
        if operation == "put":
            await primary.put(key, b"new")
        elif operation == "delete":
            await primary.delete(key)
        else:
            await primary.reset(key)
        expected = await primary.stat(key) if operation in {"put", "delete"} else await fallback.stat(key)
        assert await storage.stat(key) == expected
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("mismatch", "missing", "transport"))
async def test_stat_preserves_real_origin_failures(failure: str) -> None:
    class BrokenStatBackend(InMemoryAssetBackend):
        broken = False
        async def stat(self, key):
            info = await super().stat(key)
            if not self.broken:
                return info
            if failure == "transport":
                raise OSError("origin transport failed")
            return None if failure == "missing" else replace(info, size=info.size + 1)
    backend = BrokenStatBackend()
    storage = StorageOverlay(backend, writer=backend)
    key = AssetKey("resource", "broken")
    await storage.initialize()
    try:
        await storage.put(key, b"value")
        assert await storage.stat(key) is not None
        backend.broken = True
        if failure == "transport":
            with pytest.raises(OSError, match="origin transport failed"):
                await storage.stat(key)
        else:
            with pytest.raises(AIError) as caught:
                await storage.stat(key)
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_stat_does_not_hot_reload_a_static_layer() -> None:
    primary, fallback = InMemoryAssetBackend(), InMemoryAssetBackend()
    key = AssetKey("resource", "static")
    await fallback.put(key, b"old")
    storage = StorageOverlay(primary, layers=(StorageLayer("fallback", fallback),))
    await storage.initialize()
    try:
        assert await storage.stat(key) == await fallback.stat(key)
        await fallback.put(key, b"new")
        with pytest.raises(AIError) as caught:
            await storage.stat(key)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("continuous", (False, True))
async def test_stat_bounds_mutations_between_metadata_and_origin(continuous: bool) -> None:
    class RacingBackend(InMemoryAssetBackend):
        armed = False
        mutations = 0
        async def stat(self, key):
            if self.armed:
                self.mutations += 1
                await self.put(key, str(self.mutations).encode())
                self.armed = continuous
            return await super().stat(key)
    backend = RacingBackend()
    storage = StorageOverlay(backend, writer=backend)
    key = AssetKey("resource", "racing")
    await storage.initialize()
    try:
        await storage.put(key, b"old")
        assert await storage.stat(key) is not None
        backend.armed = True
        if continuous:
            with pytest.raises(AIError) as caught:
                await storage.stat(key)
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert backend.mutations <= 2
        else:
            assert await storage.stat(key) == await backend.stat(key)
    finally:
        await storage.close()


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


def test_overlay_does_not_expose_layer_ambiguous_numeric_history() -> None:
    from linktools.ai.storage import VersionedStorage

    backend = InMemoryAssetBackend()
    storage = StorageOverlay(backend, writer=backend)
    assert isinstance(backend, VersionedStorage)
    assert not isinstance(storage, VersionedStorage)
