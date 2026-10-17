#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision notifications belong to the backend that accepted the write."""

import pytest

from linktools.ai.asset import AssetKey, InMemoryAssetBackend
from linktools.ai.storage import StorageChange, StorageLayer, StorageOperation, StorageOverlay, StorageRevision


class _RevisionSource:
    def __init__(self, revision: StorageRevision) -> None:
        self.revision = revision
        self.notifications: list[StorageRevision] = []

    async def head_revision(self) -> StorageRevision:
        return self.revision

    async def revision_bumped(self, revision: StorageRevision) -> None:
        self.notifications.append(revision)
        self.revision = revision


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_is_primary", (False, True))
@pytest.mark.parametrize("operation", ("put", "delete", "reset", "batch"))
async def test_overlay_notifies_only_the_writers_revision_source(
    writer_is_primary: bool, operation: str,
) -> None:
    primary, fallback = InMemoryAssetBackend(), InMemoryAssetBackend()
    key = AssetKey("resource", "write")
    await primary.put(AssetKey("resource", "primary-a"), b"a")
    await primary.put(AssetKey("resource", "primary-b"), b"b")
    writer = primary if writer_is_primary else fallback
    if operation in {"delete", "reset"}:
        await writer.put(key, b"old")
    primary_revision = await primary.head_revision()
    assert primary_revision is not None
    source = _RevisionSource(primary_revision)
    overlay = StorageOverlay(primary, writer=writer, revision_source=source,
                             layers=(StorageLayer("fallback", fallback),))
    await overlay.initialize()
    try:
        source.notifications.clear()
        if operation == "batch":
            result = await overlay.apply_batch((StorageChange(StorageOperation.PUT, key, b"new", None),))
        elif operation == "put":
            result = await overlay.put(key, b"new")
        elif operation == "delete":
            result = await overlay.delete(key)
        else:
            result = await overlay.reset(key)
        if writer_is_primary:
            assert source.notifications == [result.store_revision]
            assert source.revision == await primary.head_revision()
        else:
            assert source.notifications == []
            assert source.revision == primary_revision == await primary.head_revision()
        assert await overlay.get(AssetKey("resource", "primary-a")) == b"a"
        assert await writer.head_revision() == result.store_revision
    finally:
        await overlay.close()
