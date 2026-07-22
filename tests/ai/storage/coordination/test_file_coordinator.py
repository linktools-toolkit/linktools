#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/coordination/test_file_coordinator.py"""

import pytest

from linktools.ai.storage.coordination.file import FileAssetCoordinator


@pytest.mark.asyncio
async def test_revision_hint_starts_none(tmp_path):
    coord = FileAssetCoordinator(root=tmp_path)
    assert await coord.revision_hint() is None


@pytest.mark.asyncio
async def test_publish_then_hint_roundtrip(tmp_path):
    coord = FileAssetCoordinator(root=tmp_path)
    await coord.publish_revision(5)
    reopened = FileAssetCoordinator(root=tmp_path)
    assert await reopened.revision_hint() == 5


@pytest.mark.asyncio
async def test_lock_is_exclusive_within_process(tmp_path):
    import asyncio

    coord = FileAssetCoordinator(root=tmp_path)
    order = []

    async def holder():
        async with coord.lock("k"):
            order.append("holder-acquired")
            await asyncio.sleep(0.05)
            order.append("holder-released")

    async def waiter():
        await asyncio.sleep(0.01)
        async with coord.lock("k"):
            order.append("waiter-acquired")

    await asyncio.gather(holder(), waiter())
    assert order == ["holder-acquired", "holder-released", "waiter-acquired"]


@pytest.mark.asyncio
async def test_lock_provides_exclusion_across_separate_coordinator_instances(tmp_path):
    import asyncio

    coord_a = FileAssetCoordinator(root=tmp_path)
    coord_b = FileAssetCoordinator(root=tmp_path)
    order = []

    async def holder():
        async with coord_a.lock("k"):
            order.append("holder-acquired")
            await asyncio.sleep(0.05)
            order.append("holder-released")

    async def waiter():
        await asyncio.sleep(0.01)
        async with coord_b.lock("k"):
            order.append("waiter-acquired")

    await asyncio.gather(holder(), waiter())
    assert order == ["holder-acquired", "holder-released", "waiter-acquired"]
