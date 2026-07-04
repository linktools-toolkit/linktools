#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/coordination/test_redis_coordinator.py"""
import pytest

redis = pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from linktools.ai.storage.coordination.redis import RedisResourceCoordinator


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis()


@pytest.mark.asyncio
async def test_revision_hint_starts_none(redis_client):
    coord = RedisResourceCoordinator(redis=redis_client)
    assert await coord.revision_hint() is None


@pytest.mark.asyncio
async def test_publish_then_hint_roundtrip(redis_client):
    coord = RedisResourceCoordinator(redis=redis_client)
    await coord.publish_revision(7)
    assert await coord.revision_hint() == 7


@pytest.mark.asyncio
async def test_lock_is_exclusive(redis_client):
    import asyncio
    coord = RedisResourceCoordinator(redis=redis_client)
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
