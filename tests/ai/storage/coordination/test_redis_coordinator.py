#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/coordination/test_redis_coordinator.py"""

import pytest

redis = pytest.importorskip("redis")
fakeredis = pytest.importorskip("fakeredis")

from linktools.ai.storage.coordination.redis import RedisResourceCoordinator  # noqa: E402


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


@pytest.mark.asyncio
async def test_lock_release_does_not_remove_a_different_holders_lock(redis_client):
    coord = RedisResourceCoordinator(redis=redis_client)
    lock_key = coord._lock_key("k")

    async with coord.lock("k"):
        # Simulate: this holder's lock TTL expired and a different holder has since
        # acquired the same key with a different token, all while we still believe
        # we're inside our own critical section.
        await redis_client.set(lock_key, "someone-elses-token")

    # The coordinator's own release path (inside lock()'s finally block) must not
    # have deleted the other holder's value, since the token it holds no longer
    # matches what's currently stored.
    assert await redis_client.get(lock_key) == b"someone-elses-token"
