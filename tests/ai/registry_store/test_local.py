import pytest

from linktools.ai.registry_store.local import InMemoryCapabilityCache, InMemoryCapabilityRepository
from linktools.ai.registry_store.protocols import CapabilityCacheProtocol, CapabilityRepositoryProtocol


@pytest.mark.asyncio
async def test_repository_upsert_then_get_roundtrips():
    repo = InMemoryCapabilityRepository()
    file_id, version, changed = await repo.upsert_file(
        kind="skill", file_path="demo/SKILL.md", content="hello", checksum="c1", updated_by="test",
    )
    assert changed is True
    assert version == 1

    row = await repo.get_file("skill", "demo/SKILL.md")
    assert row is not None
    assert row["content"] == "hello"
    assert row["id"] == file_id


@pytest.mark.asyncio
async def test_repository_upsert_same_checksum_is_noop():
    repo = InMemoryCapabilityRepository()
    await repo.upsert_file(kind="skill", file_path="demo/SKILL.md", content="hello", checksum="c1", updated_by="t")
    _, version, changed = await repo.upsert_file(kind="skill", file_path="demo/SKILL.md", content="hello", checksum="c1", updated_by="t")
    assert changed is False
    assert version == 1


@pytest.mark.asyncio
async def test_cache_try_acquire_is_exclusive():
    cache = InMemoryCapabilityCache()
    assert await cache.try_acquire("lock:a", "owner-1", ttl=30) is True
    assert await cache.try_acquire("lock:a", "owner-2", ttl=30) is False
    assert await cache.release_if_owner("lock:a", "owner-1") is True
    assert await cache.try_acquire("lock:a", "owner-2", ttl=30) is True


def test_implementations_satisfy_protocols():
    assert isinstance(InMemoryCapabilityRepository(), CapabilityRepositoryProtocol)
    assert isinstance(InMemoryCapabilityCache(), CapabilityCacheProtocol)
