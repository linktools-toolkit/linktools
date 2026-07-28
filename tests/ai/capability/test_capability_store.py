import pytest

from linktools.ai.capability.entries import CapabilityEntry, CapabilityEntryInfo
from linktools.ai.capability.store import CapabilityStore
from linktools.ai.storage.cache import MemoryContentCache


class Repository:
    async def current_revision(self):
        return 1

    async def list_info(self, *, kind=None):
        return (CapabilityEntryInfo("a", "agent", 1, "etag"),)

    async def list_changes(self, *, after_revision, through_revision):
        return ()

    async def get(self, path):
        return CapabilityEntry(CapabilityEntryInfo(path, "agent", 1, "etag"), b"body")

    async def stat(self, path):
        return CapabilityEntryInfo(path, "agent", 1, "etag")


@pytest.mark.asyncio
async def test_capability_store_uses_metadata_identity_before_content_cache():
    store = CapabilityStore(Repository(), content_cache=MemoryContentCache(max_bytes=100))
    assert (await store.get("a")).content == b"body"
    assert (await store.list_active("agent")) == ("a",)
