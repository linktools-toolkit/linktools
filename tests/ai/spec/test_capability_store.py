import pytest

from linktools.ai.errors import StorageFeatureSupportError
from linktools.ai.spec.document import SpecDocument, SpecDocumentInfo
from linktools.ai.spec.store import SpecStore
from linktools.ai.spec.cache import MemoryContentCache
from linktools.ai.spec.multi import StorageLayer


class Backend:
    def __init__(self, documents):
        self.documents = {item.info.path: item for item in documents}
        self.revision = 1

    async def current_revision(self):
        return self.revision

    async def list_info(self):
        return tuple(item.info for item in self.documents.values())

    async def list_changes(self, *, after_revision, through_revision):
        return ()

    async def get(self, path):
        return self.documents.get(path)

    async def put(self, document):
        self.documents[document.info.path] = document
        self.revision += 1
        return document

    async def delete(self, path):
        self.documents.pop(path, None)
        self.revision += 1

    async def reset(self, documents):
        self.documents = {item.info.path: item for item in documents}
        self.revision += 1


def document(path, body=b"body", *, version=1):
    return SpecDocument(
        SpecDocumentInfo(path, "agent", version, f"e{version}"),
        body,
    )


@pytest.mark.asyncio
async def test_unversioned_reader_refreshes_and_uses_content_cache():
    backend = Backend((document("a"),))
    store = SpecStore(
        backend,
        content_cache=MemoryContentCache(max_bytes=100),
    )
    assert (await store.get("a")).content == b"body"
    backend.documents["b"] = document("b")
    assert await store.list_active("agent") == ("a", "b")


@pytest.mark.asyncio
async def test_writer_is_an_explicit_optional_capability():
    backend = Backend((document("a"),))
    read_only = SpecStore(backend)
    with pytest.raises(StorageFeatureSupportError, match="read-only"):
        await read_only.put(document("b"))

    writable = SpecStore(backend, writer=backend)
    await writable.put(document("b"))
    assert await backend.get("b") == document("b")


@pytest.mark.asyncio
async def test_static_overlay_is_remerged_when_primary_revision_changes():
    primary = Backend((document("same", b"primary"),))
    overlay = Backend((document("same", b"overlay"),))
    store = SpecStore(
        primary,
        overlays=(StorageLayer(overlay),),
        revision=primary,
        changes=primary,
    )
    assert (await store.get("same")).content == b"primary"
    await primary.delete("same")
    assert (await store.get("same")).content == b"overlay"


@pytest.mark.asyncio
async def test_cache_failure_never_changes_backend_result():
    class FailingCache:
        async def get(self, key):
            raise RuntimeError("cache unavailable")

        async def put(self, key, content):
            raise RuntimeError("cache unavailable")

    backend = Backend((document("a"),))
    store = SpecStore(backend, content_cache=FailingCache())
    assert await store.get("a") == document("a")


@pytest.mark.asyncio
async def test_unstable_revision_fallback_bypasses_content_cache():
    class UnstableBackend(Backend):
        async def list_info(self):
            values = await super().list_info()
            self.revision += 1
            return values

    class StaleCache:
        reads = 0
        writes = 0

        async def get(self, key):
            self.reads += 1
            return b"stale"

        async def put(self, key, content):
            self.writes += 1

    backend = UnstableBackend((document("a", b"origin"),))
    cache = StaleCache()
    store = SpecStore(backend, revision=backend, content_cache=cache)
    assert (await store.get("a")).content == b"origin"
    assert cache.reads == 0
    assert cache.writes == 0


@pytest.mark.asyncio
async def test_list_active_can_preload_content_cache():
    entry = document("agent/a", b"cached")
    backend = Backend((entry,))
    store = SpecStore(
        backend,
        revision=backend,
        content_cache=MemoryContentCache(max_bytes=100),
    )
    assert await store.list_active("agent", preload=True) == ("agent/a",)
    backend.documents.clear()
    assert await store.get("agent/a") == entry
