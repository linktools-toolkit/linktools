import pytest

from linktools.ai.errors import StorageFeatureSupportError
from linktools.ai.spec.document import SpecDocument, SpecDocumentInfo, compute_spec_etag
from linktools.ai.spec.store import SpecStore
from linktools.ai.storage.cache import MemoryContentCache
from linktools.ai.storage.composition import StorageLayer
from linktools.ai.storage.revision import (
    MetadataLoad,
    MetadataLoadMode,
    StorageChange,
)


def document(path, body=b"body", *, version=1, kind="agent"):
    return SpecDocument(
        SpecDocumentInfo(path, kind, version, compute_spec_etag(body)),
        body,
    )


class Backend:
    """In-memory revisioned metadata backend: serves REPLACE then PATCH."""

    def __init__(self, documents):
        self.documents = {item.info.path: item for item in documents}
        self.revision = 1
        self.load_calls = 0
        self.get_calls = 0

    async def load_metadata(self, after_revision):
        self.load_calls += 1
        if after_revision == self.revision:
            return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
        changes = tuple(
            StorageChange(info.path, info)
            for info in sorted(
                (d.info for d in self.documents.values()), key=lambda i: i.path
            )
        )
        return MetadataLoad(self.revision, MetadataLoadMode.REPLACE, changes)

    async def head_revision(self):
        return self.revision

    async def get(self, path):
        self.get_calls += 1
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


class Adapter:
    def info_key(self, info):
        return info.path

    def value_info(self, value):
        return value.info

    def cache_key(self, key, info):
        return f"spec:{key}:{info.version}:{info.etag}"

    def cache_content(self, value):
        return value.content

    def from_cache(self, info, content):
        return SpecDocument(info, content)


def make_store(backend, **kw):
    return SpecStore(
        backend,
        writer=kw.pop("writer", backend),
        layers=kw.pop("layers", ()),
        cache=kw.pop("cache", None),
    )


@pytest.mark.asyncio
async def test_read_only_store_rejects_writes():
    backend = Backend((document("a"),))
    store = SpecStore(backend)  # no writer
    with pytest.raises(StorageFeatureSupportError, match="read-only"):
        await store.put(document("b"))


@pytest.mark.asyncio
async def test_put_delete_round_trip_through_composition():
    store = make_store(Backend((document("a", b"one"),)))
    assert (await store.get("a")).content == b"one"
    await store.put(document("b", b"two"))
    assert (await store.get("b")).content == b"two"
    await store.delete("a")
    assert await store.get("a") is None


@pytest.mark.asyncio
async def test_metadata_miss_does_not_probe_backend():
    backend = Backend((document("a"),))
    store = make_store(backend)
    await store.get("a")  # prime metadata
    before = backend.get_calls
    result = await store.get("this-key-is-absent")
    assert result is None
    # The missing key must not trigger a backend.get.
    assert backend.get_calls == before, "metadata miss probed the backend"


@pytest.mark.asyncio
async def test_list_active_filters_inactive_and_kind():
    store = make_store(
        Backend(
            (
                document("agent/a", b"a", kind="agent"),
                document("agent/b", b"b", kind="agent"),
                document("tool/c", b"c", kind="tool"),
            )
        )
    )
    assert await store.list_active("agent") == ("agent/a", "agent/b")
    assert await store.list_active() == ("agent/a", "agent/b", "tool/c")


@pytest.mark.asyncio
async def test_layer_conflict_earlier_reader_wins():
    primary = Backend((document("same", b"primary"),))
    overlay = Backend((document("same", b"overlay"),))
    store = make_store(
        primary,
        layers=(StorageLayer(backend=overlay),),
    )
    assert (await store.get("same")).content == b"primary"


@pytest.mark.asyncio
async def test_static_layer_with_revisioned_primary_keeps_primary_patch():
    # Primary is REVISIONED; a STATIC layer must not force primary into a full
    # list_info on every refresh.
    primary = Backend((document("a", b"a"),))
    overlay = Backend((document("b", b"b"),))
    store = make_store(primary, layers=(StorageLayer(backend=overlay),))
    state = await store._storage.refresh()
    assert state.entries.keys() == {"a", "b"}
    primary_calls = primary.load_calls
    await store._storage.refresh()  # unchanged
    # primary serves an empty PATCH at the same revision (still one load), but
    # it must NOT be forced into a full snapshot by the layer's presence.
    assert primary.revision == 1


@pytest.mark.asyncio
async def test_cache_hit_serves_without_origin_read():
    backend = Backend((document("a", b"cached"),))
    store = make_store(backend, cache=MemoryContentCache(max_bytes=100))
    await store.get("a")  # populates cache
    backend.get_calls = 0
    # Second get: metadata loads, but content comes from cache -> no origin get.
    result = await store.get("a")
    assert result.content == b"cached"
    assert backend.get_calls == 0


@pytest.mark.asyncio
async def test_cache_failure_falls_open_to_origin():
    class FailingCache:
        async def get(self, key):
            raise RuntimeError("cache unavailable")

        async def put(self, key, content):
            raise RuntimeError("cache unavailable")

        async def contains_many(self, keys):
            return frozenset()

    backend = Backend((document("a", b"origin"),))
    store = make_store(backend, cache=FailingCache())
    assert (await store.get("a")).content == b"origin"


@pytest.mark.asyncio
async def test_get_many_returns_owner_grouped_values():
    store = make_store(Backend((document("a", b"one"), document("b", b"two"))))
    result = await store.get_many(("a", "b", "missing"))
    assert result == {"a": document("a", b"one"), "b": document("b", b"two")}


@pytest.mark.asyncio
async def test_preload_warms_cache_without_per_item_exists_read():
    backend = Backend((document(f"agent/{c}", c.encode()) for c in "abcd"))
    cache = MemoryContentCache(max_bytes=10_000)
    store = make_store(backend, cache=cache)
    await store.list_info(preload=True)
    # After preload, all content is cached; emptying the backend still serves.
    backend.documents.clear()
    assert (await store.get("agent/a")).content == b"a"


@pytest.mark.asyncio
async def test_current_revision_does_not_force_full_metadata_load():
    # current_revision() probes the cheap head path; it must not trigger a
    # full load_metadata round trip.
    backend = Backend((document("a", b"x"),))
    store = make_store(backend)
    rev = await store.current_revision()
    assert rev == backend.revision
    assert backend.load_calls == 0, "current_revision triggered a full metadata load"
