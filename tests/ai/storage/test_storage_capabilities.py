from dataclasses import dataclass

import pytest

from linktools.ai.errors import StorageCapabilityError
from linktools.ai.storage.cache import MemoryContentCache
from linktools.ai.storage.composition import StorageComposition
from linktools.ai.storage.multi import OverlayRefreshPolicy, StorageLayer


class Source:
    def __init__(self):
        self.revision = 2

    async def current_revision(self):
        return self.revision

    async def list_changes(self, *, after_revision, through_revision):
        return tuple(range(after_revision + 1, through_revision + 1))

    async def get(self, key):
        return None

    async def list_info(self):
        return ()


def test_storage_composition_only_records_explicit_capabilities():
    source = Source()
    composition = StorageComposition(primary=source, writer=source)
    assert composition.backend is source
    assert composition.writer is source


@dataclass(frozen=True)
class Info:
    key: str
    version: int


@dataclass(frozen=True)
class Value:
    info: Info
    content: bytes


@dataclass(frozen=True)
class Change:
    key: str
    info: Info | None


class Reader:
    def __init__(self, values, revision=1):
        self.values = {value.info.key: value for value in values}
        self.revision = revision
        self.changes = ()
        self.batch_reads = []

    async def current_revision(self):
        return self.revision

    async def list_changes(self, *, after_revision, through_revision):
        return self.changes

    async def get(self, key):
        return self.values.get(key)

    async def list_info(self):
        return tuple(value.info for value in self.values.values())

    async def get_many(self, keys):
        self.batch_reads.append(keys)
        return {
            key: self.values[key]
            for key in keys
            if key in self.values
        }


class Adapter:
    def info_key(self, info):
        return info.key

    def change_key(self, change):
        return change.key

    def change_value(self, change):
        return change.info

    def value_info(self, value):
        return value.info

    def cache_key(self, key, info):
        return key, info.version, str(info.version)

    def cache_content(self, value):
        return value.content

    def from_cache(self, info, content):
        return Value(info, content)


@pytest.mark.asyncio
async def test_composition_owns_layers_revision_snapshot_and_cache():
    primary_value = Value(Info("shared", 1), b"primary")
    overlay_shared = Value(Info("shared", 2), b"overlay-shared")
    overlay_value = Value(Info("overlay", 1), b"overlay")
    primary = Reader((primary_value,))
    overlay = Reader((overlay_shared, overlay_value))
    composition = StorageComposition(
        primary,
        overlays=(
            StorageLayer(
                overlay,
                refresh=OverlayRefreshPolicy.REVISIONED,
                revision=overlay,
            ),
        ),
        revision=primary,
        changes=primary,
        cache=MemoryContentCache(max_bytes=100),
        adapter=Adapter(),
        cache_adapter=Adapter(),
    )

    first_revision = await composition.current_revision()
    assert await composition.list_info() == (
        overlay_value.info,
        primary_value.info,
    )
    assert await composition.get("shared") == primary_value
    primary.values.clear()
    primary.revision += 1
    assert await composition.get("shared") == overlay_shared
    overlay.revision += 1
    assert await composition.current_revision() != first_revision


@pytest.mark.asyncio
async def test_composition_initializes_independent_revision_and_change_sources():
    class Revision:
        initialized_with = None

        async def initialize_storage(self, value):
            self.initialized_with = value

        async def current_revision(self):
            return 1

    class Changes:
        initialized_with = None

        async def initialize_storage(self, value):
            self.initialized_with = value

        async def list_changes(self, *, after_revision, through_revision):
            return ()

    revision = Revision()
    changes = Changes()
    composition = StorageComposition(
        Reader(()),
        revision=revision,
        changes=changes,
        adapter=Adapter(),
    )
    await composition.initialize("configured")
    assert revision.initialized_with == "configured"
    assert changes.initialized_with == "configured"


@pytest.mark.asyncio
async def test_metadata_preload_batches_initial_and_changed_content():
    first = Value(Info("a", 1), b"a1")
    unchanged = Value(Info("b", 1), b"b1")
    reader = Reader((first, unchanged))
    cache = MemoryContentCache(max_bytes=100)
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        revision=reader,
        changes=reader,
        cache=cache,
        adapter=adapter,
        cache_adapter=adapter,
    )

    assert await composition.list_info(preload=True) == (
        first.info,
        unchanged.info,
    )
    assert reader.batch_reads == [("a", "b")]

    updated = Value(Info("a", 2), b"a2")
    reader.values["a"] = updated
    reader.revision = 2
    reader.changes = (Change("a", updated.info),)
    assert await composition.list_info(preload=True) == (
        updated.info,
        unchanged.info,
    )
    assert reader.batch_reads == [("a", "b"), ("a",)]

    reader.values.clear()
    assert await composition.get("a") == updated
    assert await composition.get("b") == unchanged


@pytest.mark.asyncio
async def test_unstable_metadata_does_not_preload_content():
    class UnstableReader(Reader):
        async def list_info(self):
            values = await super().list_info()
            self.revision += 1
            return values

    value = Value(Info("a", 1), b"a")
    reader = UnstableReader((value,))
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        revision=reader,
        cache=MemoryContentCache(max_bytes=100),
        adapter=adapter,
        cache_adapter=adapter,
    )
    state = await composition.refresh(preload=True)
    assert state is not None
    assert state.cacheable is False
    assert reader.batch_reads == []


@pytest.mark.asyncio
async def test_preload_requires_revision_source():
    value = Value(Info("a", 1), b"a")
    reader = Reader((value,))
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        cache=MemoryContentCache(max_bytes=100),
        adapter=adapter,
        cache_adapter=adapter,
    )
    with pytest.raises(
        StorageCapabilityError,
        match="preload requires a revision source",
    ):
        await composition.refresh(preload=True)


@pytest.mark.asyncio
async def test_preload_retries_when_revision_changes_during_batch_read():
    initial = Value(Info("a", 1), b"a1")
    updated = Value(Info("a", 2), b"a2")

    class ChangingReader(Reader):
        changed = False

        async def get_many(self, keys):
            values = await super().get_many(keys)
            if not self.changed:
                self.changed = True
                self.values["a"] = updated
                self.changes = (Change("a", updated.info),)
                self.revision += 1
            return values

    reader = ChangingReader((initial,))
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        revision=reader,
        changes=reader,
        cache=MemoryContentCache(max_bytes=100),
        adapter=adapter,
        cache_adapter=adapter,
    )
    state = await composition.refresh(preload=True)
    assert state is not None
    assert state.entries == {"a": updated.info}
    assert reader.batch_reads == [("a",), ("a",)]
    reader.values.clear()
    assert await composition.get("a") == updated


@pytest.mark.asyncio
async def test_preload_cache_failure_is_best_effort_and_retryable():
    class FailingCache:
        async def get(self, key):
            return None

        async def put(self, key, content):
            raise RuntimeError("cache unavailable")

    value = Value(Info("a", 1), b"a")
    reader = Reader((value,))
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        revision=reader,
        cache=FailingCache(),
        adapter=adapter,
        cache_adapter=adapter,
    )
    assert await composition.list_info(preload=True) == (value.info,)
    assert await composition.list_info(preload=True) == (value.info,)
    assert reader.batch_reads == [("a",), ("a",)]


@pytest.mark.asyncio
async def test_content_revision_change_to_unstable_snapshot_never_caches():
    value = Value(Info("a", 1), b"a")

    class ChangingThenUnstableReader(Reader):
        unstable_metadata = False

        async def list_info(self):
            values = await super().list_info()
            if self.unstable_metadata:
                self.revision += 1
            return values

        async def get_many(self, keys):
            values = await super().get_many(keys)
            self.unstable_metadata = True
            self.revision += 1
            return values

    reader = ChangingThenUnstableReader((value,))
    cache = MemoryContentCache(max_bytes=100)
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        revision=reader,
        cache=cache,
        adapter=adapter,
        cache_adapter=adapter,
    )
    state = await composition.refresh(preload=True)
    assert state is not None
    assert state.cacheable is False
    assert reader.batch_reads == [("a",)]
    assert await cache.get(adapter.cache_key("a", value.info)) is None


@pytest.mark.asyncio
async def test_preload_restores_content_evicted_from_cache():
    first = Value(Info("a", 1), b"a")
    second = Value(Info("b", 1), b"b")
    reader = Reader((first, second))
    cache = MemoryContentCache(max_bytes=2)
    adapter = Adapter()
    composition = StorageComposition(
        reader,
        revision=reader,
        cache=cache,
        adapter=adapter,
        cache_adapter=adapter,
    )
    await composition.refresh(preload=True)
    await cache.put(("external", 1, "1"), b"x")
    await composition.refresh(preload=True)
    assert reader.batch_reads == [("a", "b"), ("a",)]
