import asyncio

import pytest

from linktools.ai.spec.document import SpecDocumentChange, SpecDocumentInfo
from linktools.ai.storage.multi import (
    MultiBackend,
    OverlayRefreshPolicy,
    StorageLayer,
)
from linktools.ai.storage.revision import (
    CompositeRevisionSource,
    MetadataSnapshot,
)


class Layer:
    def __init__(self, values, revision=1):
        self.values = dict(values)
        self.revision = revision
        self.full_reads = 0

    async def current_revision(self):
        return self.revision

    async def get(self, key):
        return self.values.get(key)

    async def list_info(self):
        self.full_reads += 1
        return tuple(self.values.values())


@pytest.mark.asyncio
async def test_primary_wins_and_overlay_fills_missing_keys():
    primary = Layer({"same": ("same", "primary"), "one": ("one", "one")})
    overlay = Layer({"same": ("same", "overlay"), "two": ("two", "two")})
    reader = MultiBackend(primary, (StorageLayer(overlay),))
    assert await reader.get("same") == ("same", "primary")
    assert await reader.get("two") == ("two", "two")
    assert await reader.list_info(key=lambda value: value[0]) == (
        ("one", "one"),
        ("same", "primary"),
        ("two", "two"),
    )


@pytest.mark.asyncio
async def test_get_many_fallback_is_bounded_and_primary_first():
    class SlowLayer(Layer):
        def __init__(self, values):
            super().__init__(values)
            self.active = 0
            self.peak = 0
            self.requested = []

        async def get(self, key):
            self.requested.append(key)
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return await super().get(key)

    primary = SlowLayer(
        {
            "primary": "primary",
            "shared": "primary-shared",
        }
    )
    overlay = SlowLayer(
        {
            "overlay": "overlay",
            "shared": "overlay-shared",
        }
    )
    reader = MultiBackend(primary, (StorageLayer(overlay),))
    assert await reader.get_many(
        ("primary", "shared", "overlay", "missing"),
        concurrency=2,
    ) == {
        "primary": "primary",
        "shared": "primary-shared",
        "overlay": "overlay",
    }
    assert primary.peak == 2
    assert overlay.peak == 2
    assert overlay.requested == ["overlay", "missing"]


@pytest.mark.asyncio
async def test_revisioned_overlay_contributes_to_composite_revision():
    primary = Layer({}, 1)
    overlay = Layer({}, 1)
    layer = StorageLayer(
        overlay,
        refresh=OverlayRefreshPolicy.REVISIONED,
        revision=overlay,
    )
    reader = MultiBackend(primary, (layer,))
    revisions = CompositeRevisionSource(primary, *reader.overlay_revisions)
    first = await revisions.current_revision()
    overlay.revision = 2
    assert await revisions.current_revision() != first


def test_revisioned_overlay_requires_explicit_revision_source():
    with pytest.raises(ValueError, match="requires a revision source"):
        StorageLayer(object(), refresh=OverlayRefreshPolicy.REVISIONED)


@pytest.mark.asyncio
async def test_unversioned_always_overlay_forces_full_snapshot_refresh():
    info = SpecDocumentInfo("a", "agent", 1, "e1")
    primary = Layer({"a": info})
    overlay = Layer({})
    reader = MultiBackend(
        primary,
        (StorageLayer(overlay, refresh=OverlayRefreshPolicy.ALWAYS),),
    )
    snapshot = MetadataSnapshot(
        reader,
        revision=primary,
        always_refresh=reader.always_refresh,
    )
    await snapshot.get("a")
    await snapshot.get("a")
    assert primary.full_reads == 2
    assert overlay.full_reads == 2


@pytest.mark.asyncio
async def test_always_overlay_does_not_publish_mixed_primary_revision():
    old = SpecDocumentInfo("old", "agent", 1, "old")
    new = SpecDocumentInfo("new", "agent", 1, "new")

    class ChangingPrimary(Layer):
        async def list_info(self):
            self.full_reads += 1
            if self.full_reads == 1:
                self.values["new"] = new
                self.revision += 1
                return (old,)
            return tuple(self.values.values())

    primary = ChangingPrimary({"old": old})
    overlay = Layer({})
    reader = MultiBackend(
        primary,
        (StorageLayer(overlay, refresh=OverlayRefreshPolicy.ALWAYS),),
    )
    snapshot = MetadataSnapshot(
        reader,
        revision=primary,
        always_refresh=True,
    )
    state = await snapshot.refresh()
    assert state is not None
    assert state.revision == 2
    assert set(state.entries) == {"old", "new"}
    assert primary.full_reads == 2


@pytest.mark.asyncio
async def test_unstable_revision_falls_back_to_uncached_repository_read():
    info = SpecDocumentInfo("a", "agent", 1, "e1")

    class UnstablePrimary(Layer):
        unstable = False

        async def list_info(self):
            values = await super().list_info()
            if self.unstable:
                self.revision += 1
            return values

    primary = UnstablePrimary({"a": info})
    snapshot = MetadataSnapshot(primary, revision=primary)
    first = await snapshot.refresh()
    primary.unstable = True
    primary.revision += 1
    fallback = await snapshot.refresh()
    assert fallback is not None
    assert fallback is not first
    assert fallback.entries == first.entries
    assert primary.full_reads == 5


@pytest.mark.asyncio
async def test_initial_unstable_revision_still_returns_repository_data():
    info = SpecDocumentInfo("a", "agent", 1, "e1")

    class UnstablePrimary(Layer):
        async def list_info(self):
            values = await super().list_info()
            self.revision += 1
            return values

    primary = UnstablePrimary({"a": info})
    snapshot = MetadataSnapshot(primary, revision=primary)
    state = await snapshot.refresh()
    assert state is not None
    assert state.entries == {"a": info}
    assert primary.full_reads == 4


@pytest.mark.asyncio
async def test_primary_only_snapshot_uses_explicit_delta_source():
    info = SpecDocumentInfo("a", "agent", 1, "e1")

    class Primary(Layer):
        def __init__(self):
            super().__init__({"a": info}, 1)
            self.delta_reads = 0

        async def list_changes(self, *, after_revision, through_revision):
            self.delta_reads += 1
            assert after_revision == 1
            assert through_revision == 2
            return (SpecDocumentChange(2, "a", info),)

    primary = Primary()
    snapshot = MetadataSnapshot(
        primary,
        revision=primary,
        changes=primary,
    )
    assert await snapshot.get("a") == info
    primary.revision = 2
    assert await snapshot.get("a") == info
    assert primary.delta_reads == 1


@pytest.mark.asyncio
async def test_unversioned_snapshot_reads_fresh_metadata_each_time():
    info = SpecDocumentInfo("a", "agent", 1, "e1")
    reader = Layer({"a": info})
    snapshot = MetadataSnapshot(reader)
    await snapshot.get("a")
    await snapshot.get("a")
    assert reader.full_reads == 2
