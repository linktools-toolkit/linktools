#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageComposition capability recording and layer merge.

Validates that composition records only the features explicitly given, merges
layers primary-first with owner tracking, and reports an effective revision
without re-querying backends."""

from dataclasses import dataclass

import pytest

from linktools.ai.errors import StorageFeatureSupportError
from linktools.ai.storage.composition import StorageComposition, StorageLayer
from linktools.ai.storage.revision import (
    LayerRefreshPolicy,
    MetadataLoad,
    MetadataLoadMode,
    StorageChange,
)


@dataclass(frozen=True)
class Info:
    path: str


@dataclass(frozen=True)
class Doc:
    info: Info
    content: bytes


class Adapter:
    def info_key(self, info):
        return info.path

    def value_info(self, value):
        return value.info

    def cache_key(self, key, info):
        return f"k:{key}"

    def cache_content(self, value):
        return value.content

    def from_cache(self, info, content):
        return Doc(info, content)


class MetadataBackend:
    def __init__(self, docs, revision=1):
        self.docs = {d.info.path: d for d in docs}
        self.revision = revision
        self.loads = 0

    async def load_metadata(self, after_revision):
        self.loads += 1
        if after_revision == self.revision:
            return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
        changes = tuple(
            StorageChange(d.info.path, d.info)
            for d in sorted(self.docs.values(), key=lambda d: d.info.path)
        )
        return MetadataLoad(self.revision, MetadataLoadMode.REPLACE, changes)

    async def head_revision(self):
        return self.revision

    async def get(self, path):
        return self.docs.get(path)

    async def put(self, doc):
        self.docs[doc.info.path] = doc
        self.revision += 1
        return doc

    async def delete(self, path):
        self.docs.pop(path, None)
        self.revision += 1

    async def reset(self, docs):
        self.docs = {d.info.path: d for d in docs}
        self.revision += 1


def _doc(path, body=b"b"):
    return Doc(Info(path), body)


def test_composition_records_only_explicit_features():
    primary = MetadataBackend((_doc("a"),))
    composition = StorageComposition(primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter())
    assert composition.writer is primary
    assert composition.adapter is not None
    assert composition.layers == ()


def test_composition_requires_adapter_for_features():
    primary = MetadataBackend((_doc("a"),))
    with pytest.raises(ValueError, match="adapter"):
        StorageComposition(primary, layers=(StorageLayer(backend=primary),))


def test_read_only_composition_requires_writer_to_write():
    primary = MetadataBackend((_doc("a"),))
    composition = StorageComposition(primary, adapter=Adapter(), cache_adapter=Adapter())
    with pytest.raises(StorageFeatureSupportError, match="read-only"):
        composition.require_writer()


@pytest.mark.asyncio
async def test_layer_merge_records_owner_and_earlier_wins():
    primary = MetadataBackend((_doc("same", b"primary"), _doc("only-primary", b"pp"),))
    layer = MetadataBackend((_doc("same", b"layer"), _doc("only-layer", b"ll"),))
    composition = StorageComposition(
        primary,
        layers=(StorageLayer(backend=layer),),
        adapter=Adapter(),
        cache_adapter=Adapter(),
    )
    state = await composition.refresh()
    assert state.owners["same"] == 0  # primary wins
    assert state.owners["only-layer"] == 1
    assert (await composition.get("same")).content == b"primary"
    assert (await composition.get("only-layer")).content == b"ll"


@pytest.mark.asyncio
async def test_effective_revision_single_primary_is_primary_revision():
    primary = MetadataBackend((_doc("a"),), revision=7)
    composition = StorageComposition(primary, adapter=Adapter(), cache_adapter=Adapter())
    state = await composition.refresh()
    assert state.revision == 7


@pytest.mark.asyncio
async def test_effective_revision_multi_layer_is_hash_of_loaded():
    primary = MetadataBackend((_doc("a"),), revision=3)
    layer = MetadataBackend((_doc("b"),), revision=5)
    composition = StorageComposition(
        primary,
        layers=(StorageLayer(backend=layer),),
        adapter=Adapter(),
        cache_adapter=Adapter(),
    )
    s1 = await composition.refresh()
    s2 = await composition.refresh()
    # No change -> same hash revision.
    assert s1.revision == s2.revision
    assert isinstance(s1.revision, str)  # hashed, not a bare int


@pytest.mark.asyncio
async def test_revisioned_layer_keeps_independent_patch_from_primary():
    primary = MetadataBackend((_doc("a"),))
    layer = MetadataBackend((_doc("b"),))
    composition = StorageComposition(
        primary,
        layers=(StorageLayer(backend=layer, refresh=LayerRefreshPolicy.REVISIONED),),
        adapter=Adapter(),
        cache_adapter=Adapter(),
    )
    await composition.refresh()
    captured: list = []
    orig = layer.load_metadata

    async def spy(after_revision):
        load = await orig(after_revision)
        captured.append(load.mode)
        return load

    layer.load_metadata = spy
    # Mutating primary must not force the layer into a REPLACE (full reload);
    # the layer still serves PATCH (unchanged revision -> empty PATCH).
    await primary.put(_doc("c"))
    await composition.refresh()
    assert captured, "layer was not consulted at all"
    assert all(mode is not MetadataLoadMode.REPLACE for mode in captured), captured


@pytest.mark.asyncio
async def test_current_revision_probes_head_without_loading_entries():
    # current_revision() must use the cheap head probe, not a full load_metadata:
    # it returns the revision with zero backend loads.
    primary = MetadataBackend((_doc("a"),), revision=9)
    composition = StorageComposition(primary, adapter=Adapter(), cache_adapter=Adapter())
    rev = await composition.current_revision()
    assert rev == 9
    assert primary.loads == 0, "current_revision triggered a full metadata load"


@pytest.mark.asyncio
async def test_current_revision_multi_layer_probes_each_head():
    primary = MetadataBackend((_doc("a"),), revision=3)
    layer = MetadataBackend((_doc("b"),), revision=5)
    composition = StorageComposition(
        primary,
        layers=(StorageLayer(backend=layer, refresh=LayerRefreshPolicy.REVISIONED),),
        adapter=Adapter(),
        cache_adapter=Adapter(),
    )
    rev = await composition.current_revision()
    # Hashed across both heads; neither backend was fully loaded.
    assert isinstance(rev, str)
    assert primary.loads == 0
    assert layer.loads == 0


@pytest.mark.asyncio
async def test_current_revision_falls_back_to_refresh_for_always_layer():
    # An ALWAYS layer has no cheap head probe (head_revision -> None); the
    # composition must fall back to a full refresh for an accurate revision.
    primary = MetadataBackend((_doc("a"),), revision=2)

    class AlwaysBackend:
        async def list_info(self):
            return (_doc("b").info,)

    composition = StorageComposition(
        primary,
        layers=(StorageLayer(backend=AlwaysBackend()),),
        adapter=Adapter(),
        cache_adapter=Adapter(),
    )
    # Falls back: a full refresh runs, so primary is loaded once.
    rev = await composition.current_revision()
    assert rev != 0  # something was loaded
    assert primary.loads >= 1


@pytest.mark.asyncio
async def test_get_short_circuits_repeated_refresh_via_default_source():
    # A plain StorageComposition (no explicit revision_source) auto-wires the
    # default _BackendHeadRevisionSource on the primary. So after the first
    # get() primes metadata, subsequent get()s reuse the held state via the
    # head probe and issue ZERO load_metadata calls.
    primary = MetadataBackend((_doc("a", b"one"),))
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter()
    )
    await composition.get("a")  # primes metadata (REPLACE) -> 1 load
    assert primary.loads == 1
    await composition.get("a")  # head probe says unchanged -> short-circuit
    await composition.get("a")
    assert primary.loads == 1, f"repeated get() must not reload, got {primary.loads}"


@pytest.mark.asyncio
async def test_get_reloads_after_invalidate_until_reprimed():
    # After invalidate() (the post-write path), the held state is gone so the
    # next get() must reload; then short-circuit resumes.
    primary = MetadataBackend((_doc("a", b"one"),))
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter()
    )
    await composition.get("a")
    assert primary.loads == 1
    composition.primary_view.invalidate()  # simulate post-write invalidation
    await composition.get("a")  # held state dropped -> reload
    assert primary.loads == 2
    await composition.get("a")  # re-primed -> short-circuit again
    assert primary.loads == 2


class _RecordingRevisionSource:
    """RevisionSource double: head_revision() returns a held revision (or
    None); revision_bumped records every call so a test asserts post-write
    notification."""

    def __init__(self, revision=None):
        self.held = revision
        self.bumps = []

    async def head_revision(self):
        return self.held

    async def revision_bumped(self, revision):
        self.bumps.append(revision)
        self.held = revision


@pytest.mark.asyncio
async def test_put_notifies_revision_source_post_commit():
    # After a put commits, the composition probes the new head revision once and
    # calls source.revision_bumped(N) so a caching source can refresh/publish.
    primary = MetadataBackend((_doc("a", b"one"),))
    source = _RecordingRevisionSource()
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter(),
        revision_source=source,
    )
    before = primary.revision
    await composition.put(_doc("b", b"two"))
    # The fake put bumps revision by 1; the source must be told the new value.
    assert source.bumps == [primary.revision]
    assert source.bumps[0] == before + 1


@pytest.mark.asyncio
async def test_delete_and_reset_also_notify_revision_source():
    primary = MetadataBackend((_doc("a", b"one"), _doc("b", b"two")))
    source = _RecordingRevisionSource()
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter(),
        revision_source=source,
    )
    await composition.delete("a")
    await composition.reset((_doc("c", b"three"),))
    # Both writes bumped the revision and both notified the source.
    assert len(source.bumps) == 2


@pytest.mark.asyncio
async def test_no_source_means_no_notification_overhead():
    # With no revision_source injected, writes still work and the auto-wired
    # default source's revision_bumped is a no-op (it reads head live).
    primary = MetadataBackend((_doc("a", b"one"),))
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter(),
    )
    await composition.put(_doc("b", b"two"))  # must not raise
    assert composition.primary_view.revision_source is not None


@pytest.mark.asyncio
async def test_get_retry_invalidates_on_content_info_mismatch():
    # When content read races the held metadata, _get_with_retry must
    # invalidate the owner view so the recursive refresh loads fresh metadata
    # (bypassing the revision-source short-circuit) instead of re-serving the
    # stale state. To exercise the bug, a STALE revision source is injected:
    # it reports the OLD revision even after content changed, so a plain
    # refresh() would short-circuit and return the stale state forever. Only
    # invalidate() (which drops _state, bypassing the short-circuit) recovers.

    @dataclass(frozen=True)
    class EtagInfo:
        path: str
        etag: str

    @dataclass(frozen=True)
    class EtagDoc:
        info: EtagInfo
        content: bytes

    class EtagAdapter:
        def info_key(self, info):
            return info.path

        def value_info(self, value):
            return value.info

        def cache_key(self, key, info):
            return f"k:{key}:{info.etag}"

        def cache_content(self, value):
            return value.content

        def from_cache(self, info, content):
            return EtagDoc(info, content)

    class EtagBackend:
        """Backend whose metadata evolves: first load reports etag 'v1' at
        revision 1; after the bump it reports etag 'v2' at revision 2. get()
        always serves the v2 content (the write already happened)."""

        def __init__(self):
            self.loads = 0
            self.bumped = False

        async def load_metadata(self, after_revision):
            self.loads += 1
            if not self.bumped:
                return MetadataLoad(
                    1, MetadataLoadMode.REPLACE,
                    (StorageChange("a", EtagInfo("a", "v1")),),
                )
            return MetadataLoad(
                2, MetadataLoadMode.REPLACE,
                (StorageChange("a", EtagInfo("a", "v2")),),
            )

        async def head_revision(self):
            return 2 if self.bumped else 1

        async def get(self, path):
            # Content already reflects the write (v2), even before metadata sees it.
            return EtagDoc(EtagInfo("a", "v2"), b"fresh")

    class StaleSource:
        """A revision source that is stuck at revision 1 (stale cache): its
        head_revision() returns 1 even after the backend bumped to 2. This
        forces the refresh() short-circuit to serve the stale v1 state."""

        async def head_revision(self):
            return 1

        async def revision_bumped(self, revision):
            pass

    primary = EtagBackend()
    composition = StorageComposition(
        primary, adapter=EtagAdapter(), cache_adapter=EtagAdapter(),
        revision_source=StaleSource(),
    )
    # Prime: metadata loads v1 at revision 1.
    await composition.refresh()
    # Now the backend's content is v2 (a concurrent write happened), but the
    # stale source still claims revision 1. A plain refresh() would short-
    # circuit on (source=1 == _state=1) and keep serving v1 metadata; get()
    # sees content v2 != metadata v1 -> mismatch -> must invalidate so the
    # retry loads v2 metadata (not short-circuited) and serves v2.
    primary.bumped = True
    result = await composition.get("a")
    assert result is not None
    assert result.content == b"fresh"


@pytest.mark.asyncio
async def test_get_many_multi_owner_groups_by_layer():
    # Multi-owner get_many: keys spread across primary + a layer; each is loaded
    # from its owning backend in parallel via _load_by_owner.

    class Info2:
        def __init__(self, path):
            self.path = path

        def __eq__(self, other):
            return isinstance(other, Info2) and self.path == other.path

        def __hash__(self):
            return hash(self.path)

    class Doc2:
        def __init__(self, path, content):
            self.info = Info2(path)
            self.content = content

    class Adapter2:
        def info_key(self, info):
            return info.path

        def value_info(self, value):
            return value.info

        def cache_key(self, key, info):
            return f"k:{key}"

        def cache_content(self, value):
            return value.content

        def from_cache(self, info, content):
            return Doc2(info.path, content)

    class Backend2:
        def __init__(self, docs):
            self.docs = {d.info.path: d for d in docs}

        async def load_metadata(self, after_revision):
            from linktools.ai.storage.revision import MetadataLoad, MetadataLoadMode, StorageChange
            return MetadataLoad(
                1, MetadataLoadMode.REPLACE,
                tuple(StorageChange(d.info.path, d.info) for d in self.docs.values()),
            )

        async def head_revision(self):
            return 1

        async def get(self, path):
            return self.docs.get(path)

    primary = Backend2((Doc2("p", b"primary"),))
    layer = Backend2((Doc2("l", b"layer"),))
    composition = StorageComposition(
        primary,
        layers=(StorageLayer(backend=layer, refresh=LayerRefreshPolicy.REVISIONED),),
        adapter=Adapter2(),
        cache_adapter=Adapter2(),
    )
    result = await composition.get_many(("p", "l"))
    assert set(result) == {"p", "l"}
    assert result["p"].content == b"primary"
    assert result["l"].content == b"layer"


@pytest.mark.asyncio
async def test_reset_changes_observable_state():
    # reset() observable semantics: after reset, the store reflects the
    # new document set (old keys gone, new keys present).

    class Backend(MetadataBackend):
        async def load_metadata(self, after_revision):
            from linktools.ai.storage.revision import MetadataLoad, MetadataLoadMode, StorageChange
            self.loads += 1
            if after_revision == self.revision:
                return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
            changes = tuple(
                StorageChange(d.info.path, d.info)
                for d in sorted(self.docs.values(), key=lambda d: d.info.path)
            )
            return MetadataLoad(self.revision, MetadataLoadMode.REPLACE, changes)

    primary = Backend((_doc("a", b"one"), _doc("b", b"two"),))
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter()
    )
    assert set(await composition.list_info()) >= {Info("a"), Info("b")}
    await composition.reset((_doc("c", b"three"),))
    infos = set(await composition.list_info())
    assert Info("c") in infos
    assert Info("a") not in infos
    assert Info("b") not in infos


@pytest.mark.asyncio
async def test_preload_oversized_blob_not_marked_preloaded():
    # When the cache silently drops an oversized blob (put returns without
    # raising but stores nothing because len(content) > max_bytes), _preload
    # must NOT permanently mark the key preloaded. Without the contains_many
    # re-check, write_cache returns True and the key is recorded in _preloaded,
    # suppressing all future preload attempts while the cache holds nothing.
    from linktools.ai.storage.cache import MemoryContentCache

    primary = MetadataBackend((_doc("a", b"x" * 100),))  # 100-byte content
    # Cache admits only 10 bytes -> put silently drops the 100-byte blob.
    cache = MemoryContentCache(max_bytes=10)
    composition = StorageComposition(
        primary, writer=primary, adapter=Adapter(), cache_adapter=Adapter(),
        cache=cache,
    )
    await composition.list_info(preload=True)
    # The key must NOT be marked preloaded (the cache does not hold it).
    assert "a" not in composition._preloaded, (
        "oversized-blob key was permanently marked preloaded; the contains_many "
        "re-check is missing or broken"
    )
    # And a second preload re-attempts (not suppressed).
    await composition.list_info(preload=True)
    assert "a" not in composition._preloaded


@pytest.mark.asyncio
async def test_get_many_invalidates_owner_on_content_race():
    # When get_many loads content that races the held metadata (value_info
    # mismatch), the key is omitted from the result AND the owning view is
    # invalidated so the next read loads fresh metadata. Without the invalidate,
    # the stale metadata persists and the key stays unreachable.

    @dataclass(frozen=True)
    class EtagInfo:
        path: str
        etag: str

    @dataclass(frozen=True)
    class EtagDoc:
        info: EtagInfo
        content: bytes

    class EtagAdapter:
        def info_key(self, info):
            return info.path

        def value_info(self, value):
            return value.info

        def cache_key(self, key, info):
            return f"k:{key}:{info.etag}"

        def cache_content(self, value):
            return value.content

        def from_cache(self, info, content):
            return EtagDoc(info, content)

    class RacingBackend:
        """Metadata reports etag 'stale'; get() serves content with etag 'fresh'
        to model a read that raced a concurrent write."""

        def __init__(self):
            self.loads = 0

        async def load_metadata(self, after_revision):
            self.loads += 1
            return MetadataLoad(
                1, MetadataLoadMode.REPLACE,
                (StorageChange("a", EtagInfo("a", "stale")),),
            )

        async def head_revision(self):
            return 1

        async def get(self, path):
            return EtagDoc(EtagInfo("a", "fresh"), b"fresh")

    primary = RacingBackend()
    composition = StorageComposition(
        primary, adapter=EtagAdapter(), cache_adapter=EtagAdapter()
    )
    # Prime metadata (stale etag held).
    await composition.refresh()
    # get_many sees fresh content vs stale metadata -> omit + invalidate owner.
    result = await composition.get_many(("a",))
    assert "a" not in result, "raced key should be omitted from get_many result"
    # The owner view was invalidated: its held _state is None now.
    assert composition.primary_view._state is None, (
        "owner view was not invalidated after a content/metadata race in get_many"
    )
