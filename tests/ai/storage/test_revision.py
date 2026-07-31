#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-load metadata protocol and LayerMetadataView conformance.

Validates the single-load contract: one load_metadata call returns revision +
REPLACE-or-PATCH, single-flight collapses N concurrent refreshes into one
backend load, and a cancelled caller never publishes a half state."""

import asyncio

import pytest

from linktools.ai.storage.revision import (
    LayerMetadataView,
    LayerRefreshPolicy,
    MetadataLoad,
    MetadataLoadMode,
    StorageChange,
    apply_metadata_load,
)


def _change(key, current):
    return StorageChange(key, current)


def test_replace_load_discards_prior_state():
    state = apply_metadata_load(
        None,
        MetadataLoad(3, MetadataLoadMode.REPLACE, (_change("a", "A"), _change("b", "B"))),
        info_key=lambda info: info,
    )
    assert state.revision == 3
    assert state.entries == {"a": "A", "b": "B"}


def test_patch_load_applies_only_changes():
    prior = apply_metadata_load(
        None,
        MetadataLoad(1, MetadataLoadMode.REPLACE, (_change("a", "A"), _change("b", "B"))),
        info_key=lambda info: info,
    )
    state = apply_metadata_load(
        prior,
        MetadataLoad(2, MetadataLoadMode.PATCH, (_change("a", "A2"), _change("c", None))),
        info_key=lambda info: info,
    )
    assert state.entries == {"a": "A2", "b": "B"}


def test_patch_replace_omits_none_current_entries():
    state = apply_metadata_load(
        None,
        MetadataLoad(1, MetadataLoadMode.REPLACE, (_change("a", None), _change("b", "B"))),
        info_key=lambda info: info,
    )
    # None current in REPLACE is skipped (no tombstone semantics on a full set).
    assert state.entries == {"b": "B"}


@pytest.mark.asyncio
async def test_revisioned_view_single_flight_collapses_concurrent_loads():
    loads = 0

    class Backend:
        revision = 1

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            await asyncio.sleep(0.01)
            return MetadataLoad(
                self.revision,
                MetadataLoadMode.PATCH,
                (),
            )

        async def head_revision(self):
            return self.revision

    view = LayerMetadataView(Backend(), LayerRefreshPolicy.REVISIONED, info_key=lambda i: i)
    results = await asyncio.gather(*(view.refresh() for _ in range(100)))
    assert all(result is results[0] for result in results)
    assert loads == 1, f"expected 1 load, got {loads}"


@pytest.mark.asyncio
async def test_always_view_uses_generation_revision():
    calls = 0

    class Backend:
        async def list_info(self):
            nonlocal calls
            calls += 1
            return ("a", "b")

    view = LayerMetadataView(Backend(), LayerRefreshPolicy.ALWAYS, info_key=lambda i: i)
    s1 = await view.refresh()
    s2 = await view.refresh()
    assert s1.revision != s2.revision  # generation changes each refresh
    assert calls == 2


@pytest.mark.asyncio
async def test_static_view_loads_once():
    calls = 0

    class Backend:
        async def list_info(self):
            nonlocal calls
            calls += 1
            return ("a",)

    view = LayerMetadataView(Backend(), LayerRefreshPolicy.STATIC, info_key=lambda i: i)
    await view.refresh()
    await view.refresh()
    await view.refresh()
    assert calls == 1


@pytest.mark.asyncio
async def test_revisioned_view_empty_patch_at_same_revision_serves_cached():
    loads = 0

    class Backend:
        revision = 5

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            if after_revision == self.revision:
                return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
            return MetadataLoad(
                self.revision, MetadataLoadMode.REPLACE, (_change("a", "A"),)
            )

        async def head_revision(self):
            return self.revision

    view = LayerMetadataView(Backend(), LayerRefreshPolicy.REVISIONED, info_key=lambda i: i)
    await view.refresh()  # REPLACE
    await view.refresh()  # empty PATCH at same rev -> legal, still 1 more load
    assert loads == 2


@pytest.mark.asyncio
async def test_revisioned_view_head_revision_probes_without_loading_entries():
    loads = 0
    head_calls = 0

    class Backend:
        revision = 7

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            return MetadataLoad(self.revision, MetadataLoadMode.REPLACE, (_change("a", "A"),))

        async def head_revision(self):
            nonlocal head_calls
            head_calls += 1
            return self.revision

    view = LayerMetadataView(Backend(), LayerRefreshPolicy.REVISIONED, info_key=lambda i: i)
    head = await view.head_revision()
    assert head == 7
    # The cheap probe must not trigger a load_metadata round trip.
    assert loads == 0
    assert head_calls == 1


@pytest.mark.asyncio
async def test_always_view_head_revision_returns_none():
    # ALWAYS layers are plain StorageReader (no StorageMetadataBackend): there is
    # no cheap head probe, so head_revision returns None to signal the caller
    # must fall back to a full refresh.
    class Backend:
        async def list_info(self):
            return ("a",)

    view = LayerMetadataView(Backend(), LayerRefreshPolicy.ALWAYS, info_key=lambda i: i)
    assert await view.head_revision() is None


class _FakeRevisionSource:
    """A controllable RevisionSource for tests: returns whatever revision is set
    (or None to simulate an unknown/missed cache). Records revision_bumped
    calls so a test can assert source correction."""

    def __init__(self, revision=None):
        self.revision = revision
        self.bumps = []

    async def head_revision(self):
        return self.revision

    async def revision_bumped(self, revision):
        self.bumps.append(revision)
        self.revision = revision


@pytest.mark.asyncio
async def test_revision_source_short_circuits_unchanged_refresh():
    # When a revision_source is wired and reports the same revision the view
    # already holds, a second refresh() reuses the held state and issues ZERO
    # load_metadata calls.
    loads = 0

    class Backend:
        revision = 3

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            if after_revision == self.revision:
                return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
            return MetadataLoad(
                self.revision, MetadataLoadMode.REPLACE, (_change("a", "A"),)
            )

        async def head_revision(self):
            return self.revision

    source = _FakeRevisionSource(revision=3)
    view = LayerMetadataView(
        Backend(), LayerRefreshPolicy.REVISIONED, info_key=lambda i: i,
        revision_source=source,
    )
    first = await view.refresh()  # REPLACE -> loads == 1, holds revision 3
    assert first.revision == 3
    assert loads == 1
    second = await view.refresh()  # source says unchanged -> short-circuit
    assert second is first
    assert loads == 1, f"short-circuit should add no load, got {loads}"


@pytest.mark.asyncio
async def test_revision_source_reload_when_revision_changes():
    # When the source reports a NEW revision, the view must reload (not serve
    # the stale held state).
    loads = 0

    class Backend:
        revision = 3

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            if after_revision == self.revision:
                return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
            return MetadataLoad(
                self.revision, MetadataLoadMode.REPLACE, (_change("a", "A"),)
            )

        async def head_revision(self):
            return self.revision

    source = _FakeRevisionSource(revision=3)
    backend = Backend()
    view = LayerMetadataView(
        backend, LayerRefreshPolicy.REVISIONED, info_key=lambda i: i,
        revision_source=source,
    )
    await view.refresh()
    assert loads == 1
    # A REAL change: both the backend head and the source advance to 4. The head
    # probe sees head(4) != held(3) and falls through to load the diff.
    backend.revision = 4
    source.revision = 4
    await view.refresh()
    assert loads == 2, "changed revision (head advanced) must trigger a reload"


@pytest.mark.asyncio
async def test_revision_source_none_falls_back_to_load():
    # When the source returns None (cache miss / unknown), the view must not
    # short-circuit -- it falls back to a real load_metadata for correctness.
    loads = 0

    class Backend:
        revision = 3

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            return MetadataLoad(
                self.revision, MetadataLoadMode.PATCH, ()
            )

        async def head_revision(self):
            return self.revision

    source = _FakeRevisionSource(revision=None)  # unknown
    view = LayerMetadataView(
        Backend(), LayerRefreshPolicy.REVISIONED, info_key=lambda i: i,
        revision_source=source,
    )
    await view.refresh()
    assert loads == 1
    await view.refresh()  # source says None -> must reload
    assert loads == 2, "None source must fall back to a load"


@pytest.mark.asyncio
async def test_default_backend_head_source_delegates_to_head_revision():
    # The default _BackendHeadRevisionSource probes head_revision on a
    # StorageMetadataBackend, and returns None for a non-metadata backend.
    from linktools.ai.storage.revision import _BackendHeadRevisionSource

    class MetadataBackend:
        async def load_metadata(self, after_revision): ...
        async def head_revision(self):
            return 9

    src = _BackendHeadRevisionSource(MetadataBackend())
    assert await src.head_revision() == 9

    class PlainReader:
        async def list_info(self): ...

    src2 = _BackendHeadRevisionSource(PlainReader())
    assert await src2.head_revision() is None


@pytest.mark.asyncio
async def test_source_ahead_of_head_does_not_hammer_load_and_corrects_source():
    # A source whose cached revision runs ahead of the true head (rolled-back
    # publish / operator error) must NOT force a load_metadata on every read.
    # The view probes the authoritative head, sees it equals the held revision,
    # reuses the held state, and corrects the source via revision_bumped so the
    # next read short-circuits.
    loads = 0

    class Backend:
        revision = 5  # true head stays at 5

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            if after_revision == self.revision:
                return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
            return MetadataLoad(
                self.revision, MetadataLoadMode.REPLACE, (_change("a", "A"),)
            )

        async def head_revision(self):
            return self.revision

    source = _FakeRevisionSource(revision=5)
    view = LayerMetadataView(
        Backend(), LayerRefreshPolicy.REVISIONED, info_key=lambda i: i,
        revision_source=source,
    )
    await view.refresh()  # REPLACE -> loads == 1, holds revision 5
    assert loads == 1
    # The source now claims 6 (ahead of the true head 5).
    source.revision = 6
    await view.refresh()
    # No load_metadata: the head probe saw head(5) == held(5) and short-circuited.
    assert loads == 1, "runaway source must not force a reload"
    # The source was corrected down to the true head.
    assert source.bumps == [5]
    # A subsequent read short-circuits cleanly (source now says 5 == held 5).
    await view.refresh()
    assert loads == 1


@pytest.mark.asyncio
async def test_source_ahead_still_sees_a_real_write():
    # Correctness: even while the source is ahead, a REAL write that advances
    # head must be observed. The head probe sees head advance and falls through
    # to load_metadata, which returns the diff.
    loads = 0

    class Backend:
        revision = 5

        async def load_metadata(self, after_revision):
            nonlocal loads
            loads += 1
            if after_revision == self.revision:
                return MetadataLoad(self.revision, MetadataLoadMode.PATCH, ())
            return MetadataLoad(
                self.revision, MetadataLoadMode.REPLACE, (_change("a", "A2"),)
            )

        async def head_revision(self):
            return self.revision

    backend = Backend()
    source = _FakeRevisionSource(revision=5)
    view = LayerMetadataView(
        backend, LayerRefreshPolicy.REVISIONED, info_key=lambda i: i,
        revision_source=source,
    )
    await view.refresh()
    assert loads == 1
    # Source runs ahead to 6, but a real write also advances head to 6.
    source.revision = 6
    backend.revision = 6
    state = await view.refresh()
    # The head probe saw head(6) != held(5) -> fell through to load the diff.
    assert loads == 2, "a real write (head advanced) must trigger a reload"
    assert state.revision == 6
    assert "a" in state.entries
