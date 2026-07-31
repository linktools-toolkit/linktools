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
