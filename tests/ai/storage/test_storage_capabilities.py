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
