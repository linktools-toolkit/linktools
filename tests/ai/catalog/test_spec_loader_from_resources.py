#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpecLoader.from_assets against the real AssetStore API (get/list,
no .list/.revision). Prefix sandbox + AssetPath usage."""

import pytest

from linktools.ai.catalog.parsing import SpecLoader
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore


async def _store_with(prefix_files: "dict[str, str]") -> AssetStore:
    backend = MemoryAssetBackend()
    store = AssetStore(primary=backend)
    for path, text in prefix_files.items():
        await backend.raw_put(
            AssetPath(path),
            text.encode("utf-8"),
            content_type="text/markdown",
            metadata={},
        )
    return store


@pytest.mark.asyncio
async def test_from_assets_reads_via_resourcepath():
    store = await _store_with(
        {
            "/specs/agents/writer.md": "---\nname: writer\n---\nbody\n",
            "/specs/agents/minimal.md": "---\nname: minimal\n---\nbody\n",
        }
    )
    loader = SpecLoader.from_assets(store, prefix="specs/agents")
    text = await loader.read("writer.md")
    assert "writer" in text


@pytest.mark.asyncio
async def test_from_assets_list_ids_uses_list():
    store = await _store_with(
        {
            "/specs/skills/sql.md": "x",
            "/specs/skills/audit.md": "y",
            "/specs/skills/sub/ignored.md": "z",  # depth.ONE must not recurse
        }
    )
    loader = SpecLoader.from_assets(store, prefix="specs/skills")
    ids = await loader.list_ids(".md")
    assert set(ids) == {"audit", "sql"}


@pytest.mark.asyncio
async def test_from_assets_prefix_leading_slash_tolerated():
    store = await _store_with({"/specs/agents/a.md": "x"})
    loader = SpecLoader.from_assets(store, prefix="/specs/agents")
    assert await loader.list_ids(".md") == ("a",)


@pytest.mark.asyncio
async def test_from_assets_revision_reflects_modify_add_delete():
    """revision() is a stable hash over live asset metadata, so the registry
    cache refreshes after any change -- not pinned to a constant 0."""
    store = await _store_with({"/specs/agents/a.md": "v1"})
    loader = SpecLoader.from_assets(store, prefix="specs/agents")
    rev_initial = await loader.revision()
    assert rev_initial != 0, "revision must not be a constant 0"

    # Modify: overwrite a.md with new content -> etag/size/version change.
    await store.put(
        AssetPath("/specs/agents/a.md"),
        b"v2",
        options=WriteOptions(content_type="text/markdown"),
    )
    rev_after_modify = await loader.revision()
    assert rev_after_modify != rev_initial, "revision must change on modify"

    # Add: a new asset -> revision changes again.
    await store.put(
        AssetPath("/specs/agents/b.md"),
        b"x",
        options=WriteOptions(content_type="text/markdown"),
    )
    rev_after_add = await loader.revision()
    assert rev_after_add != rev_after_modify, "revision must change on add"

    # Delete: b.md removed -> revision changes again, and list_ids drops it.
    await store.delete(AssetPath("/specs/agents/b.md"))
    rev_after_delete = await loader.revision()
    assert rev_after_delete != rev_after_add, "revision must change on delete"


@pytest.mark.asyncio
async def test_from_assets_revision_stable_across_unchanged_reads():
    """An unchanged asset set yields the same revision (no spurious cache
    invalidation), and list_ids tracks adds/deletes."""
    store = await _store_with({"/specs/agents/a.md": "x"})
    loader = SpecLoader.from_assets(store, prefix="specs/agents")
    first = await loader.revision()
    second = await loader.revision()
    assert first == second, "unchanged set must keep a stable revision"
    assert await loader.list_ids(".md") == ("a",)

    await store.put(
        AssetPath("/specs/agents/b.md"),
        b"y",
        options=WriteOptions(content_type="text/markdown"),
    )
    assert await loader.list_ids(".md") == ("a", "b"), "list_ids must see new id"


@pytest.mark.asyncio
async def test_from_assets_rejects_parent_traversal():
    store = await _store_with({"/specs/agents/a.md": "x"})
    loader = SpecLoader.from_assets(store, prefix="specs/agents")
    with pytest.raises(Exception):
        await loader.read("../etc/passwd")
