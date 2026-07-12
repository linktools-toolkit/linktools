#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpecLoader.from_resources against the real ResourceStore API (get/propfind,
no .list/.revision). Prefix sandbox + ResourcePath usage."""

import pytest

from linktools.ai.registry.parser import SpecLoader
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.resource.store import ResourceStore


async def _store_with(prefix_files: "dict[str, str]") -> ResourceStore:
    backend = MemoryResourceBackend()
    store = ResourceStore(primary=backend)
    for path, text in prefix_files.items():
        await backend.raw_put(
            ResourcePath(path),
            text.encode("utf-8"),
            content_type="text/markdown",
            metadata={},
        )
    return store


@pytest.mark.asyncio
async def test_from_resources_reads_via_resourcepath():
    store = await _store_with(
        {
            "/specs/agents/writer.md": "---\nname: writer\n---\nbody\n",
            "/specs/agents/minimal.md": "---\nname: minimal\n---\nbody\n",
        }
    )
    loader = SpecLoader.from_resources(store, prefix="specs/agents")
    text = await loader.read("writer.md")
    assert "writer" in text


@pytest.mark.asyncio
async def test_from_resources_list_ids_uses_propfind():
    store = await _store_with(
        {
            "/specs/skills/sql.md": "x",
            "/specs/skills/audit.md": "y",
            "/specs/skills/sub/ignored.md": "z",  # depth.ONE must not recurse
        }
    )
    loader = SpecLoader.from_resources(store, prefix="specs/skills")
    ids = await loader.list_ids(".md")
    assert set(ids) == {"audit", "sql"}


@pytest.mark.asyncio
async def test_from_resources_prefix_leading_slash_tolerated():
    store = await _store_with({"/specs/agents/a.md": "x"})
    loader = SpecLoader.from_resources(store, prefix="/specs/agents")
    assert await loader.list_ids(".md") == ("a",)


@pytest.mark.asyncio
async def test_from_resources_revision_is_constant_zero():
    store = await _store_with({"/specs/agents/a.md": "x"})
    loader = SpecLoader.from_resources(store, prefix="specs/agents")
    assert await loader.revision() == 0


@pytest.mark.asyncio
async def test_from_resources_rejects_parent_traversal():
    store = await _store_with({"/specs/agents/a.md": "x"})
    loader = SpecLoader.from_resources(store, prefix="specs/agents")
    with pytest.raises(Exception):
        await loader.read("../etc/passwd")
