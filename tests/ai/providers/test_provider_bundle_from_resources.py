#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProviderBundle.from_resources over a real ResourceStore (spec §4.6)."""

import pytest

from linktools.ai.providers import ProviderBundle, ProviderPrefixes
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.resource.store import ResourceStore


async def _store():
    backend = MemoryResourceBackend()
    files = {
        "/specs/agents/writer.md": "---\nname: writer\nmodel:\n  primary: gpt-4o\n---\nhi\n",
        "/specs/skills/sql.md": "---\nname: sql\n---\nx\n",
        "/specs/mcp/search.yaml": "name: search\ntransport: stdio\ncommand: python\n",
    }
    for path, text in files.items():
        await backend.raw_put(ResourcePath(path), text.encode("utf-8"),
                              content_type="text/plain", metadata={})
    return ResourceStore(primary=backend)


@pytest.mark.asyncio
async def test_from_resources_loads_agent_skill_mcp():
    store = await _store()
    bundle = ProviderBundle.from_resources(store, prefixes=ProviderPrefixes())
    assert bundle.agents is not None
    assert bundle.skills is not None
    assert bundle.mcp_servers is not None
    assert "writer" in await bundle.agents.list_ids()
    assert "sql" in await bundle.skills.list_ids()
    assert "search" in await bundle.mcp_servers.list_ids()


@pytest.mark.asyncio
async def test_from_resources_does_not_call_store_list_or_revision():
    store = await _store()
    # ResourceStore has no .list/.revision attributes at all -- from_resources
    # must not reach for them (would AttributeError if it did).
    assert not hasattr(store, "list")
    bundle = ProviderBundle.from_resources(store, prefixes=ProviderPrefixes())
    # exercising list_ids proves the propfind path works without .list/.revision
    await bundle.agents.list_ids()
