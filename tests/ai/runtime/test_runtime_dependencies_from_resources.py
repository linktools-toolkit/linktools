#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeDependencies.from_resources over a real AssetStore (contract)."""

import pytest

from linktools.ai.runtime import RuntimeDependencies
from linktools.ai._runtime.dependencies import ProviderPrefixes
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore


async def _store():
    backend = MemoryAssetBackend()
    files = {
        "/specs/agents/writer.md": "---\nname: writer\nmodel:\n  primary: gpt-4o\n---\nhi\n",
        "/specs/skills/sql.md": "---\nname: sql\n---\nx\n",
        "/specs/mcp/search.yaml": "name: search\ntransport: stdio\ncommand: python\n",
    }
    for path, text in files.items():
        await backend.raw_put(
            AssetPath(path),
            text.encode("utf-8"),
            content_type="text/plain",
            metadata={},
        )
    return AssetStore(primary=backend)


@pytest.mark.asyncio
async def test_from_resources_loads_agent_skill_mcp():
    store = await _store()
    bundle = RuntimeDependencies.from_resources(store, prefixes=ProviderPrefixes())
    assert bundle.agents is not None
    assert bundle.skills is not None
    assert bundle.mcp_servers is not None
    assert "writer" in await bundle.agents.list_ids()
    assert "sql" in await bundle.skills.list_ids()
    assert "search" in await bundle.mcp_servers.list_ids()


@pytest.mark.asyncio
async def test_from_resources_does_not_call_store_list_or_revision():
    store = await _store()
    # AssetStore has no .list/.revision attributes at all -- from_resources
    # must not reach for them (would AttributeError if it did).
    assert not hasattr(store, "list")
    bundle = RuntimeDependencies.from_resources(store, prefixes=ProviderPrefixes())
    # exercising list_ids proves the propfind path works without .list/.revision
    await bundle.agents.list_ids()
