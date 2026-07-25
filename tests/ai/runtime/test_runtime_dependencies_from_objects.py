#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeDependencies.from_objects over a real ObjectStore (contract)."""

import pytest

from linktools.ai.runtime import RuntimeDependencies
from linktools.ai.runtime.dependencies import ProviderPrefixes
from linktools.ai.storage.backends.memory.object import MemoryObjectStore
from linktools.ai.storage.object.models import StorageKey, WriteOptions


async def _store() -> MemoryObjectStore:
    store = MemoryObjectStore()
    files = {
        "/specs/agents/writer.md": "---\nname: writer\nmodel:\n  primary: gpt-4o\n---\nhi\n",
        "/specs/skills/sql.md": "---\nname: sql\n---\nx\n",
        "/specs/mcp/search.yaml": "name: search\ntransport: stdio\ncommand: python\n",
    }
    for path, text in files.items():
        await store.put(
            StorageKey(path),
            text.encode("utf-8"),
            options=WriteOptions(content_type="text/plain"),
        )
    return store


@pytest.mark.asyncio
async def test_from_objects_loads_agent_skill_mcp():
    store = await _store()
    bundle = RuntimeDependencies.from_objects(store, prefixes=ProviderPrefixes())
    assert bundle.agents is not None
    assert bundle.skills is not None
    assert bundle.mcp_servers is not None
    assert "writer" in await bundle.agents.list_ids()
    assert "sql" in await bundle.skills.list_ids()
    assert "search" in await bundle.mcp_servers.list_ids()
