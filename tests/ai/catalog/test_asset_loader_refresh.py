#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Object-backed registry refresh: the registry cache must reflect object
changes via SpecLoader.from_objects' revision. The MCP enabled_tools refresh
case is the key security regression -- a server reconfigured from
enabled_tools=[read] to enabled_tools=[] must take effect on the next read."""

import pytest

from linktools.ai.mcp.catalog import MCPCatalog
from linktools.ai.catalog.parsing import SpecLoader
from linktools.ai.storage.backends.memory.object import MemoryObjectStore
from linktools.ai.storage.object.models import StorageKey, WriteOptions


def _yaml(enabled_tools: "list[str] | None") -> str:
    base = "transport: stdio\ncommand: ['python', '-m', 'r']\n"
    if enabled_tools is None:
        return base
    import yaml

    return base + yaml.safe_dump({"enabled_tools": enabled_tools}, sort_keys=False)


async def _store_with(path_text: "dict[str, str]") -> MemoryObjectStore:
    store = MemoryObjectStore()
    for path, text in path_text.items():
        await store.put(
            StorageKey(path),
            text.encode("utf-8"),
            options=WriteOptions(content_type="application/yaml"),
        )
    return store


@pytest.mark.asyncio
async def test_mcp_registry_refreshes_enabled_tools_after_update():
    """A server reconfigured from enabled_tools=[read] to enabled_tools=[]
    must take effect on the next registry read -- the revision change drops the
    cached spec and re-parses the object."""
    store = await _store_with({"/specs/mcp/risk.yaml": _yaml(["read"])})
    registry = MCPCatalog.from_specloader(SpecLoader.from_objects(store, prefix="specs/mcp"))

    spec = await registry.get("risk")
    assert spec.enabled_tools == ("read",)

    await store.put(
        StorageKey("/specs/mcp/risk.yaml"),
        _yaml([]).encode("utf-8"),
        options=WriteOptions(content_type="application/yaml"),
    )
    refreshed = await registry.get("risk")
    assert refreshed.enabled_tools == (), (
        "enabled_tools=[] must take effect after the object is updated"
    )


@pytest.mark.asyncio
async def test_mcp_registry_sees_new_and_deleted_servers():
    """list_ids refreshes as servers are added/removed: the revision-based cache
    invalidation drops the id listing alongside the per-id cache."""
    store = await _store_with({"/specs/mcp/risk.yaml": _yaml(None)})
    registry = MCPCatalog.from_specloader(SpecLoader.from_objects(store, prefix="specs/mcp"))

    assert await registry.list_ids() == ("risk",)

    await store.put(
        StorageKey("/specs/mcp/audit.yaml"),
        _yaml(None).encode("utf-8"),
        options=WriteOptions(content_type="application/yaml"),
    )
    assert await registry.list_ids() == ("audit", "risk")

    await store.delete(StorageKey("/specs/mcp/risk.yaml"))
    assert await registry.list_ids() == ("audit",)
