#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resource-backed registry refresh: the registry cache must reflect resource
changes via SpecLoader.from_resources' revision. The MCP enabled_tools refresh
case is the key security regression -- a server reconfigured from
enabled_tools=[read] to enabled_tools=[] must take effect on the next read."""

import pytest

from linktools.ai.registry.mcp import MCPRegistry
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.storage.resource.memory import MemoryResourceBackend
from linktools.ai.storage.resource.models import WriteOptions
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.resource.store import ResourceStore


def _yaml(enabled_tools: "list[str] | None") -> str:
    base = "transport: stdio\ncommand: ['python', '-m', 'r']\n"
    if enabled_tools is None:
        return base
    import yaml

    return base + yaml.safe_dump({"enabled_tools": enabled_tools}, sort_keys=False)


async def _store_with(path_text: "dict[str, str]") -> ResourceStore:
    backend = MemoryResourceBackend()
    store = ResourceStore(primary=backend)
    for path, text in path_text.items():
        await store.put(
            ResourcePath(path),
            text.encode("utf-8"),
            options=WriteOptions(content_type="application/yaml"),
        )
    return store


@pytest.mark.asyncio
async def test_mcp_registry_refreshes_enabled_tools_after_update():
    """A server reconfigured from enabled_tools=[read] to enabled_tools=[]
    must take effect on the next registry read -- the revision change drops the
    cached spec and re-parses the resource."""
    store = await _store_with({"/specs/mcp/risk.yaml": _yaml(["read"])})
    registry = MCPRegistry(SpecLoader.from_resources(store, prefix="specs/mcp"))

    spec = await registry.get("risk")
    assert spec.enabled_tools == ("read",)

    await store.put(
        ResourcePath("/specs/mcp/risk.yaml"),
        _yaml([]).encode("utf-8"),
        options=WriteOptions(content_type="application/yaml"),
    )
    refreshed = await registry.get("risk")
    assert refreshed.enabled_tools == (), (
        "enabled_tools=[] must take effect after the resource is updated"
    )


@pytest.mark.asyncio
async def test_mcp_registry_sees_new_and_deleted_servers():
    """list_ids refreshes as servers are added/removed: the revision-based cache
    invalidation drops the id listing alongside the per-id cache."""
    store = await _store_with({"/specs/mcp/risk.yaml": _yaml(None)})
    registry = MCPRegistry(SpecLoader.from_resources(store, prefix="specs/mcp"))

    assert await registry.list_ids() == ("risk",)

    await store.put(
        ResourcePath("/specs/mcp/audit.yaml"),
        _yaml(None).encode("utf-8"),
        options=WriteOptions(content_type="application/yaml"),
    )
    assert await registry.list_ids() == ("audit", "risk")

    await store.delete(ResourcePath("/specs/mcp/risk.yaml"))
    assert await registry.list_ids() == ("audit",)
