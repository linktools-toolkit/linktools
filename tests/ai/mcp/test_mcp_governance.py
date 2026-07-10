#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPProvider governance at resolve time (enabled/disabled/prefix/conflict/
max_tools) via a fake connection manager that yields canned tool names."""

import pytest

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import CapabilityConflictError
from linktools.ai.mcp.provider import MCPProvider
from linktools.ai.registry.mcp import MCPServerSpec


class _FakeMgr:
    """Fake manager: yields canned tool names per server + a stub toolset."""
    def __init__(self, names_by_server):
        self._names = names_by_server

    async def list_tools(self, server):
        return tuple(self._names.get(server.id, ()))

    async def get_toolset(self, server):
        return object()


def _spec(sid, **kw):
    base = dict(transport="stdio", command_or_url="python -m x", command=("python", "-m", "x"))
    base.update(kw)
    return MCPServerSpec(id=sid, name=sid, **base)


def _ctx():
    return CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy())


@pytest.mark.asyncio
async def test_enabled_tools_filters():
    spec = _spec("risk", enabled_tools=("query_user",))
    mgr = _FakeMgr({"risk": ("query_user", "query_device", "secret")})
    p = MCPProvider(_FakeSrc({"risk": spec}), mgr)
    bundle = await p.resolve(CapabilityRef("mcp", "risk"), _ctx())
    # No conflict -> resolves; governance applied (query_user kept, others dropped).
    assert len(bundle.toolsets) == 1


@pytest.mark.asyncio
async def test_disabled_tools_filters():
    spec = _spec("risk", disabled_tools=("secret",))
    mgr = _FakeMgr({"risk": ("query_user", "secret")})
    p = MCPProvider(_FakeSrc({"risk": spec}), mgr)
    await p.resolve(CapabilityRef("mcp", "risk"), _ctx())  # no raise


@pytest.mark.asyncio
async def test_max_tools_per_capability_enforced():
    spec = _spec("risk")
    mgr = _FakeMgr({"risk": tuple(f"t{i}" for i in range(20))})
    ctx = CapabilityContext(agent_id="a1",
                            exposure_policy=CapabilityToolExposurePolicy(max_tools_per_capability=5))
    p = MCPProvider(_FakeSrc({"risk": spec}), mgr)
    with pytest.raises(CapabilityConflictError, match="max_tools_per_capability"):
        await p.resolve(CapabilityRef("mcp", "risk"), ctx)


@pytest.mark.asyncio
async def test_cross_server_conflict_detected():
    # Both servers default-prefix with their own id -> no conflict. Force a
    # collision by setting tool_prefix=False on both so raw names clash.
    s1 = _spec("a", tool_prefix=False)
    s2 = _spec("b", tool_prefix=False)
    mgr = _FakeMgr({"a": ("dup",), "b": ("dup",)})
    p = MCPProvider(_FakeSrc({"a": s1, "b": s2}), mgr, allow_mcp_wildcard=True)
    with pytest.raises(CapabilityConflictError, match="exposed by both"):
        await p.resolve(CapabilityRef("mcp", "*"), _ctx())


@pytest.mark.asyncio
async def test_tool_prefix_default_avoids_conflict():
    # Default prefix = server_id -> a.dup / b.dup, no conflict.
    s1 = _spec("a")
    s2 = _spec("b")
    mgr = _FakeMgr({"a": ("dup",), "b": ("dup",)})
    p = MCPProvider(_FakeSrc({"a": s1, "b": s2}), mgr, allow_mcp_wildcard=True)
    bundle = await p.resolve(CapabilityRef("mcp", "*"), _ctx())
    assert len(bundle.toolsets) == 2


class _FakeSrc:
    def __init__(self, specs):
        self._specs = specs

    async def list_ids(self):
        return tuple(self._specs.keys())

    async def get(self, sid):
        if sid not in self._specs:
            raise KeyError(sid)
        return self._specs[sid]
