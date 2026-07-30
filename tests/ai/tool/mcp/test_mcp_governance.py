#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPToolProvider governance at resolve time (enabled/disabled/prefix/conflict/
max_tools) via a fake connection manager that yields canned tool names."""

import pytest

from linktools.ai.agent.tool.exposure import ToolExposurePolicy
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.agent.assembly.models import AgentFeatureRef
from linktools.ai.errors import AgentFeatureConflictError
from linktools.ai.agent.mcp.models import MCPConnectionRef
from linktools.ai.agent.mcp.tool_provider import MCPDiscoveryResult, MCPToolProvider, MCPToolInfo
from linktools.ai.agent.mcp.models import MCPRuntimePolicy
from linktools.ai.agent.mcp.spec import MCPServerSpec


class _FakeMgr:
    """Fake manager: yields canned tool names per server via list_tools_result."""

    def __init__(self, names_by_server):
        self._names = names_by_server
        self._ref = MCPConnectionRef("fake", "fp")

    async def list_tools_result(self, server):
        names = tuple(self._names.get(server.id, ()))
        return MCPDiscoveryResult(
            tools=tuple(MCPToolInfo(name=n) for n in names),
            verified=True,
            connection_ref=self._ref,
        )

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        return {}


def _spec(sid, **kw):
    base = dict(transport="stdio", command=("python", "-m", "x"))
    base.update(kw)
    return MCPServerSpec(id=sid, name=sid, **base)


def _ctx():
    return AgentFeatureContext(
        agent_id="a1",
        execution_id="e1",
        root_execution_id="e1",
        parent_execution_id=None,
        session_id="s1",
        tenant_id="t1",
        user_id="u1",
        workspace=None,
        sandbox=None,
    )


@pytest.mark.asyncio
async def test_enabled_tools_filters():
    spec = _spec("risk", enabled_tools=("query_user",))
    mgr = _FakeMgr({"risk": ("query_user", "query_device", "secret")})
    p = MCPToolProvider(_FakeSrc({"risk": spec}), mgr)
    bundle = await p.resolve(AgentFeatureRef("mcp", "risk"), _ctx())
    # No conflict -> resolves; governance applied (query_user kept, others dropped).
    assert [tool.descriptor.name for tool in bundle.tools] == ["risk.query_user"]


@pytest.mark.asyncio
async def test_disabled_tools_filters():
    spec = _spec("risk", disabled_tools=("secret",))
    mgr = _FakeMgr({"risk": ("query_user", "secret")})
    p = MCPToolProvider(_FakeSrc({"risk": spec}), mgr)
    await p.resolve(AgentFeatureRef("mcp", "risk"), _ctx())  # no raise


@pytest.mark.asyncio
async def test_max_tools_per_capability_enforced():
    spec = _spec("risk")
    mgr = _FakeMgr({"risk": tuple(f"t{i}" for i in range(20))})
    p = MCPToolProvider(
        _FakeSrc({"risk": spec}),
        mgr,
        policy=MCPRuntimePolicy(max_tools_per_server=5),
    )
    with pytest.raises(AgentFeatureConflictError, match="max_tools_per_server"):
        await p.resolve(AgentFeatureRef("mcp", "risk"), _ctx())


@pytest.mark.asyncio
async def test_cross_server_conflict_detected():
    # Both servers default-prefix with their own id -> no conflict. Force a
    # collision by setting tool_prefix=False on both so raw names clash.
    s1 = _spec("a", tool_prefix=False)
    s2 = _spec("b", tool_prefix=False)
    mgr = _FakeMgr({"a": ("dup",), "b": ("dup",)})
    p = MCPToolProvider(
        _FakeSrc({"a": s1, "b": s2}),
        mgr,
        policy=MCPRuntimePolicy(allow_wildcard=True),
    )
    with pytest.raises(AgentFeatureConflictError, match="exposed by both"):
        await p.resolve(AgentFeatureRef("mcp", "*"), _ctx())


@pytest.mark.asyncio
async def test_tool_prefix_default_avoids_conflict():
    # Default prefix = server_id -> a.dup / b.dup, no conflict.
    s1 = _spec("a")
    s2 = _spec("b")
    mgr = _FakeMgr({"a": ("dup",), "b": ("dup",)})
    p = MCPToolProvider(
        _FakeSrc({"a": s1, "b": s2}),
        mgr,
        policy=MCPRuntimePolicy(allow_wildcard=True),
    )
    bundle = await p.resolve(AgentFeatureRef("mcp", "*"), _ctx())
    assert len(bundle.tools) == 2


class _FakeSrc:
    def __init__(self, specs):
        self._specs = specs

    async def list_ids(self):
        return tuple(self._specs.keys())

    async def get(self, sid):
        if sid not in self._specs:
            raise KeyError(sid)
        return self._specs[sid]
