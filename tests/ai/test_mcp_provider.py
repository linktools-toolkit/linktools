#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPProvider + toolset shaping + spec/client (spec §15). Deterministic policy
is unit-tested; live connection is environment-dependent and excluded."""

import pytest

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import (
    CapabilityConflictError, CapabilityResolutionError, InvalidSpecError,
    MCPServerNotFoundError,
)
from linktools.ai.mcp import (
    MCPConnectionManager, MCPProvider, build_mcp_server, parse_mcp_spec,
)
from linktools.ai.mcp.toolset import (
    detect_mcp_conflicts, filter_tool_names, final_tool_name,
)


# --- toolset shaping (§15.6/§15.7) ----------------------------------------

def test_final_tool_name_defaults_to_server_prefix():
    assert final_tool_name("risk", "query_user", None) == "risk.query_user"
    assert final_tool_name("risk", "query_user", True) == "risk.query_user"


def test_final_tool_name_custom_prefix():
    assert final_tool_name("risk", "query_user", "r") == "r.query_user"


def test_final_tool_name_false_keeps_original():
    assert final_tool_name("risk", "query_user", False) == "query_user"


def test_filter_enabled_then_disabled():
    names = ("a", "b", "c", "d")
    assert filter_tool_names(names, ("a", "b", "c"), ("b",)) == ("a", "c")


def test_filter_disabled_only():
    assert filter_tool_names(("a", "b"), None, ("a",)) == ("b",)


def test_detect_conflicts_raises_on_duplicate_final_name():
    with pytest.raises(CapabilityConflictError, match="query_user"):
        detect_mcp_conflicts({"risk": ("risk.query_user",), "legacy": ("query_user", "risk.query_user")})


def test_detect_conflicts_ok_when_distinct():
    detect_mcp_conflicts({"a": ("a.x",), "b": ("b.x",)})  # no raise


# --- spec parsing / transport validation (§15.1/§15.5) --------------------

def test_parse_stdio_spec_structured_fields():
    spec = parse_mcp_spec("s", {"transport": "stdio", "command": ["python", "-m", "x"], "cwd": "/a",
                                "tool_prefix": "r", "enabled_tools": ["query_user"], "disabled_tools": ["secret"],
                                "headers": {"X": "1"}, "timeout_seconds": 30, "env": {"K": "v"}})
    assert spec.command == ("python", "-m", "x")
    assert spec.cwd == "/a"
    assert spec.tool_prefix == "r"
    assert spec.enabled_tools == ("query_user",)
    assert spec.disabled_tools == ("secret",)
    assert spec.headers == {"X": "1"}
    assert spec.timeout_seconds == 30.0
    assert spec.command_or_url == "python -m x"  # backward-compat derivation


def test_parse_http_requires_url():
    with pytest.raises(InvalidSpecError, match="http transport requires 'url'"):
        parse_mcp_spec("h", {"transport": "http"})


def test_parse_stdio_requires_command():
    with pytest.raises(InvalidSpecError, match="stdio transport requires 'command'"):
        parse_mcp_spec("z", {"transport": "stdio"})


def test_parse_sse_with_url_and_headers():
    spec = parse_mcp_spec("r", {"transport": "sse", "url": "https://x/sse", "headers": {"Auth": "t"}})
    assert spec.url == "https://x/sse"
    assert spec.command is None
    assert spec.command_or_url == "https://x/sse"


# --- client construction (§15.3) ------------------------------------------

def test_build_mcp_server_stdio_constructs_without_connecting():
    spec = parse_mcp_spec("s", {"transport": "stdio", "command": ["python", "-m", "x"]})
    server = build_mcp_server(spec)
    assert server is not None  # MCPServerStdio; no connection opened


def test_build_mcp_server_rejects_misconfigured_transport():
    from linktools.ai.errors import MCPConnectionError

    # parse_mcp_spec always validates transport inputs, so build the misconfigured
    # case directly via a bare spec-like object.
    class _Bare:
        id = "x"; transport = "http"; command = None; url = None
        cwd = None; env = {}; headers = {}; timeout_seconds = None; tool_prefix = None

    with pytest.raises(MCPConnectionError):
        build_mcp_server(_Bare())


# --- MCPProvider (§11.5/§15) ----------------------------------------------

class _FakeManager:
    def __init__(self):
        self.closed = []

    async def get_toolset(self, spec):
        from pydantic_ai.toolsets import FunctionToolset
        ts = FunctionToolset()
        async def query_user(user_id: str = "") -> dict:
            """query a user"""
            return {"id": user_id}
        ts.add_function(query_user)
        return ts

    async def close(self):
        self.closed.append("all")


class _FakeSpecProvider:
    def __init__(self, specs):
        self._specs = specs

    async def list_ids(self):
        return tuple(self._specs.keys())

    async def get(self, server_id):
        if server_id not in self._specs:
            raise KeyError(server_id)
        return self._specs[server_id]


def _ctx():
    return CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy())


@pytest.mark.asyncio
async def test_mcp_single_server_exposes_toolset():
    spec = parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager())
    bundle = await provider.resolve(CapabilityRef("mcp", "risk"), _ctx())
    assert len(bundle.toolsets) == 1
    assert "query_user" in bundle.toolsets[0].tools


@pytest.mark.asyncio
async def test_mcp_missing_server_raises_not_found():
    provider = MCPProvider(_FakeSpecProvider({}), _FakeManager())
    with pytest.raises(MCPServerNotFoundError):
        await provider.resolve(CapabilityRef("mcp", "ghost"), _ctx())


@pytest.mark.asyncio
async def test_mcp_wildcard_denied_by_default():
    spec = parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager())
    with pytest.raises(CapabilityResolutionError, match="allow_mcp_wildcard"):
        await provider.resolve(CapabilityRef("mcp", "*"), _ctx())


@pytest.mark.asyncio
async def test_mcp_wildcard_allowed_via_flag():
    spec = parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager(), allow_mcp_wildcard=True)
    bundle = await provider.resolve(CapabilityRef("mcp", "*"), _ctx())
    assert len(bundle.toolsets) == 1


@pytest.mark.asyncio
async def test_mcp_wildcard_ref_config_cannot_self_grant():
    # spec §11.5 #2: the Runtime gate is authoritative. A tool ref's own config
    # must NOT be able to self-grant the mcp:* wildcard.
    spec = parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager())
    ref = CapabilityRef("mcp", "*", config={"allow_mcp_wildcard": True})
    with pytest.raises(CapabilityResolutionError, match="allow_mcp_wildcard"):
        await provider.resolve(ref, _ctx())


@pytest.mark.asyncio
async def test_mcp_no_connection_manager_yields_empty(tmp_path):
    spec = parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), None)
    bundle = await provider.resolve(CapabilityRef("mcp", "risk"), _ctx())
    assert bundle.toolsets == ()


@pytest.mark.asyncio
async def test_connection_manager_closes_toolsets():
    spec = parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    mgr = MCPConnectionManager()
    # Use the real manager's close path with an object exposing close().
    class _TS:
        closed = False
        async def close(self):
            _TS.closed = True
    mgr._toolsets["risk"] = _TS()
    await mgr.close()
    assert _TS.closed
    assert mgr._toolsets == {}
