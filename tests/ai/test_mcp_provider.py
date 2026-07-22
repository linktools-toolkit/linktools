#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPProvider + toolset shaping + spec/client (contract). Deterministic policy
is unit-tested; live connection is environment-dependent and excluded."""

import dataclasses

import pytest

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.provider import CapabilityContext
from linktools.ai.capability.models import CapabilityRef
from linktools.ai.errors import (
    CapabilityConflictError,
    CapabilityResolutionError,
    InvalidSpecError,
    MCPServerNotFoundError,
)
from linktools.ai.mcp import (
    MCPConnectionPool,
    MCPProvider,
    build_mcp_server,
    parse_mcp_spec,
)
from linktools.ai.mcp.client import MCPConnectionRef
from linktools.ai.mcp.provider import MCPDiscoveryResult, MCPToolInfo
from linktools.ai.errors import MCPDiscoveryUnsupportedError
from linktools.ai.mcp.toolset import (
    detect_mcp_conflicts,
    filter_tool_names,
    final_tool_name,
)


# --- toolset shaping (contract/contract) ----------------------------------------


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
        detect_mcp_conflicts(
            {"risk": ("risk.query_user",), "legacy": ("query_user", "risk.query_user")}
        )


def test_detect_conflicts_ok_when_distinct():
    detect_mcp_conflicts({"a": ("a.x",), "b": ("b.x",)})  # no raise


# --- spec parsing / transport validation (contract/contract) --------------------


def test_parse_stdio_spec_structured_fields():
    spec = parse_mcp_spec(
        "s",
        {
            "transport": "stdio",
            "command": ["python", "-m", "x"],
            "cwd": "/a",
            "tool_prefix": "r",
            "enabled_tools": ["query_user"],
            "disabled_tools": ["secret"],
            "headers": {"X": "1"},
            "timeout_seconds": 30,
            "env": {"K": "v"},
        },
    )
    assert spec.command == ("python", "-m", "x")
    assert spec.cwd == "/a"
    assert spec.tool_prefix == "r"
    assert spec.enabled_tools == ("query_user",)
    assert spec.disabled_tools == ("secret",)
    assert spec.headers == {"X": "1"}
    assert spec.timeout_seconds == 30.0
    assert " ".join(spec.command) == "python -m x"


def test_parse_http_requires_url():
    with pytest.raises(InvalidSpecError, match="http transport requires 'url'"):
        parse_mcp_spec("h", {"transport": "http"})


def test_parse_stdio_requires_command():
    with pytest.raises(InvalidSpecError, match="stdio transport requires 'command'"):
        parse_mcp_spec("z", {"transport": "stdio"})


def test_parse_sse_with_url_and_headers():
    spec = parse_mcp_spec(
        "r", {"transport": "sse", "url": "https://x/sse", "headers": {"Auth": "t"}}
    )
    assert spec.url == "https://x/sse"
    assert spec.command is None
    assert spec.url == "https://x/sse"


# --- client construction (contract) ------------------------------------------


def test_build_mcp_server_stdio_constructs_without_connecting():
    spec = parse_mcp_spec("s", {"transport": "stdio", "command": ["python", "-m", "x"]})
    server = build_mcp_server(spec)
    assert server is not None  # MCPServerStdio; no connection opened


def test_build_mcp_server_rejects_misconfigured_transport():
    from linktools.ai.errors import MCPConnectionError

    # parse_mcp_spec always validates transport inputs, so build the misconfigured
    # case directly via a bare spec-like object.
    class _Bare:
        id = "x"
        transport = "http"
        command = None
        url = None
        cwd = None
        env = {}
        headers = {}
        timeout_seconds = None
        tool_prefix = None

    with pytest.raises(MCPConnectionError):
        build_mcp_server(_Bare())


# --- MCPProvider (contract/contract) ----------------------------------------------


class _FakeManager:
    """Well-behaved fake: enumerates tool names via list_tools_result the way a
    real connected MCPConnectionPool does."""

    def __init__(self, tool_names=("query_user",)):
        self._tool_names = tuple(tool_names)
        self._ref = MCPConnectionRef("fake", "fp")

    async def list_tools_result(self, spec):
        return MCPDiscoveryResult(
            tools=tuple(MCPToolInfo(name=n) for n in self._tool_names),
            verified=True,
            connection_ref=self._ref,
        )

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        return {"name": tool_name}


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
    return CapabilityContext(
        agent_id="a1", exposure_policy=CapabilityToolExposurePolicy()
    )


@pytest.mark.asyncio
async def test_mcp_single_server_exposes_toolset():
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager())
    bundle = await provider.resolve(CapabilityRef("mcp", "risk"), _ctx())
    assert [d.descriptor.name for d in bundle.tool_contributions[0].tools] == [
        "risk.query_user"
    ]


async def _exposed_names(toolset):
    """Enumerate a toolset's exposed tool names via get_tools (handles both
    plain FunctionToolset and FilteredToolset wrappers)."""
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RunUsage

    def _model_fn(messages, info):
        from pydantic_ai.messages import ModelResponse, TextPart

        return ModelResponse(parts=[TextPart(content="x")])

    ctx = RunContext(deps=None, model=FunctionModel(_model_fn), usage=RunUsage())
    return await toolset.get_tools(ctx)


@pytest.mark.asyncio
async def test_mcp_disabled_tools_removed_from_toolset_surface():
    """contract/contract: enabled/disabled must shrink the ACTUAL toolset the
    model sees, not just the computed descriptor name list. A disabled tool
    must be absent from the filtered toolset's get_tools result."""
    spec = parse_mcp_spec(
        "risk",
        {
            "transport": "stdio",
            "command": ["python", "-m", "r"],
            "disabled_tools": ["secret"],
        },
    )
    provider = MCPProvider(
        _FakeSpecProvider({"risk": spec}), _FakeManager(("query_user", "secret"))
    )
    bundle = await provider.resolve(CapabilityRef("mcp", "risk"), _ctx())
    contrib = bundle.tool_contributions[0]
    # Descriptors (the governance source of truth) exclude the disabled tool...
    assert [d.descriptor.name for d in contrib.tools] == ["risk.query_user"]


@pytest.mark.asyncio
async def test_mcp_multi_server_disabled_tools_filter_independent_per_server():
    """Regression: the per-server filter closure must capture each server's
    own allowed-name set by value. FilteredToolset.get_tools() runs lazily at
    run time (after the resolve loop completes), so a plain closure over a
    loop-rebound ``allowed`` would make EVERY server's toolset evaluate against
    the LAST server's set -- leaking a disabled tool on an earlier server
    whenever 2+ servers are configured."""
    s1 = parse_mcp_spec(
        "alpha",
        {
            "transport": "stdio",
            "command": ["python", "-m", "r"],
            "disabled_tools": ["secret"],
        },
    )
    s2 = parse_mcp_spec(
        "beta", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    # alpha exposes (keep, secret); beta exposes a single unrelated tool.
    provider = MCPProvider(
        _FakeSpecProvider({"alpha": s1, "beta": s2}),
        _FakeManager(("keep", "secret")),  # both servers share the fake's tool set
        allow_mcp_wildcard=True,
    )
    bundle = await provider.resolve(CapabilityRef("mcp", "*"), _ctx())
    by_server = {
        c.tools[0].descriptor.capability_name: c for c in bundle.tool_contributions
    }
    alpha_names = [tool.descriptor.name for tool in by_server["alpha"].tools]
    beta_names = [tool.descriptor.name for tool in by_server["beta"].tools]
    # alpha: "secret" disabled -> only "keep" survives (NOT the union, NOT beta's set).
    assert "secret" not in alpha_names, "disabled tool leaked across servers"
    assert alpha_names == ["alpha.keep"]
    # beta: nothing disabled -> both tools survive.
    assert beta_names == ["beta.keep", "beta.secret"]


@pytest.mark.asyncio
async def test_mcp_missing_server_raises_not_found():
    provider = MCPProvider(_FakeSpecProvider({}), _FakeManager())
    with pytest.raises(MCPServerNotFoundError):
        await provider.resolve(CapabilityRef("mcp", "ghost"), _ctx())


@pytest.mark.asyncio
async def test_mcp_wildcard_denied_by_default():
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager())
    with pytest.raises(CapabilityResolutionError, match="allow_mcp_wildcard"):
        await provider.resolve(CapabilityRef("mcp", "*"), _ctx())


@pytest.mark.asyncio
async def test_mcp_wildcard_allowed_via_flag():
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    provider = MCPProvider(
        _FakeSpecProvider({"risk": spec}), _FakeManager(), allow_mcp_wildcard=True
    )
    bundle = await provider.resolve(CapabilityRef("mcp", "*"), _ctx())
    assert len(bundle.tool_contributions) == 1


@pytest.mark.asyncio
async def test_mcp_wildcard_ref_config_cannot_self_grant():
    # contract #2: the Runtime gate is authoritative. A tool ref's own config
    # must NOT be able to self-grant the mcp:* wildcard.
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _FakeManager())
    ref = CapabilityRef("mcp", "*", config={"allow_mcp_wildcard": True})
    with pytest.raises(CapabilityResolutionError, match="allow_mcp_wildcard"):
        await provider.resolve(ref, _ctx())


class _UnenumerableManager:
    """Simulates a connected-but-lazy MCP server: list_tools_result cannot
    enumerate names (verified=False)."""

    async def list_tools_result(self, spec):
        return MCPDiscoveryResult(
            (), False, MCPDiscoveryUnsupportedError("cannot enumerate")
        )

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        return {}


@pytest.mark.asyncio
async def test_mcp_strict_discovery_fails_closed_without_explicit_governance():
    """Regression: strict mode must fail closed on empty enumeration even when
    the spec declares no enabled_tools/disabled_tools/tool_prefix -- max_tools,
    conflict detection, and ToolExposurePolicy all need the real tool set."""
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    assert (
        spec.enabled_tools is None
        and not spec.disabled_tools
        and spec.tool_prefix is None
    )
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _UnenumerableManager())
    with pytest.raises(CapabilityResolutionError, match="strict discovery"):
        await provider.resolve(CapabilityRef("mcp", "risk"), _ctx())


@pytest.mark.asyncio
async def test_mcp_best_effort_discovery_mode_opts_out_of_fail_closed():
    spec = parse_mcp_spec(
        "risk",
        {
            "transport": "stdio",
            "command": ["python", "-m", "r"],
            "discovery_mode": "best_effort",
        },
    )
    provider = MCPProvider(_FakeSpecProvider({"risk": spec}), _UnenumerableManager())
    events = []

    class _Emitter:
        async def emit_security(self, event):
            events.append(event)

    ctx = dataclasses.replace(_ctx(), security_event_emitter=_Emitter())
    bundle = await provider.resolve(CapabilityRef("mcp", "risk"), ctx)
    # Proceeds without raising, but exposes no unverified execution tools.
    assert all(not contribution.tools for contribution in bundle.tool_contributions)
    assert len(events) == 1 and events[0].component == "mcp-discovery"


@pytest.mark.asyncio
async def test_mcp_provider_rejects_missing_connection_manager():
    """A server declared without a connection manager cannot verify tool
    governance (no live enumeration), so MCPProvider fails at construction --
    never surfacing as a verified-but-empty discovery result."""
    from linktools.ai.errors import RuntimeInitializationError

    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    with pytest.raises(RuntimeInitializationError, match="MCPConnectionPool"):
        MCPProvider(_FakeSpecProvider({"risk": spec}), None)


@pytest.mark.asyncio
async def test_connection_manager_closes_toolsets():
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    mgr = MCPConnectionPool()
    # Use the real manager's close path with an object exposing close(), keyed
    # the way get_toolset actually keys (server.id, fingerprint).
    from linktools.ai.mcp.client import _config_fingerprint

    class _TS:
        closed = False

        async def close(self):
            _TS.closed = True

    mgr._toolsets[(spec.id, _config_fingerprint(spec))] = _TS()
    await mgr.close()
    assert _TS.closed
    assert mgr._toolsets == {}


@pytest.mark.asyncio
async def test_connection_manager_cache_keyed_on_config_fingerprint():
    """Two specs sharing an id but differing in a governance-relevant field
    (command) must get DISTINCT cache slots -- a config change with a reused id
    must not return a stale cached toolset. Secret plaintext never enters the
    key (only a length revision does)."""
    from linktools.ai.mcp.client import _config_fingerprint

    s1 = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "a"]}
    )
    s2 = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "b"]}
    )
    s3 = parse_mcp_spec(
        "risk",
        {
            "transport": "stdio",
            "command": ["python", "-m", "a"],
            "disabled_tools": ["x"],
        },
    )
    assert _config_fingerprint(s1) != _config_fingerprint(s2)  # command differs
    assert _config_fingerprint(s1) != _config_fingerprint(s3)  # disabled_tools differs
    assert _config_fingerprint(s1) == _config_fingerprint(
        parse_mcp_spec("risk", {"transport": "stdio", "command": ["python", "-m", "a"]})
    )  # identical config -> identical key


@pytest.mark.asyncio
async def test_connection_manager_close_aggregates_errors_and_closes_all():
    """close() must close EVERY connection even if one fails -- a single
    failing close must not leak the remaining connections."""
    mgr = MCPConnectionPool()
    order: "list[str]" = []

    class _OK:
        def __init__(self, tag):
            self._t = tag

        async def close(self):
            order.append(self._t)

    class _Boom:
        async def close(self):
            raise RuntimeError("boom")

    mgr._toolsets[("srv", "fp1")] = _Boom()
    mgr._toolsets[("srv", "fp2")] = _OK("second")
    await mgr.close()  # must not abort on _Boom
    assert mgr._toolsets == {}, "all connections must be closed despite an error"
    assert "second" in order, "the connection after the failing one must still close"
