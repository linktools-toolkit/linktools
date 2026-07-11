#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.mcp: MCP server capability wiring. Re-exports the extended
MCPServerSpec, the connection manager, and the MCPProvider.

Tool-name shaping (final_tool_name / filter_tool_names / detect_mcp_conflicts)
lives in ``toolset.py`` as a private module: applying it to live MCP toolsets
requires enumerating a server's tools at run time (pydantic-ai's MCPToolset is
lazy, with no native name filter), which is environment-gated. It is
intentionally NOT part of the public surface until a live connection can yield
the tool list -- advertising it would promise capability that is not wired in."""

from ..registry.mcp import MCPServerSpec, parse_mcp_spec
from .client import (
    MCPConnectionManager, MCPConnectionRef, MCPToolsetHandle,
    LegacyMCPConnectionManagerAdapter, build_mcp_server,
)
from .provider import MCPProvider, MCPDiscoveryResult, MCPToolInfo, MCPExposedTool

__all__ = [
    "MCPServerSpec", "parse_mcp_spec",
    "MCPConnectionManager", "MCPConnectionRef", "MCPToolsetHandle",
    "LegacyMCPConnectionManagerAdapter", "build_mcp_server",
    "MCPProvider",
    "MCPDiscoveryResult", "MCPToolInfo", "MCPExposedTool",
]
