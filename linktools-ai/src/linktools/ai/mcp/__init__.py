#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.mcp: MCP server capability wiring (spec §15). Re-exports the
extended MCPServerSpec, the connection manager, the tool-name shaping helpers,
and the MCPProvider."""

from ..registry.mcp import MCPServerSpec, parse_mcp_spec
from .client import MCPConnectionManager, build_mcp_server
from .provider import MCPProvider
from .toolset import detect_mcp_conflicts, filter_tool_names, final_tool_name

__all__ = [
    "MCPServerSpec", "parse_mcp_spec",
    "MCPConnectionManager", "build_mcp_server",
    "final_tool_name", "filter_tool_names", "detect_mcp_conflicts",
    "MCPProvider",
]
