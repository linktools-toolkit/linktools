#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MCP connection, discovery, and tool adaptation."""

from .client import (
    MCPConnectionPool,
    MCPToolsetHandle,
    McpCloseFailure,
    McpCloseResult,
    build_mcp_server,
)
from .codec import parse_mcp_spec
from .models import (
    MCPConnectionRef,
    MCPDiscoveryResult,
    MCPExposedTool,
    MCPRuntimePolicy,
    MCPToolInfo,
)
from .spec import MCPServerSpec
from .tool_provider import MCPToolProvider

__all__ = [
    "MCPConnectionPool",
    "MCPConnectionRef",
    "MCPDiscoveryResult",
    "MCPExposedTool",
    "MCPRuntimePolicy",
    "MCPServerSpec",
    "MCPToolInfo",
    "MCPToolProvider",
    "MCPToolsetHandle",
    "McpCloseFailure",
    "McpCloseResult",
    "build_mcp_server",
    "parse_mcp_spec",
]
