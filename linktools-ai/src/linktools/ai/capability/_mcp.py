#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard MCP selectors and stdio materialization."""

import re

from ..errors import AIError, ErrorCode


def mcp_server_namespace(server_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", server_id)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return value


def mcp_server_selector(server_id: str) -> str:
    return f"mcp__{mcp_server_namespace(server_id)}"


def mcp_tool_name(server_id: str, tool_name: str) -> str:
    if not tool_name or tool_name != tool_name.strip() or "*" in tool_name:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return f"{mcp_server_selector(server_id)}__{tool_name}"


def mcp_selector_server(selector: str) -> "tuple[str, str | None] | None":
    """Return namespace and exact tool name for one canonical MCP selector."""
    if not selector.startswith("mcp__"):
        return None
    tail = selector[5:]
    namespace, separator, tool = tail.partition("__")
    if not namespace:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not separator:
        return namespace, None
    if not tool:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return namespace, None if tool == "*" else tool


__all__ = [
    "mcp_selector_server",
    "mcp_server_namespace",
    "mcp_server_selector",
    "mcp_tool_name",
]
