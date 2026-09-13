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
    if (
        not tool_name
        or tool_name != tool_name.strip()
        or "*" in tool_name
        or "__" in tool_name
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return f"{mcp_server_selector(server_id)}__{tool_name}"


__all__ = [
    "mcp_server_namespace",
    "mcp_server_selector",
    "mcp_tool_name",
]
