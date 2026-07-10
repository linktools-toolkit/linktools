#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic MCP tool-name shaping: per-server prefixing,
enabled/disabled filtering, and cross-server conflict detection. Pure functions
so the policy is testable without a live MCP connection."""

from typing import Iterable, Mapping

from ..errors import CapabilityConflictError


def final_tool_name(
    server_id: str, tool_name: str, tool_prefix: "str | bool | None",
) -> str:
    """Compute the exposed tool name for one MCP tool.

    - tool_prefix False            -> keep the server's original name verbatim.
    - tool_prefix None / True      -> default: ``{server_id}.{tool_name}``.
    - tool_prefix <str>            -> ``{prefix}.{tool_name}``.

    Keeping the original name (tool_prefix=False) is allowed only when the
    caller can guarantee no collisions; conflicts still raise at assembly time.
    """
    if tool_prefix is False:
        return tool_name
    prefix = server_id if tool_prefix in (None, True) else str(tool_prefix)
    return f"{prefix}.{tool_name}"


def filter_tool_names(
    names: "Iterable[str]",
    enabled_tools: "tuple[str, ...] | None",
    disabled_tools: "tuple[str, ...]",
) -> "tuple[str, ...]":
    """Apply enabled_tools then disabled_tools. Order is preserved
    from the input; disabled always wins when both are set."""
    out: "list[str]" = []
    enabled_set = set(enabled_tools) if enabled_tools else None
    disabled_set = set(disabled_tools) if disabled_tools else set()
    for name in names:
        if enabled_set is not None and name not in enabled_set:
            continue
        if name in disabled_set:
            continue
        out.append(name)
    return tuple(out)


def detect_mcp_conflicts(final_names_by_server: "Mapping[str, Iterable[str]]") -> None:
    """Raise CapabilityConflictError if two servers expose the same final tool
    name. MCP tool collisions are never silently overwritten."""
    seen: "dict[str, str]" = {}
    for server_id, names in final_names_by_server.items():
        for final_name in names:
            if final_name in seen:
                raise CapabilityConflictError(
                    f"MCP tool name {final_name!r} exposed by both "
                    f"{seen[final_name]} and {server_id}"
                )
            seen[final_name] = server_id
