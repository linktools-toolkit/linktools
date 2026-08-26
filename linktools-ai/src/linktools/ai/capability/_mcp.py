#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard MCP selectors and stdio materialization."""

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai.capabilities.mcp import MCP

from ..core import Principal, ResourceRef
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec


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


async def materialize_mcp_servers(
    servers: Sequence[MCPServerSpec],
    selectors: Sequence[str],
    *,
    principal: Principal,
    execution: ResourceRef,
    execution_root: str,
) -> "tuple[MCP[object], ...]":
    """Materialize only compiler-selected stdio servers using the compiled selector policy."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai.mcp import MCPToolset

    if principal.tenant_id != execution.tenant_id:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    policy = _selector_policy(selectors)
    values: list[MCP[object]] = []
    seen_namespaces: set[str] = set()
    for server in servers:
        namespace = mcp_server_namespace(server.id)
        if namespace in seen_namespaces:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        seen_namespaces.add(namespace)
        allowed = policy.get(namespace)
        if allowed is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        client = Client(
            StdioTransport(
                server.command,
                list(server.args),
                cwd=str(Path(execution_root).expanduser().resolve()),
            )
        )
        toolset = MCPToolset(client, id=f"mcp:{server.id}")
        if allowed:
            names = frozenset(allowed)
            toolset = toolset.filtered(lambda _ctx, tool_def, selected=names: tool_def.name in selected)
        prefixed = toolset.prefixed(f"mcp__{namespace}__")
        values.append(MCP(local=prefixed, id=mcp_server_selector(server.id)))
    return tuple(values)


def _selector_policy(selectors: Sequence[str]) -> "dict[str, frozenset[str]]":
    result: dict[str, set[str] | None] = {}
    for selector in selectors:
        parsed = mcp_selector_server(selector)
        if parsed is None:
            continue
        namespace, tool = parsed
        if tool is None:
            result[namespace] = None
            continue
        current = result.get(namespace)
        if current is None and namespace in result:
            continue
        if current is None:
            current = set()
            result[namespace] = current
        current.add(tool)
    return {
        namespace: frozenset() if tools is None else frozenset(tools)
        for namespace, tools in result.items()
    }


__all__ = [
    "materialize_mcp_servers",
    "mcp_selector_server",
    "mcp_server_namespace",
    "mcp_server_selector",
    "mcp_tool_name",
]
