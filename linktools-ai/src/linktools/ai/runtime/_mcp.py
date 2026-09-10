#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize the MCP transport selected by a compiled runtime binding."""

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai.toolsets import AbstractToolset

from ..capability import (
    mcp_selector_server,
    mcp_server_namespace,
)
from ..core import Principal, ResourceRef
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec


async def materialize_mcp_servers(
    servers: Sequence[MCPServerSpec],
    selectors: Sequence[str],
    *,
    principal: Principal,
    execution: ResourceRef,
    execution_root: str,
) -> "tuple[AbstractToolset[object], ...]":
    """Materialize only compiler-selected stdio servers."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai.mcp import MCPToolset

    if principal.tenant_id != execution.tenant_id:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    policy = _selector_policy(selectors)
    values: list[AbstractToolset[object]] = []
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
            toolset = toolset.filtered(
                lambda _ctx, tool_def, selected=names: tool_def.name in selected
            )
        prefixed = toolset.prefixed(f"mcp__{namespace}__")
        values.append(prefixed)
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


__all__ = ["materialize_mcp_servers"]
