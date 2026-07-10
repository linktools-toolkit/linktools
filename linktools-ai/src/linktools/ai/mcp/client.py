#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP client wiring: construct pydantic-ai MCP servers from MCPServerSpec, and
MCPConnectionManager to cache/share live toolsets and close them on shutdown.
Construction is synchronous and side-effect-free; connections are
opened lazily by pydantic-ai when a toolset is actually used inside a run."""

from typing import Any

from ..errors import MCPConnectionError
from ..registry.mcp import MCPServerSpec


def _resolved_tool_prefix(spec: MCPServerSpec) -> "str | None":
    """Map spec.tool_prefix onto the pydantic-ai MCPServer tool_prefix arg.

    None / True -> default server_id prefix; <str> -> that prefix; False -> None
    (keep the server's original tool names; collisions then fail at assembly)."""
    tp = spec.tool_prefix
    if tp is False:
        return None
    if tp in (None, True):
        return spec.id
    return str(tp)


def build_mcp_server(spec: MCPServerSpec) -> Any:
    """Build the pydantic-ai MCPServer for a spec (stdio/sse/http). Raises
    MCPConnectionError for a misconfigured transport. ``command``/``url`` are
    read from the structured fields (command_or_url is a compat-only string).
    The per-server ``tool_prefix`` is applied here."""
    from pydantic_ai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio

    timeout = spec.timeout_seconds
    prefix = _resolved_tool_prefix(spec)
    if spec.transport == "stdio":
        if not spec.command:
            raise MCPConnectionError(f"mcp {spec.id}: stdio requires a command")
        # MCPServerStdio splits the executable from its args.
        return MCPServerStdio(
            command=spec.command[0],
            args=list(spec.command[1:]),
            cwd=spec.cwd,
            env=dict(spec.env),
            timeout=timeout,
            tool_prefix=prefix,
        )
    if spec.transport == "sse":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: sse requires a url")
        return MCPServerSSE(url=spec.url, headers=dict(spec.headers), timeout=timeout,
                            tool_prefix=prefix)
    if spec.transport == "http":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: http requires a url")
        return MCPServerHTTP(url=spec.url, headers=dict(spec.headers), timeout=timeout,
                             tool_prefix=prefix)
    raise MCPConnectionError(f"mcp {spec.id}: unknown transport {spec.transport!r}")


class MCPConnectionManager:
    """Owns the lifecycle of live MCP toolsets. ``get_toolset`` builds (and
    caches) a pydantic-ai MCPToolset for a server; ``close`` / ``close_server``
    release them. Runtime closes this on shutdown so connections do not leak."""

    def __init__(self) -> None:
        self._toolsets: "dict[str, Any]" = {}

    async def get_toolset(self, server: MCPServerSpec) -> Any:
        cached = self._toolsets.get(server.id)
        if cached is not None:
            return cached
        from pydantic_ai.mcp import MCPToolset
        mcp_server = build_mcp_server(server)
        toolset = MCPToolset(mcp_server)
        self._toolsets[server.id] = toolset
        return toolset

    async def close_server(self, server_id: str) -> None:
        toolset = self._toolsets.pop(server_id, None)
        if toolset is None:
            return
        closer = getattr(toolset, "close", None)
        if closer is not None:
            result = closer()
            if hasattr(result, "__await__"):
                await result

    async def close(self) -> None:
        ids = list(self._toolsets.keys())
        for server_id in ids:
            await self.close_server(server_id)
