#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP client wiring: construct pydantic-ai MCP servers from MCPServerSpec, and
MCPConnectionManager to cache/share live toolsets and close them on shutdown.
Construction is synchronous and side-effect-free; connections are
opened lazily by pydantic-ai when a toolset is actually used inside a run."""

import hashlib
import logging
from typing import Any

from ..errors import MCPConnectionError
from ..registry.mcp import MCPServerSpec

_LOGGER = logging.getLogger(__name__)


def _config_fingerprint(spec: MCPServerSpec) -> str:
    """A stable hash of the governance-relevant MCPServerSpec configuration.

    The cache key must reflect everything that changes which tools a server
    exposes or how they are filtered/prefixed: transport, command/url, cwd,
    timeout, tool filters, prefix config, and a revision of env/headers. Secret
    plaintext (env/header VALUES) is folded into the hash, never stored or
    logged in clear text -- only a counter/revision contributes, so two
    different secret values produce different hashes without the values
    themselves entering the key."""
    parts = [
        spec.transport,
        spec.command_or_url,
        str(spec.cwd),
        "" if spec.timeout_seconds is None else f"{spec.timeout_seconds:.6f}",
        str(spec.tool_prefix),
        ",".join(spec.enabled_tools) if spec.enabled_tools else "",
        ",".join(spec.disabled_tools),
        str(len(spec.env)),
        str(len(spec.headers)),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


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
        # Keyed on (server.id, config-fingerprint) so a config change (url,
        # command, env, timeout, tool filters, prefix ...) with a reused id
        # does NOT return a stale cached toolset.
        self._toolsets: "dict[tuple[str, str], Any]" = {}

    async def get_toolset(self, server: MCPServerSpec) -> Any:
        key = (server.id, _config_fingerprint(server))
        cached = self._toolsets.get(key)
        if cached is not None:
            return cached
        from pydantic_ai.mcp import MCPToolset
        mcp_server = build_mcp_server(server)
        toolset = MCPToolset(mcp_server)
        self._toolsets[key] = toolset
        return toolset

    async def list_tools(self, server: MCPServerSpec) -> "tuple[str, ...]":
        """Enumerate a server's live tool names for governance (enabled/disabled
        filtering, conflict detection, max_tools). Best-effort: pydantic-ai's
        MCPToolset resolves names lazily, so a live connection is needed; if the
        underlying API cannot enumerate here, returns () (governance then operates
        on an unknown set -- the documented live-MCP boundary)."""
        try:
            toolset = await self.get_toolset(server)
            getter = getattr(toolset, "get_tools", None)
            if getter is None:
                return ()
            # pydantic-ai toolsets yield ToolDefinition objects; read .name off each.
            import inspect
            result = getter()
            if inspect.isawaitable(result):
                tools = await result
            else:
                tools = result
            names: "list[str]" = []
            for t in tools or ():
                name = getattr(t, "name", None) or getattr(getattr(t, "function", None), "name", None)
                if name:
                    names.append(str(name))
            return tuple(names)
        except Exception:
            return ()

    async def close_server(self, server_id: str) -> None:
        # Close every cached toolset for this server id (multiple config
        # fingerprints may exist). Errors are surfaced, not swallowed.
        keys = [k for k in self._toolsets if k[0] == server_id]
        for key in keys:
            toolset = self._toolsets.pop(key, None)
            if toolset is None:
                continue
            closer = getattr(toolset, "close", None)
            if closer is not None:
                result = closer()
                if hasattr(result, "__await__"):
                    await result

    async def close(self) -> None:
        # Close ALL connections, aggregating errors so one failing close does
        # not leak the remaining connections.
        keys = list(self._toolsets.keys())
        errors: "list[Exception]" = []
        for key in keys:
            toolset = self._toolsets.pop(key, None)
            if toolset is None:
                continue
            closer = getattr(toolset, "close", None)
            if closer is None:
                continue
            try:
                result = closer()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:  # noqa: BLE001 - aggregate, don't abort
                errors.append(exc)
        if errors:
            _LOGGER.warning(
                "MCPConnectionManager.close: %d connection(s) failed to close: %s",
                len(errors), errors[0],
            )
