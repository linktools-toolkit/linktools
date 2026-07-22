#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP client wiring: construct pydantic-ai MCP servers from MCPServerSpec, and
MCPConnectionPool to cache/share live toolsets and close them on shutdown.
Construction is synchronous and side-effect-free; connections are
opened lazily by pydantic-ai when a toolset is actually used inside a run."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import MCPConnectionError, MCPDiscoveryError, MCPDiscoveryUnsupportedError
from ..json import canonical_json
from .spec import MCPServerSpec

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MCPConnectionRef:
    server_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class MCPToolsetHandle:
    connection_ref: MCPConnectionRef
    toolset: Any


def _digest_mapping(values: "Mapping[str, str]") -> str:
    """Irreversible SHA-256 digest of a mapping's canonical JSON. Two different
    secret VALUES (or keys) produce different digests, but the secret plaintext
    never enters the fingerprint, logs, or exceptions."""
    canonical = json.dumps(
        sorted(values.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _config_fingerprint(spec: MCPServerSpec) -> str:
    """A stable hash of the governance-relevant MCPServerSpec configuration.

    The cache key reflects everything that changes which tools a server exposes
    or how they are filtered/prefixed: transport, command/url, cwd, timeout,
    tool filters, prefix config, discovery mode, and a DIGEST of env/headers.
    The digest covers both keys AND values, so a changed secret value (e.g. a
    rotated Authorization token) invalidates the cache -- without the secret
    plaintext ever entering the key, logs, or exceptions.

    The payload is canonical JSON (sorted keys, compact) so there is no
    ambiguous delimiter-based join: ``("a b","c")`` and ``("a","b c")`` hash
    differently, and a None allowlist hashes differently from an empty one."""
    payload = {
        "transport": spec.transport,
        "command": list(spec.command) if spec.command is not None else None,
        "url": spec.url,
        "cwd": spec.cwd,
        "timeout_seconds": spec.timeout_seconds,
        "tool_prefix": spec.tool_prefix,
        "enabled_tools": (
            list(spec.enabled_tools) if spec.enabled_tools is not None else None
        ),
        "disabled_tools": list(spec.disabled_tools),
        "discovery_mode": getattr(spec, "discovery_mode", "strict"),
        "env_digest": _digest_mapping(spec.env),
        "headers_digest": _digest_mapping(spec.headers),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


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
    read from the structured fields (command/url).
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
        return MCPServerSSE(
            url=spec.url,
            headers=dict(spec.headers),
            timeout=timeout,
            tool_prefix=prefix,
        )
    if spec.transport == "http":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: http requires a url")
        return MCPServerHTTP(
            url=spec.url,
            headers=dict(spec.headers),
            timeout=timeout,
            tool_prefix=prefix,
        )
    raise MCPConnectionError(f"mcp {spec.id}: unknown transport {spec.transport!r}")


class MCPConnectionPool:
    """Owns the lifecycle of live MCP toolsets. ``get_toolset`` builds (and
    caches) a pydantic-ai MCPToolset for a server; ``close`` / ``close_server``
    release them. Runtime closes this on shutdown so connections do not leak."""

    def __init__(self) -> None:
        # Keyed on (server.id, config-fingerprint) so a config change (url,
        # command, env, timeout, tool filters, prefix ...) with a reused id
        # does NOT return a stale cached toolset.
        self._toolsets: "dict[tuple[str, str], Any]" = {}
        # Per-key lock so two concurrent get_toolset() calls for the same
        # server build only ONE toolset (double-checked locking).
        self._lock = asyncio.Lock()

    async def get_toolset(self, server: MCPServerSpec) -> MCPToolsetHandle:
        key = (server.id, _config_fingerprint(server))
        cached = self._toolsets.get(key)
        if cached is not None:
            return MCPToolsetHandle(MCPConnectionRef(*key), cached)
        async with self._lock:
            # Double-check inside the lock.
            cached = self._toolsets.get(key)
            if cached is not None:
                return MCPToolsetHandle(MCPConnectionRef(*key), cached)

            toolset = build_mcp_server(server)
            self._toolsets[key] = toolset
            return MCPToolsetHandle(MCPConnectionRef(*key), toolset)

    async def list_tools(self, server: MCPServerSpec) -> "tuple[str, ...]":
        """Enumerate a server's live tool names for governance (enabled/disabled
        filtering, conflict detection, max_tools). Best-effort: pydantic-ai's
        MCPToolset resolves names lazily, so a live connection is needed; if the
        underlying API cannot enumerate here, returns () (governance then operates
        on an unknown set -- the documented live-MCP boundary)."""
        result = await self.list_tools_result(server)
        return tuple(item.name for item in result.tools)

    async def list_tools_result(self, server: MCPServerSpec):
        from .provider import MCPDiscoveryResult

        handle = None
        try:
            handle = await self.get_toolset(server)
            toolset = handle.toolset
            lister = getattr(toolset, "list_tools", None)
            if lister is None:
                return MCPDiscoveryResult(
                    (),
                    False,
                    MCPDiscoveryUnsupportedError(
                        f"MCP server {server.id!r} cannot enumerate tools"
                    ),
                    handle.connection_ref,
                )
            raw_tools = await lister()
            tools = tuple(self._convert_tool_info(t) for t in raw_tools or ())
            return MCPDiscoveryResult(tools, True, None, handle.connection_ref)
        except Exception as exc:
            error = self._normalize_discovery_error(exc)
            return MCPDiscoveryResult(
                (), False, error, handle.connection_ref if handle else None
            )

    @staticmethod
    def _convert_tool_info(tool: Any):
        from .provider import MCPToolInfo
        from ..errors import MCPToolDefinitionError

        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise MCPToolDefinitionError("MCP tool name must be non-empty")
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or getattr(tool, "parameters_json_schema", None)
            or {"type": "object", "properties": {}}
        )
        if not isinstance(schema, Mapping):
            raise MCPToolDefinitionError(f"invalid schema for MCP tool {name!r}")
        from ..tool.schema import validate_schema

        try:
            validate_schema(schema)
        except Exception as exc:
            raise MCPToolDefinitionError(
                f"invalid schema for MCP tool {name!r}"
            ) from exc
        # Only an explicit readOnlyHint annotation marks a tool non-mutating;
        # absent annotations stay None so unknown tools remain high-risk at the
        # provider layer (mutating = not bool(read_only)).
        annotations = getattr(tool, "annotations", None)
        hint = (
            getattr(annotations, "readOnlyHint", None)
            if annotations is not None
            else None
        )
        if hint is True:
            read_only = True
        elif hint is False:
            read_only = False
        else:
            read_only = None
        return MCPToolInfo(
            name=name,
            description=getattr(tool, "description", None),
            parameters_json_schema=schema,
            read_only=read_only,
            metadata=getattr(tool, "metadata", {}) or {},
        )

    @staticmethod
    def _normalize_discovery_error(exc: BaseException):
        from ..errors import MCPAuthenticationError, MCPConnectionError

        if isinstance(exc, MCPDiscoveryError):
            return exc
        name = type(exc).__name__.lower()
        text = str(exc)
        if (
            "auth" in name
            or "unauthorized" in text.lower()
            or "forbidden" in text.lower()
        ):
            return MCPAuthenticationError("MCP authentication failed")
        if "unsupported" in name or "notimplemented" in name:
            return MCPDiscoveryUnsupportedError("MCP discovery is unsupported")
        if "connect" in name or "timeout" in name or "transport" in name:
            return MCPConnectionError("MCP connection failed")
        return MCPDiscoveryError("MCP discovery failed")

    async def call_tool(
        self,
        *,
        connection_ref: MCPConnectionRef,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        key = (connection_ref.server_id, connection_ref.fingerprint)
        toolset = self._toolsets.get(key)
        if toolset is None:
            from ..errors import MCPConnectionUnavailableError

            raise MCPConnectionUnavailableError(
                f"MCP connection {key!r} is not available"
            )
        caller = getattr(toolset, "direct_call_tool", None)
        if caller is None:
            raise MCPConnectionError(
                f"MCP server {connection_ref.server_id!r} has no direct tool caller"
            )
        return await caller(tool_name, dict(arguments))

    async def close_server(self, server_id: str) -> None:
        keys = [key for key in self._toolsets if key[0] == server_id]
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
        keys = list(self._toolsets)
        errors: list[Exception] = []
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
            except Exception as exc:
                errors.append(exc)
        if errors:
            _LOGGER.warning("MCP connection close failures: %d", len(errors))
