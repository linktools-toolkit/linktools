#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP connection wiring and lifecycle.

Construct pydantic-ai MCP servers from MCPServerSpec, and cache/share live
toolsets until runtime shutdown.
Construction is synchronous and side-effect-free; connections are
opened lazily by pydantic-ai when a toolset is actually used inside a run."""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ...errors import MCPConnectionUnavailableError
from ...json import canonical_json_bytes
from .models import MCPConnectionRef, MCPDiscoveryResult
from .spec import MCPServerSpec
from .client import MCPClient, build_mcp_server

_LOGGER = logging.getLogger(__name__)


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
        "env_digest": _digest_mapping(spec.env),
        "headers_digest": _digest_mapping(spec.headers),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:16]


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
        handle = await self.get_toolset(server)
        return await MCPClient(handle.toolset).discover(
            server_id=server.id,
            connection_ref=handle.connection_ref,
        )

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
            raise MCPConnectionUnavailableError(
                f"MCP connection {key!r} is not available"
            )
        return await MCPClient(toolset).call(
            server_id=connection_ref.server_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    async def close_server(self, server_id: str) -> None:
        keys = [key for key in self._toolsets if key[0] == server_id]
        for key in keys:
            toolset = self._toolsets.pop(key, None)
            if toolset is None:
                continue
            await MCPClient(toolset).close()

    async def close(self) -> None:
        keys = list(self._toolsets)
        errors: list[Exception] = []
        for key in keys:
            toolset = self._toolsets.pop(key, None)
            if toolset is None:
                continue
            try:
                await MCPClient(toolset).close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            _LOGGER.warning("MCP connection close failures: %d", len(errors))
