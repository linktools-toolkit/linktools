#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpec  + MCPRegistry: loads MCP server
declarations from {name}.yaml via SpecLoader, revision-cached. Mirrors
ToolRegistry's YAML pattern (the loader exposes a revision() monotonic clock;
whenever it changes the per-(id, revision) cache and id listing are dropped)."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import InvalidSpecError, RegistryNotFoundError
from .parser import SpecLoader, StrictConfigReader, parse_yaml_text


_VALID_TRANSPORTS = ("stdio", "sse", "http")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    name: str
    transport: str  # "stdio" | "sse" | "http"
    discovery_mode: str = "strict"  # "strict" | "best_effort"
    # Structured transport fields: stdio carries `command`; sse/http carry `url`.
    command: "tuple[str, ...] | None" = None
    url: "str | None" = None
    cwd: "str | None" = None
    env: "Mapping[str, str]" = field(default_factory=dict)
    headers: "Mapping[str, str]" = field(default_factory=dict)
    timeout_seconds: "float | None" = None
    tool_prefix: "str | bool | None" = None
    enabled_tools: "tuple[str, ...] | None" = None
    disabled_tools: "tuple[str, ...]" = ()
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


def _as_command_tuple(command_raw: Any) -> "tuple[str, ...]":
    if isinstance(command_raw, str):
        if not command_raw.strip():
            raise InvalidSpecError("mcp command must not be empty")
        return (command_raw,)
    if not isinstance(command_raw, list) or any(
        not isinstance(part, str) or not part for part in command_raw
    ):
        raise InvalidSpecError("mcp command must be a string or list of strings")
    return tuple(command_raw)


def parse_mcp_spec(mcp_id: str, payload: "dict[str, Any]") -> MCPServerSpec:
    """Build an MCPServerSpec from a parsed YAML dict.

    - name falls back to mcp_id when omitted.
    - transport comes from `transport`, defaulting to stdio; it must be one of
      {stdio, sse, http}.
    - stdio requires `command`; sse/http require `url`.
    """
    allowed = {
        "name",
        "transport",
        "command",
        "url",
        "cwd",
        "env",
        "headers",
        "timeout_seconds",
        "tool_prefix",
        "enabled_tools",
        "disabled_tools",
        "discovery_mode",
        "metadata",
    }
    reader = StrictConfigReader(payload, allowed=allowed, context=f"mcp {mcp_id}")

    name = reader.optional_str("name") or mcp_id
    transport = reader.optional_str("transport") or "stdio"
    if transport not in _VALID_TRANSPORTS:
        raise InvalidSpecError(
            f"mcp {mcp_id}: unknown transport: {transport!r} "
            f"(expected one of {_VALID_TRANSPORTS})"
        )

    command_raw = payload.get("command")
    command = _as_command_tuple(command_raw) if command_raw is not None else None
    url = reader.optional_str("url")

    # Transport validation: stdio needs a command; sse/http need a url.
    if transport == "stdio":
        if not command:
            raise InvalidSpecError(f"mcp {mcp_id}: stdio transport requires 'command'")
    else:
        if not url:
            raise InvalidSpecError(
                f"mcp {mcp_id}: {transport} transport requires 'url'"
            )

    env = reader.mapping("env") or {}
    headers = reader.mapping("headers") or {}
    metadata = reader.mapping("metadata") or {}
    cwd = reader.optional_str("cwd")
    timeout_seconds = reader.positive_number("timeout_seconds")
    tool_prefix = payload.get("tool_prefix")
    enabled_tools = reader.string_tuple("enabled_tools") or None
    disabled_tools = reader.string_tuple("disabled_tools")
    discovery_mode = reader.optional_str("discovery_mode") or "strict"
    if discovery_mode not in ("strict", "best_effort"):
        raise InvalidSpecError(
            f"mcp {mcp_id}: unknown discovery_mode: {discovery_mode!r} "
            f"(expected 'strict' or 'best_effort')"
        )

    return MCPServerSpec(
        id=mcp_id,
        name=name,
        transport=transport,
        discovery_mode=discovery_mode,
        command=command,
        url=url,
        cwd=cwd,
        env=env,
        headers=headers,
        timeout_seconds=timeout_seconds,
        tool_prefix=tool_prefix,
        enabled_tools=enabled_tools if enabled_tools else None,
        disabled_tools=disabled_tools,
        metadata=metadata,
    )


class MCPRegistry:
    """Loads MCPServerSpecs from `{name}.yaml` files via a SpecLoader,
    revision-cached. Mirrors ToolRegistry."""

    def __init__(self, loader: SpecLoader, *, suffix: str = ".yaml") -> None:
        self._loader = loader
        self._suffix = suffix
        self._cache: "dict[tuple[str, int], MCPServerSpec]" = {}
        self._cached_revision: "int | None" = None
        self._ids: "tuple[str, ...] | None" = None

    async def _ensure_fresh(self) -> None:
        revision = await self._loader.revision()
        if revision != self._cached_revision:
            self._cache.clear()
            self._ids = None
            self._cached_revision = revision

    async def list_ids(self) -> "tuple[str, ...]":
        await self._ensure_fresh()
        if self._ids is None:
            self._ids = await self._loader.list_ids(self._suffix)
        return self._ids

    async def get(self, mcp_id: str) -> MCPServerSpec:
        await self._ensure_fresh()
        revision = self._cached_revision if self._cached_revision is not None else 0
        cache_key = (mcp_id, revision)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            text = await self._loader.read(f"{mcp_id}{self._suffix}")
        except RegistryNotFoundError:
            raise
        payload = parse_yaml_text(text, source=f"{mcp_id}{self._suffix}")
        spec = parse_mcp_spec(mcp_id, payload)
        self._cache[cache_key] = spec
        return spec
