#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpec (section 27 minimal) + MCPRegistry: loads MCP server
declarations from {name}.yaml via SpecLoader, revision-cached. Mirrors
ToolRegistry's YAML pattern (the loader exposes a revision() monotonic clock;
whenever it changes the per-(id, revision) cache and id listing are dropped)."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import InvalidSpecError, RegistryNotFoundError
from .parser import SpecLoader, parse_yaml_text


_VALID_TRANSPORTS = ("stdio", "sse", "http")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    name: str
    transport: str  # "stdio" | "sse" | "http"
    command_or_url: str
    env: "Mapping[str, str]" = field(default_factory=dict)
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


def parse_mcp_spec(mcp_id: str, payload: "dict[str, Any]") -> MCPServerSpec:
    """Build an MCPServerSpec from a parsed YAML dict.

    - name falls back to mcp_id when omitted.
    - transport comes from `transport` (or legacy `type`), defaulting to stdio;
      it must be one of {stdio, sse, http}.
    - command_or_url is the `command` (stdio, list joined with spaces) or the
      `url` (sse/http). At least one must be present.
    """
    name = payload.get("name") or mcp_id
    transport = str(payload.get("transport") or payload.get("type") or "stdio")
    if transport not in _VALID_TRANSPORTS:
        raise InvalidSpecError(
            f"mcp {mcp_id}: unknown transport: {transport!r} "
            f"(expected one of {_VALID_TRANSPORTS})"
        )

    command_raw = payload.get("command")
    url_raw = payload.get("url")
    if command_raw is not None:
        if isinstance(command_raw, (list, tuple)):
            command_or_url = " ".join(str(part) for part in command_raw)
        else:
            command_or_url = str(command_raw)
    elif url_raw is not None:
        command_or_url = str(url_raw)
    else:
        raise InvalidSpecError(
            f"mcp {mcp_id}: 'command' (stdio) or 'url' (sse/http) is required"
        )

    env = dict(payload.get("env") or {})
    metadata = dict(payload.get("metadata") or {})
    return MCPServerSpec(
        id=mcp_id,
        name=str(name),
        transport=transport,
        command_or_url=command_or_url,
        env=env,
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
