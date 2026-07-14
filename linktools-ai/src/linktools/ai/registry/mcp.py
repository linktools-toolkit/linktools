#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpec  + MCPRegistry: loads MCP server
declarations from {name}.yaml via SpecLoader, revision-cached. Mirrors
ToolRegistry's YAML pattern (the loader exposes a revision() monotonic clock;
whenever it changes the per-(id, revision) cache and id listing are dropped)."""

from dataclasses import dataclass, field
import asyncio
from typing import Any, Mapping
import math

from ..errors import InvalidSpecError, RegistryNotFoundError
from .parser import SpecLoader, StrictConfigReader, parse_yaml_text, resolved_name


_VALID_TRANSPORTS = ("stdio", "sse", "http")
_VALID_DISCOVERY_MODES = ("strict", "best_effort")


def _validate_string_mapping(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a Mapping[str, str]")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field_name} must be a Mapping[str, str]")


def _validate_string_tuple(value: Any, *, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple[str, ...]")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must be a tuple[str, ...]")


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

    def __post_init__(self) -> None:
        # Domain invariants: a custom provider can construct an MCPServerSpec
        # directly, bypassing the registry parser, so the spec itself must
        # enforce the same contract. Validation only -- the registry owns
        # normalization (stripping, defaults); this layer just rejects an
        # object that could never be a usable, governable MCP server.
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("MCPServerSpec id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MCPServerSpec name must be a non-empty string")
        if self.transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"unknown transport: {self.transport!r} "
                f"(expected one of {_VALID_TRANSPORTS})"
            )
        if self.discovery_mode not in _VALID_DISCOVERY_MODES:
            raise ValueError(
                f"unknown discovery_mode: {self.discovery_mode!r} "
                f"(expected one of {_VALID_DISCOVERY_MODES})"
            )
        if self.command is not None:
            _validate_string_tuple(self.command, field_name="command")
        if self.enabled_tools is not None:
            _validate_string_tuple(self.enabled_tools, field_name="enabled_tools")
        _validate_string_tuple(self.disabled_tools, field_name="disabled_tools")
        _validate_string_mapping(self.env, field_name="env")
        _validate_string_mapping(self.headers, field_name="headers")
        if self.timeout_seconds is not None:
            if (
                isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0
            ):
                raise ValueError("timeout_seconds must be a positive finite number")
        # tool_prefix: None/bool (flag) or a non-empty string prefix.
        if self.tool_prefix is not None and not isinstance(self.tool_prefix, bool):
            if not isinstance(self.tool_prefix, str):
                raise TypeError("tool_prefix must be a string, boolean, or None")
            if not self.tool_prefix.strip():
                raise ValueError("tool_prefix must not be empty")
        # Transport compatibility: stdio needs a command, sse/http need a url.
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio transport requires a non-empty command")
        elif not self.url:
            raise ValueError(f"{self.transport} transport requires a url")


def _as_command_tuple(command_raw: Any) -> "tuple[str, ...]":
    """Normalize an MCP ``command`` (string or list of strings). Each element
    must be a string whose stripped form is non-empty (a whitespace-only arg is
    rejected), but the ORIGINAL element is preserved so an intentionally-spaced
    argument is not altered. An empty list is rejected."""
    if isinstance(command_raw, str):
        if not command_raw.strip():
            raise InvalidSpecError("mcp command must not be empty")
        return (command_raw,)
    if not isinstance(command_raw, list) or not command_raw:
        raise InvalidSpecError(
            "mcp command must be a non-empty string or list of strings"
        )
    command: "list[str]" = []
    for index, part in enumerate(command_raw):
        if not isinstance(part, str):
            raise InvalidSpecError(f"mcp command[{index}] must be a string")
        if not part.strip():
            raise InvalidSpecError(f"mcp command[{index}] must not be blank")
        command.append(part)
    return tuple(command)


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

    name = resolved_name(reader, mcp_id)
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

    env = reader.string_mapping("env") or {}
    headers = reader.string_mapping("headers") or {}
    metadata = reader.mapping("metadata") or {}
    cwd = reader.optional_str("cwd")
    timeout_seconds = reader.positive_number("timeout_seconds")
    tool_prefix = reader.str_or_bool("tool_prefix")
    enabled_tools = reader.string_tuple("enabled_tools", default=None)
    disabled_tools = reader.string_tuple("disabled_tools", default=())
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
        enabled_tools=enabled_tools,
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
        self._refresh_lock = asyncio.Lock()

    async def _ensure_fresh(self) -> None:
        async with self._refresh_lock:
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
