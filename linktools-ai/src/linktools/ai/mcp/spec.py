#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpec: an immutable MCP server declaration (moved here from
registry/mcp.py so the mcp domain owns its spec type). Carries the full
transport/discovery contract + ``__post_init__`` domain invariants so a custom
provider constructing one directly cannot build an ungovernable server."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


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
        # directly, bypassing the codec, so the spec itself must enforce the
        # same contract. Validation only -- the codec owns normalization
        # (stripping, defaults); this layer just rejects an object that could
        # never be a usable, governable MCP server.
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


@runtime_checkable
class MCPServerSpecProvider(Protocol):
    """Provides MCPServerSpec objects from any configuration source.

    Lives in the mcp domain (co-located with MCPServerSpec) so the mcp package
    does not import providers back just to reference its own provider surface
    (the providers package re-exports it for RuntimeDependencies)."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, server_id: str) -> "MCPServerSpec": ...


__all__: "list[str]" = ["MCPServerSpec", "MCPServerSpecProvider"]
