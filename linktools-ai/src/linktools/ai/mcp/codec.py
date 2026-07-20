#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPSpecCodec: the CatalogCodec[MCPServerSpec] for the mcp domain.

Owns the mcp-specific parsing (moved here from registry/mcp.py): a
``{name}.yaml`` item is parsed as YAML, strictly validated, and built into an
MCPServerSpec. Parse failures propagate the domain's existing errors
(InvalidSpecError / RegistryParseError)."""

from __future__ import annotations

from typing import Any

from ..catalog import CatalogCodec
from ..catalog.parsing import (
    StrictConfigReader,
    parse_yaml_text,
    resolved_name,
)
from ..errors import InvalidSpecError
from .env import expand_env_mapping
from .spec import MCPServerSpec, _VALID_TRANSPORTS


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

    # Expand ${ENV_NAME} references (fail-on-missing) at parse time so a server
    # never receives a literal placeholder and startup fails fast on an unset
    # secret.
    env = expand_env_mapping(reader.string_mapping("env") or {})
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


class MCPSpecCodec:
    """CatalogCodec[MCPServerSpec]: decode one ``{id}.yaml`` item's raw text
    into an MCPServerSpec. Strict; propagates the domain's rich errors."""

    def decode(self, item_id: str, raw: str) -> MCPServerSpec:
        source = f"{item_id}.yaml"
        payload = parse_yaml_text(raw, source=source)
        return parse_mcp_spec(item_id, payload)


__all__: "list[str]" = ["MCPSpecCodec", "parse_mcp_spec"]
