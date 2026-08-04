#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP MCP descriptor validation and conversion."""

import hashlib
import json
from typing import Any, Mapping

from .errors import request_error


def mcp_spec(descriptor: Any) -> Any:
    """Convert one official ACP descriptor to the existing MCP domain type."""
    from ..agent.mcp.spec import MCPServerSpec

    kind = _mcp_transport(descriptor)
    name = getattr(descriptor, "name", "")
    if not name:
        raise request_error("invalid_mcp_descriptor")
    if kind == "stdio":
        env = {item.name: item.value for item in getattr(descriptor, "env", ())}
        command = (descriptor.command, *tuple(getattr(descriptor, "args", ())))
        return MCPServerSpec(
            id=name,
            name=name,
            transport="stdio",
            command=command,
            env=env,
        )
    if kind in {"http", "sse"}:
        headers = {
            item.name: item.value
            for item in getattr(descriptor, "headers", ())
        }
        return MCPServerSpec(
            id=name,
            name=name,
            transport=kind,
            url=descriptor.url,
            headers=headers,
        )
    raise request_error("unsupported_mcp_transport", details={"transport": kind})


def validate_mcp_descriptors(descriptors: Any) -> None:
    try:
        for descriptor in descriptors:
            mcp_spec(descriptor)
    except Exception as exc:
        if getattr(exc, "code", None) == -32602:
            raise
        raise request_error("invalid_mcp_descriptor") from exc


def mcp_descriptor_fingerprint(descriptor: object) -> str:
    if hasattr(descriptor, "model_dump"):
        raw = descriptor.model_dump(mode="json", by_alias=True, exclude_none=True)
    elif isinstance(descriptor, Mapping):
        raw = dict(descriptor)
    else:
        raw = {"value": str(descriptor)}
    raw = _without_secret_fields(raw)
    encoded = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mcp_transport(descriptor: Any) -> str:
    kind = getattr(descriptor, "type", None)
    if kind is not None:
        return str(kind)
    return {
        "McpServerStdio": "stdio",
        "McpServerHttp": "http",
        "McpServerSse": "sse",
        "McpServerAcp": "acp",
    }.get(type(descriptor).__name__, "unknown")


def _without_secret_fields(value: Any) -> Any:
    secret_names = {
        "authorization",
        "api_key",
        "apikey",
        "env",
        "environment",
        "headers",
        "password",
        "secret",
        "token",
    }
    if isinstance(value, Mapping):
        return {
            key: _without_secret_fields(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in secret_names
        }
    if isinstance(value, (list, tuple)):
        return [_without_secret_fields(item) for item in value]
    return value


__all__ = ["mcp_descriptor_fingerprint", "mcp_spec", "validate_mcp_descriptors"]
