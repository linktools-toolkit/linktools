#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Construct one low-level Pydantic AI MCP client from a server spec."""


from dataclasses import dataclass
from typing import Any, Mapping
from linktools.core import environ
from ...errors import MCPAuthenticationError, MCPConnectionError, MCPDiscoveryError, MCPDiscoveryUnsupportedError, MCPToolDefinitionError
from .models import MCPDiscoveryResult, MCPToolInfo

from typing import TYPE_CHECKING

logger = environ.get_logger("ai.agent.mcp.client")

if TYPE_CHECKING:
    from .models import MCPConnectionRef
    from .spec import MCPServerSpec

def _resolved_tool_prefix(spec: "MCPServerSpec") -> "str | None":
    value = spec.tool_prefix
    if value is False:
        return None
    if value is None or value is True:
        return spec.id
    return value


def build_mcp_server(spec: "MCPServerSpec") -> Any:
    from pydantic_ai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio

    prefix = _resolved_tool_prefix(spec)
    if spec.transport == "stdio":
        if not spec.command:
            raise MCPConnectionError(f"mcp {spec.id}: stdio requires a command")
        return MCPServerStdio(
            command=spec.command[0],
            args=list(spec.command[1:]),
            cwd=spec.cwd,
            env=dict(spec.env),
            timeout=spec.timeout_seconds,
            tool_prefix=prefix,
        )
    if spec.transport == "sse":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: sse requires a url")
        return MCPServerSSE(
            url=spec.url,
            headers=dict(spec.headers),
            timeout=spec.timeout_seconds,
            tool_prefix=prefix,
        )
    if spec.transport == "http":
        if not spec.url:
            raise MCPConnectionError(f"mcp {spec.id}: http requires a url")
        return MCPServerHTTP(
            url=spec.url,
            headers=dict(spec.headers),
            timeout=spec.timeout_seconds,
            tool_prefix=prefix,
        )
    raise MCPConnectionError(
        f"mcp {spec.id}: unknown transport {spec.transport!r}"
    )


@dataclass(slots=True)
class MCPClient:
    """Low-level calls against one already constructed SDK client."""

    toolset: Any

    async def discover(
        self,
        *,
        server_id: str,
        connection_ref: "MCPConnectionRef",
    ) -> MCPDiscoveryResult:
        try:
            lister = getattr(self.toolset, "list_tools", None)
            if lister is None:
                return MCPDiscoveryResult(
                    (),
                    False,
                    MCPDiscoveryUnsupportedError(
                        f"MCP server {server_id!r} cannot enumerate tools"
                    ),
                    connection_ref,
                )
            raw_tools = await lister()
            return MCPDiscoveryResult(
                tuple(self.convert_tool_info(tool) for tool in raw_tools or ()),
                True,
                None,
                connection_ref,
            )
        except Exception as error:
            normalized = self.normalize_discovery_error(error)
            logger.warning(
                "MCP discovery failed (server=%s tools may be unavailable): %s: %s",
                server_id, type(normalized).__name__, normalized,
            )
            return MCPDiscoveryResult(
                (),
                False,
                normalized,
                connection_ref,
            )

    async def call(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: "Mapping[str, Any]",
    ) -> Any:
        caller = getattr(self.toolset, "direct_call_tool", None)
        if caller is None:
            raise MCPConnectionError(
                f"MCP server {server_id!r} has no direct tool caller"
            )
        return await caller(tool_name, dict(arguments))

    async def close(self) -> None:
        closer = getattr(self.toolset, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result

    @staticmethod
    def convert_tool_info(tool: Any) -> MCPToolInfo:
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
            raise MCPToolDefinitionError(
                f"invalid schema for MCP tool {name!r}"
            )
        annotations = getattr(tool, "annotations", None)
        hint = (
            getattr(annotations, "readOnlyHint", None)
            if annotations is not None
            else None
        )
        return MCPToolInfo(
            name=name,
            description=getattr(tool, "description", None),
            parameters_json_schema=schema,
            read_only=True if hint is True else False if hint is False else None,
            metadata=getattr(tool, "metadata", {}) or {},
        )

    @staticmethod
    def normalize_discovery_error(error: BaseException) -> BaseException:
        if isinstance(error, MCPDiscoveryError):
            return error
        name = type(error).__name__.lower()
        text = str(error).lower()
        if "auth" in name or "unauthorized" in text or "forbidden" in text:
            return MCPAuthenticationError("MCP authentication failed")
        if "unsupported" in name or "notimplemented" in name:
            return MCPDiscoveryUnsupportedError(
                "MCP discovery is unsupported"
            )
        if "connect" in name or "timeout" in name or "transport" in name:
            return MCPConnectionError("MCP connection failed")
        return MCPDiscoveryError("MCP discovery failed")


__all__ = ["MCPClient", "build_mcp_server"]
