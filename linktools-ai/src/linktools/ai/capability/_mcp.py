#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP capability provider boundary."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.toolsets import AbstractToolset

from ..core import JsonValue, Principal, ResourceRef, canonical_json_bytes
from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    revision: int
    command: str
    args: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1 or not self.command.strip():
            raise ValueError("MCP server spec is incomplete")

@dataclass(frozen=True, slots=True)
class MCPCallRequest:
    principal: Principal
    execution: ResourceRef
    operation_id: str
    server_id: str
    tool_name: str
    arguments: Mapping[str, JsonValue]
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.principal.tenant_id != self.execution.tenant_id or not self.operation_id.strip() or not self.server_id.strip() or not self.tool_name.strip() or not 1 <= self.timeout_seconds <= 900:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if len(canonical_json_bytes(dict(self.arguments))) > 4 * 1024 * 1024:
            raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE)


def validate_mcp_response(value: JsonValue) -> JsonValue:
    if len(canonical_json_bytes(value)) > 4 * 1024 * 1024:
        raise AIError(ErrorCode.MCP_RESPONSE_TOO_LARGE)
    return value


class MCPConnectionPool(Protocol):
    async def call(self, request: MCPCallRequest) -> JsonValue: ...


class MCPToolProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve_ref(self, server_id: str, revision: 'int | None' = None) -> MCPServerSpec: ...
    async def connect(self, server_id: str) -> MCPConnectionPool: ...
    async def toolsets(
        self,
        servers: Sequence[MCPServerSpec],
        *,
        principal: Principal,
        execution: ResourceRef,
    ) -> "tuple[AbstractToolset[None], ...]": ...


__all__ = ["MCPCallRequest", "MCPConnectionPool", "MCPServerSpec", "MCPToolProvider", "validate_mcp_response"]
