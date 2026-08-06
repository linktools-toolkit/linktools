#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP capability provider boundary."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    revision: int
    command: str
    args: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1 or not self.command.strip():
            raise ValueError("MCP server spec is incomplete")

    @property
    def asset_kind(self) -> str:
        return "mcp"

    @property
    def asset_id(self) -> str:
        return self.id


class MCPConnectionPool(Protocol):
    async def call(self, server_id: str, tool_name: str, arguments: 'dict[str, str]') -> str: ...


class MCPToolProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve_ref(self, server_id: str, revision: 'int | None' = None) -> MCPServerSpec: ...
    async def connect(self, server_id: str) -> MCPConnectionPool: ...


__all__ = ["MCPConnectionPool", "MCPServerSpec", "MCPToolProvider"]
