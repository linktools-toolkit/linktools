#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit local MCP runtime provider for workspace composition."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import Principal, ResourceRef, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

    from ._mcp import MCPConnectionPool


class PydanticMCPRuntimeProvider:
    """Create one isolated Pydantic AI MCP toolset per declared local server."""

    def __init__(self, *, execution_root: "str | Path | None" = None, fingerprint: str | None = None) -> None:
        self._fingerprint = fingerprint or canonical_sha256({"provider": "pydantic-mcp-stdio", "version": 1})
        self._execution_root = None if execution_root is None else Path(execution_root).expanduser().resolve()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def connect(self, server_id: str) -> "MCPConnectionPool":
        raise NotImplementedError("MCP connections are created per execution toolset")

    async def toolsets(
        self,
        servers: Sequence[MCPServerSpec],
        *,
        principal: Principal,
        execution: ResourceRef,
    ) -> "tuple[AbstractToolset[None], ...]":
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport
        from pydantic_ai.mcp import MCPToolset

        del principal, execution
        values: list[AbstractToolset[None]] = []
        for server in servers:
            client = Client(
                StdioTransport(
                    server.command,
                    list(server.args),
                    cwd=None if self._execution_root is None else str(self._execution_root),
                )
            )
            values.append(MCPToolset(client, id=f"mcp:{server.id}"))
        if len(values) != len(servers) or len({value.id for value in values}) != len(values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(values)


__all__ = ["PydanticMCPRuntimeProvider"]
