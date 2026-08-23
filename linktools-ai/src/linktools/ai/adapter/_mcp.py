"""Pydantic AI MCP runtime adapter."""

from collections.abc import Sequence
from pathlib import Path

from ..capability import MCPRuntime
from ..core import Principal, ResourceRef, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec


class PydanticMCPRuntime(MCPRuntime):
    def __init__(self, *, fingerprint: "str | None" = None) -> None:
        self._fingerprint = fingerprint or canonical_sha256({"provider": "pydantic-mcp-stdio", "version": 1})

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def toolsets(
        self,
        servers: Sequence[MCPServerSpec],
        *,
        principal: Principal,
        execution: ResourceRef,
        execution_root: str,
    ) -> "tuple[object, ...]":
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport
        from pydantic_ai.mcp import MCPToolset

        if principal.tenant_id != execution.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        values = []
        for server in servers:
            client = Client(
                StdioTransport(
                    server.command,
                    list(server.args),
                    cwd=str(Path(execution_root).expanduser().resolve()),
                )
            )
            values.append(MCPToolset(client, id=f"mcp:{server.id}"))
        return tuple(values)


__all__ = ["PydanticMCPRuntime"]
