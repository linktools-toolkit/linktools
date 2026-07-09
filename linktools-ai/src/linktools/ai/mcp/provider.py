#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPProvider: the CapabilityProvider for ``mcp:<server_id>`` / ``mcp:*`` tool
refs (spec §11.5/§15). Resolves server specs via an MCPServerSpecProvider and
materializes toolsets through an MCPConnectionManager.

``mcp:*`` (expose every server's every tool) is dangerous and is gated behind
``allow_mcp_wildcard`` -- off by default. Per-server refs are always allowed.

Exposure-control coverage (spec §15.6/§15.7/§15.8/§15.10):
- tool_prefix: applied at MCPServer construction (mcp/client.py).
- enabled_tools / disabled_tools filtering, cross-server conflict detection,
  max_tools_per_capability for MCP, and routing each MCP call through the
  ToolExecutor/Policy/Middleware chain: these require enumerating a server's
  live tool names. pydantic-ai's MCPToolset resolves tool names lazily (only on
  first use inside a run), and MCPServer exposes no native name filter, so they
  cannot be applied or verified without a live MCP server -- the spec itself
  lists live stdio/sse/http as environment-gated acceptance (§15.10/§21.9). The
  pure policy helpers in toolset.py (filter_tool_names / detect_mcp_conflicts /
  final_tool_name) are implemented and unit-tested, and are the integration
  point once a live connection can yield the tool list."""

from dataclasses import dataclass, field
from typing import Any

from ..capability.bundle import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.ref import CapabilityRef
from ..errors import CapabilityResolutionError, MCPServerNotFoundError
from ..providers.mcp import MCPServerSpecProvider
from .client import MCPConnectionManager


@dataclass
class MCPProvider:
    """CapabilityProvider for MCP servers. Both the spec provider and the
    connection manager are injectable so tests can supply fakes; production
    wiring passes a real MCPRegistry + MCPConnectionManager."""

    mcp_provider: MCPServerSpecProvider
    connection_manager: "MCPConnectionManager | None" = None
    allow_mcp_wildcard: bool = False
    kind: str = "mcp"

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        ids = await self._target_ids(ref, context)
        toolsets: "list[Any]" = []
        for server_id in ids:
            spec = await self._spec(server_id)
            toolset = await self._toolset(spec)
            if toolset is not None:
                toolsets.append(toolset)
        return CapabilityBundle(toolsets=tuple(toolsets))

    async def _target_ids(self, ref: CapabilityRef, context: CapabilityContext) -> "tuple[str, ...]":
        if ref.name == "*":
            # Wildcard exposes EVERY server's tools -- deployment-level opt-in
            # only (spec §11.5 #2). The Runtime gate is authoritative; a tool
            # ref's own config must NOT be able to self-grant the wildcard.
            if not self.allow_mcp_wildcard:
                raise CapabilityResolutionError(
                    f"agent {context.agent_id}: mcp:* requires allow_mcp_wildcard=True"
                )
            return await self.mcp_provider.list_ids()
        return (ref.name,)

    async def _spec(self, server_id: str):
        try:
            return await self.mcp_provider.get(server_id)
        except (KeyError, LookupError):
            raise MCPServerNotFoundError(f"mcp server not found: {server_id}") from None

    async def _toolset(self, spec) -> Any:
        if self.connection_manager is None:
            # No live manager configured (e.g. conversational-only runtime):
            # nothing to expose, but not an error.
            return None
        return await self.connection_manager.get_toolset(spec)
