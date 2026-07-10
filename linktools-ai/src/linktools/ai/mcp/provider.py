#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPProvider: the CapabilityProvider for ``mcp:<server_id>`` / ``mcp:*`` tool
refs. Resolves server specs via an MCPServerSpecProvider and materializes
toolsets through an MCPConnectionManager.

``mcp:*`` (expose every server's every tool) is dangerous and is gated behind
``allow_mcp_wildcard`` -- off by default. Per-server refs are always allowed.

Exposure control applied at resolve time (when a connection manager is wired):
  1. enumerate live tools via ``MCPConnectionManager.list_tools``;
  2. filter by ``enabled_tools`` / ``disabled_tools``;
  3. apply ``tool_prefix`` to the final names;
  4. detect cross-server name conflicts (no silent overwrite);
  5. enforce ``max_tools_per_capability``.
tool_prefix is also applied at MCPServer construction (mcp/client.py).

Boundary: live tool enumeration depends on the underlying pydantic-ai MCPToolset
cooperating (it resolves names lazily). When ``list_tools`` returns () (cannot
enumerate in this environment), the filter/conflict/cap checks operate on an
empty set and are effectively skipped for live servers -- the live stdio/sse/
http path is environment-gated. The governance logic itself (filter_tool_names /
detect_mcp_conflicts / final_tool_name) is unit-tested with a fake manager that
yields canned tool names."""

from dataclasses import dataclass, field
from typing import Any

from ..capability.bundle import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.ref import CapabilityRef
from ..errors import CapabilityConflictError, CapabilityResolutionError, MCPServerNotFoundError
from ..providers.mcp import MCPServerSpecProvider
from .client import MCPConnectionManager
from .toolset import detect_mcp_conflicts, filter_tool_names, final_tool_name


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
        contributions: "list[Any]" = []
        final_names_by_server: "dict[str, tuple[str, ...]]" = {}
        max_per_cap = context.exposure_policy.max_tools_per_capability
        for server_id in ids:
            spec = await self._spec(server_id)
            # Governance: enumerate -> filter -> prefix -> cap.
            raw = await self._list_tools(spec)
            # Strict discovery: when governance config is present but tools
            # cannot be enumerated, fail closed rather than silently passthrough.
            has_governance = (spec.enabled_tools is not None or spec.disabled_tools
                              or spec.tool_prefix is not None)
            if not raw and has_governance and getattr(spec, "discovery_mode", "strict") == "strict":
                raise CapabilityResolutionError(
                    f"mcp server {server_id!r}: strict discovery mode cannot verify "
                    f"tool governance (enabled/disabled/prefix) without live enumeration"
                )
            filtered = filter_tool_names(raw, spec.enabled_tools, spec.disabled_tools)
            final = tuple(final_tool_name(spec.id, n, spec.tool_prefix) for n in filtered)
            if max_per_cap and len(final) > max_per_cap:
                raise CapabilityConflictError(
                    f"mcp server {server_id!r} exposes {len(final)} tools "
                    f"(max_tools_per_capability={max_per_cap})"
                )
            final_names_by_server[server_id] = final
            toolset = await self._toolset(spec)
            if toolset is not None:
                toolsets.append(toolset)
                # Build ToolContribution with conservative MCP descriptors.
                from ..security.descriptor import ToolDescriptor
                from ..tool.contribution import ToolContribution
                kw = dict(source="mcp", capability_kind="mcp", capability_name=server_id)
                # Conservative: unknown MCP tools are treated as write/high/mutating
                # (default conservative when mutation unknown).
                descs = tuple(
                    ToolDescriptor(
                        name=n, category="mcp-write", risk="high", mutating=True, **kw,
                    ) for n in final
                )
                contributions.append(ToolContribution(toolset=toolset, descriptors=descs))
        detect_mcp_conflicts(final_names_by_server)
        return CapabilityBundle(
            toolsets=tuple(toolsets),
            tool_contributions=tuple(contributions),
        )

    async def _target_ids(self, ref: CapabilityRef, context: CapabilityContext) -> "tuple[str, ...]":
        if ref.name == "*":
            # Wildcard exposes EVERY server's tools -- deployment-level opt-in
            # only. The Runtime gate is authoritative; a tool ref's own config
            # must NOT be able to self-grant the wildcard.
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

    async def _list_tools(self, spec) -> "tuple[str, ...]":
        if self.connection_manager is None:
            return ()
        lister = getattr(self.connection_manager, "list_tools", None)
        if lister is None:
            return ()
        try:
            return await lister(spec)
        except Exception:
            return ()

    async def _toolset(self, spec) -> Any:
        if self.connection_manager is None:
            return None
        return await self.connection_manager.get_toolset(spec)
