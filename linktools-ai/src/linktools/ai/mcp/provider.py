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

Strict discovery (``MCPServerSpec.discovery_mode`` defaults to ``"strict"``):
when a connection manager is wired but ``list_tools`` returns no names, the
server fails closed with ``CapabilityResolutionError`` rather than silently
proceeding with an empty/unenumerated tool set -- max_tools, conflict
detection, ToolExposurePolicy and ToolPolicyProvider all need the real tool
set to do their job. A ``MCPConnectionManager.list_tools`` implementation
(real or fake) MUST cooperate with enumeration for a server to be usable under
strict discovery; ``discovery_mode="best_effort"`` opts a server out. The
governance logic itself (filter_tool_names / detect_mcp_conflicts /
final_tool_name) is unit-tested with a fake manager that yields canned tool
names."""

from dataclasses import dataclass, field
from typing import Mapping
from typing import Any, ClassVar

from ..capability.bundle import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.ref import CapabilityRef
from ..errors import CapabilityConflictError, CapabilityResolutionError, MCPServerNotFoundError
from ..providers.mcp import MCPServerSpecProvider
from .client import MCPConnectionManager
from .toolset import detect_mcp_conflicts, filter_tool_names, final_tool_name


@dataclass(frozen=True)
class MCPExposedTool:
    """One MCP tool's name contract: the raw server-side name (used for the
    actual MCP call) and the exposed name (the only name the model ever sees,
    after prefixing). Carried so the raw->exposed mapping is explicit and
    auditable rather than implicit in two parallel name lists."""
    server_id: str
    raw_name: str
    exposed_name: str
    parameters_json_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    read_only: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ..utils.freeze import freeze_value
        object.__setattr__(self, "parameters_json_schema", freeze_value(dict(self.parameters_json_schema)))
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    parameters_json_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    read_only: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPDiscoveryResult:
    tools: tuple[MCPToolInfo, ...] = ()
    verified: bool = False
    error: BaseException | None = None


@dataclass
class MCPProvider:
    """CapabilityProvider for MCP servers. Both the spec provider and the
    connection manager are injectable so tests can supply fakes; production
    wiring passes a real MCPRegistry + MCPConnectionManager."""

    mcp_provider: MCPServerSpecProvider
    connection_manager: "MCPConnectionManager | None" = None
    allow_mcp_wildcard: bool = False
    kind: str = "mcp"
    supported_kinds: "ClassVar[frozenset[str]]" = frozenset({"mcp"})

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
            discovery = await self._discover(spec)
            raw_infos = discovery.tools
            raw = tuple(info.name for info in raw_infos)
            # Strict discovery (the default): a connected server whose tools
            # cannot be enumerated fails closed, full stop -- not only when
            # enabled_tools/disabled_tools/tool_prefix happen to be declared.
            # max_tools, conflict detection, ToolExposurePolicy, and
            # ToolPolicyProvider all need the real tool set to do their job;
            # proceeding on an empty/unknown set would silently skip every one
            # of them. ``connection_manager is None`` is a distinct, explicit
            # "MCP declared but not wired" no-op mode, not a discovery failure.
            discovery_mode = getattr(spec, "discovery_mode", "strict")
            if not discovery.verified and self.connection_manager is not None and discovery_mode == "strict":
                raise CapabilityResolutionError(
                    f"mcp server {server_id!r}: strict discovery mode cannot verify "
                    f"tool governance without live enumeration (list_tools "
                    f"returned no tools) -- set discovery_mode='best_effort' to opt out"
                )
            if (not discovery.verified and self.connection_manager is not None
                    and discovery_mode == "best_effort"):
                from ..capability.provider import make_event_emitter
                from ..events.payloads import SecurityDegraded
                try:
                    await make_event_emitter(context)(SecurityDegraded(
                        run_id=context.run_id, component="mcp-discovery",
                        reason=str(discovery.error or "tool enumeration unavailable"),
                    ))
                except Exception:
                    pass
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
                # Filter the ACTUAL toolset, not just the computed name list --
                # enabled/disabled must shrink the tool surface the model sees,
                # not merely the descriptor name list.
                # A tool is kept iff its exposed name is one of the surviving
                # names. The surviving set is the union of the final (prefixed)
                # names and the raw filtered names: the real MCPToolset exposes
                # prefixed names (prefix is applied at MCPServer construction),
                # while a plain FunctionToolset exposes raw names -- the union
                # covers both without ever letting a disabled name through.
                allowed = set(final) | set(filtered)
                if allowed:
                    # Default-arg capture binds ``allowed`` by value at each
                    # iteration. FilteredToolset.get_tools() runs lazily at
                    # agent-run time (after this loop completes), so a plain
                    # closure over ``allowed`` would see only the LAST server's
                    # set for every server -- leaking disabled tools / dropping
                    # legitimate ones whenever 2+ MCP servers are configured.
                    filtered_toolset = toolset.filtered(
                        lambda _ctx, tool_def, _allowed=allowed: tool_def.name in _allowed)
                else:
                    filtered_toolset = toolset.filtered(
                        lambda _ctx, tool_def: False)
                toolsets.append(filtered_toolset)
                # Build ToolContribution with conservative MCP descriptors. The
                # model only ever sees the EXPOSED (prefixed) name; the raw name
                # is carried in descriptor metadata for audit so the MCP call
                # (which uses the raw name) is traceable to the descriptor.
                from ..security.descriptor import ToolDescriptor
                from ..tool.contribution import ToolContribution, ManagedToolDefinition
                kw = dict(source="mcp", capability_kind="mcp", capability_name=server_id)
                # The explicit raw->exposed mapping (one MCPExposedTool per
                # surviving tool) drives descriptor naming so the contract is
                # centralized, not inferred from two parallel name lists.
                info_by_name = {info.name: info for info in raw_infos}
                exposed_tools = [MCPExposedTool(
                    server_id=server_id, raw_name=r, exposed_name=e,
                    parameters_json_schema=info_by_name.get(r, MCPToolInfo(r)).parameters_json_schema,
                    description=info_by_name.get(r, MCPToolInfo(r)).description,
                    read_only=info_by_name.get(r, MCPToolInfo(r)).read_only,
                    metadata=info_by_name.get(r, MCPToolInfo(r)).metadata,
                ) for r, e in zip(filtered, final)]
                # Conservative: unknown MCP tools are treated as write/high/mutating
                # (default conservative when mutation unknown).
                descs = tuple(
                    ToolDescriptor(
                        name=et.exposed_name, category="mcp-write", risk="high",
                        mutating=not bool(et.read_only), metadata={"raw_name": et.raw_name, **dict(et.metadata)}, **kw,
                    )
                    for et in exposed_tools
                )
                if hasattr(self.connection_manager, "call_tool"):
                    definitions = tuple(ManagedToolDefinition(
                        descriptor=d,
                        handler=self._handler(et),
                        parameters_json_schema=et.parameters_json_schema,
                    ) for d, et in zip(descs, exposed_tools))
                    contributions.append(ToolContribution(tools=definitions))
                else:
                    contributions.append(ToolContribution(toolset=filtered_toolset, descriptors=descs))
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

    async def _discover(self, spec) -> MCPDiscoveryResult:
        if self.connection_manager is None:
            return MCPDiscoveryResult((), True, None)
        result_getter = getattr(self.connection_manager, "list_tools_result", None)
        if result_getter is not None:
            return await result_getter(spec)
        try:
            names = tuple(MCPToolInfo(name=n) for n in await self._list_tools(spec))
            return MCPDiscoveryResult(names, bool(names), None)
        except Exception as exc:
            return MCPDiscoveryResult((), False, exc)

    def _handler(self, exposed: MCPExposedTool):
        async def call(**arguments: Any) -> Any:
            return await self.connection_manager.call_tool(
                server_id=exposed.server_id, tool_name=exposed.raw_name,
                arguments=arguments)
        return call
    async def _toolset(self, spec) -> Any:
        if self.connection_manager is None:
            return None
        return await self.connection_manager.get_toolset(spec)
