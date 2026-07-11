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
from .client import MCPConnectionRef
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

    def __post_init__(self) -> None:
        from ..utils.freeze import freeze_value
        if not isinstance(self.name, str) or not self.name.strip():
            from ..errors import MCPToolDefinitionError
            raise MCPToolDefinitionError("MCP tool name must be non-empty")
        try:
            from ..tool.schema_validate import validate_schema
            validate_schema(self.parameters_json_schema)
        except Exception as exc:
            from ..errors import MCPToolDefinitionError
            raise MCPToolDefinitionError(
                f"invalid schema for MCP tool {self.name!r}") from exc
        object.__setattr__(self, "parameters_json_schema",
                           freeze_value(dict(self.parameters_json_schema)))
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@dataclass(frozen=True)
class MCPDiscoveryResult:
    tools: tuple[MCPToolInfo, ...] = ()
    verified: bool = False
    error: BaseException | None = None
    connection_ref: MCPConnectionRef | None = None


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
                from ..events.payloads import SecurityDegraded
                if context.security_event_emitter is None:
                    raise CapabilityResolutionError(
                        f"mcp server {server_id!r}: best-effort discovery requires "
                        "a security event emitter")
                await context.security_event_emitter.emit_security(SecurityDegraded(
                    run_id=context.run_id, component="mcp-discovery",
                    reason=str(discovery.error or "tool enumeration unavailable"),
                    error_code=getattr(getattr(discovery.error, "code", None), "value",
                                       getattr(discovery.error, "code", None)),
                    server_id=server_id,
                    connection_fingerprint=(
                        discovery.connection_ref.fingerprint[:12]
                        if discovery.connection_ref else None
                    ),
                ))
            filtered = filter_tool_names(raw, spec.enabled_tools, spec.disabled_tools)
            final = tuple(final_tool_name(spec.id, n, spec.tool_prefix) for n in filtered)
            if max_per_cap and len(final) > max_per_cap:
                raise CapabilityConflictError(
                    f"mcp server {server_id!r} exposes {len(final)} tools "
                    f"(max_tools_per_capability={max_per_cap})"
                )
            final_names_by_server[server_id] = final
            if self.connection_manager is not None and not hasattr(self.connection_manager, "call_tool"):
                raise CapabilityResolutionError(
                    "MCP connection manager must implement call_tool(connection_ref=...)")
            from ..security.descriptor import ToolDescriptor
            from ..tool.contribution import ToolContribution, ManagedToolDefinition
            kw = dict(source="mcp", capability_kind="mcp", capability_name=server_id)
            info_by_name = {info.name: info for info in raw_infos}
            exposed_tools = [MCPExposedTool(
                server_id=server_id, raw_name=r, exposed_name=e,
                parameters_json_schema=info_by_name.get(r, MCPToolInfo(r)).parameters_json_schema,
                description=info_by_name.get(r, MCPToolInfo(r)).description,
                read_only=info_by_name.get(r, MCPToolInfo(r)).read_only,
                metadata=info_by_name.get(r, MCPToolInfo(r)).metadata,
            ) for r, e in zip(filtered, final)]
            descs = tuple(
                ToolDescriptor(
                    name=et.exposed_name, category="mcp-write", risk="high",
                    mutating=not bool(et.read_only),
                    metadata={"raw_name": et.raw_name, **dict(et.metadata)}, **kw,
                )
                for et in exposed_tools
            )
            definitions = tuple(ManagedToolDefinition(
                descriptor=d,
                handler=self._handler(et, discovery.connection_ref),
                parameters_json_schema=et.parameters_json_schema,
                description=et.description,
            ) for d, et in zip(descs, exposed_tools))
            contributions.append(ToolContribution(tools=definitions))
        detect_mcp_conflicts(final_names_by_server)
        return CapabilityBundle(
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

    async def _discover(self, spec) -> MCPDiscoveryResult:
        if self.connection_manager is None:
            return MCPDiscoveryResult((), True, None)
        result_getter = getattr(self.connection_manager, "list_tools_result", None)
        if result_getter is None:
            from ..errors import MCPDiscoveryUnsupportedError
            return MCPDiscoveryResult(
                (), False, MCPDiscoveryUnsupportedError(
                    "MCP manager must implement list_tools_result"))
        return await result_getter(spec)

    def _handler(self, exposed: MCPExposedTool, connection_ref: MCPConnectionRef | None):
        async def call(**arguments: Any) -> Any:
            if connection_ref is None:
                raise CapabilityResolutionError("MCP discovery did not return a connection reference")
            return await self.connection_manager.call_tool(
                connection_ref=connection_ref, tool_name=exposed.raw_name,
                arguments=arguments)
        return call
