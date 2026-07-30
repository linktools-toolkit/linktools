#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPToolProvider: the AgentFeatureProvider for ``mcp:<server_id>`` / ``mcp:*`` tool
refs. Resolves server specs via an MCPServerSpecProvider and materializes
toolsets through an MCPConnectionPool.

``mcp:*`` (expose every server's every tool) is dangerous and is gated behind
``allow_mcp_wildcard`` -- off by default. Per-server refs are always allowed.

Exposure control applied at resolve time (when a connection manager is wired):
  1. enumerate live tools via ``MCPConnectionPool.list_tools``;
  2. filter by ``enabled_tools`` / ``disabled_tools``;
  3. apply ``tool_prefix`` to the final names;
  4. detect cross-server name conflicts (no silent overwrite);
  5. enforce ``max_tools_per_server``.
tool_prefix is also applied at MCPServer construction (mcp/client.py).

Strict discovery (``MCPRuntimePolicy.discovery_mode`` defaults to ``"strict"``):
when a connection manager is wired but ``list_tools`` returns no names, the
server fails closed with ``AgentAssemblyError`` rather than silently
proceeding with an empty/unenumerated tool set -- max_tools, conflict
detection, ToolExposurePolicy and ToolPolicyResolver all need the real tool
set to do their job. A ``MCPConnectionPool.list_tools`` implementation
(real or fake) MUST cooperate with enumeration for a server to be usable under
strict discovery; ``discovery_mode="best_effort"`` opts a server out. The
governance logic itself (filter_tool_names / detect_mcp_conflicts /
final_tool_name) is unit-tested with a fake manager that yields canned tool
names."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Mapping

from ..assembly.models import AgentContribution, AgentFeatureRef
from ..assembly.provider import AgentFeatureContext
from ...errors import (
    AgentFeatureConflictError,
    AgentAssemblyError,
    MCPServerNotFoundError,
    RuntimeInitializationError,
)
from ...governance.policy.rule import RiskLevel, SideEffectKind
from ..tool.models import (
    ToolCategory,
    ToolDefinition,
    ToolDescriptor,
    ToolSource,
)
from .spec import MCPServerSpecProvider
from .connection import MCPConnectionPool
from .models import (
    MCPConnectionRef,
    MCPDiscoveryResult,
    MCPExposedTool,
    MCPToolInfo,
    MCPRuntimePolicy,
)


def final_tool_name(
    server_id: str,
    tool_name: str,
    tool_prefix: str | bool | None,
) -> str:
    if tool_prefix is False:
        return tool_name
    prefix = server_id if tool_prefix is True or tool_prefix is None else tool_prefix
    return f"{prefix}.{tool_name}"


def filter_tool_names(
    names: Iterable[str],
    enabled_tools: tuple[str, ...] | None,
    disabled_tools: tuple[str, ...],
) -> tuple[str, ...]:
    enabled = set(enabled_tools) if enabled_tools is not None else None
    disabled = set(disabled_tools)
    return tuple(
        name
        for name in names
        if (enabled is None or name in enabled) and name not in disabled
    )


def detect_mcp_conflicts(
    final_names_by_server: Mapping[str, Iterable[str]],
) -> None:
    seen: dict[str, str] = {}
    for server_id, names in final_names_by_server.items():
        for name in names:
            owner = seen.get(name)
            if owner is not None:
                raise AgentFeatureConflictError(
                    f"MCP tool name {name!r} exposed by both "
                    f"{owner} and {server_id}"
                )
            seen[name] = server_id


@dataclass
class MCPToolProvider:
    """AgentFeatureProvider for MCP servers. Both the spec provider and the
    connection manager are injectable so tests can supply fakes; production
    wiring passes a real MCPServerSpecIndex + MCPConnectionPool.

    A connection manager is REQUIRED: without one the provider cannot enumerate
    live tools, so governance (filtering, conflict detection, max_tools,
    ToolExposurePolicy, ToolPolicyResolver) would all be silently skipped. A
    missing manager is a configuration error and fails at construction -- it
    must never surface as a verified-but-empty discovery result."""

    specs: MCPServerSpecProvider
    connections: MCPConnectionPool
    policy: MCPRuntimePolicy = field(default_factory=MCPRuntimePolicy)
    supported_kinds: "ClassVar[tuple[str, ...]]" = ("mcp",)

    def __post_init__(self) -> None:
        if self.connections is None:
            raise RuntimeInitializationError(
                "MCPToolProvider requires an MCPConnectionPool; a server declared "
                "without a connection manager cannot verify tool governance"
            )

    async def resolve(
        self,
        ref: AgentFeatureRef,
        context: AgentFeatureContext,
    ) -> AgentContribution:
        ids = await self._target_ids(ref, context)
        definitions: list[ToolDefinition] = []
        final_names_by_server: "dict[str, tuple[str, ...]]" = {}
        runtime_policy = self.policy
        max_per_server = runtime_policy.max_tools_per_server
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
            # ToolPolicyResolver all need the real tool set to do their job;
            # proceeding on an empty/unknown set would silently skip every one
            # of them.
            discovery_mode = runtime_policy.discovery_mode
            if not discovery.verified and discovery_mode == "strict":
                raise AgentAssemblyError(
                    f"mcp server {server_id!r}: strict discovery mode cannot verify "
                    f"tool governance without live enumeration (list_tools "
                    f"returned no tools) -- set discovery_mode='best_effort' to opt out"
                )
            filtered = filter_tool_names(raw, spec.enabled_tools, spec.disabled_tools)
            final = tuple(
                final_tool_name(spec.id, n, spec.tool_prefix) for n in filtered
            )
            if len(final) > max_per_server:
                raise AgentFeatureConflictError(
                    f"mcp server {server_id!r} exposes {len(final)} tools "
                    f"(max_tools_per_server={max_per_server})"
                )
            final_names_by_server[server_id] = final
            if not hasattr(self.connections, "call_tool"):
                raise AgentAssemblyError(
                    "MCP connection manager must implement call_tool(connection_ref=...)"
                )
            info_by_name = {info.name: info for info in raw_infos}
            exposed_tools = [
                MCPExposedTool(
                    server_id=server_id,
                    raw_name=r,
                    exposed_name=e,
                    parameters_json_schema=info_by_name.get(
                        r, MCPToolInfo(r)
                    ).parameters_json_schema,
                    description=info_by_name.get(r, MCPToolInfo(r)).description,
                    read_only=info_by_name.get(r, MCPToolInfo(r)).read_only,
                    metadata=info_by_name.get(r, MCPToolInfo(r)).metadata,
                )
                for r, e in zip(filtered, final)
            ]
            descs = tuple(
                ToolDescriptor(
                    name=et.exposed_name,
                    source=ToolSource.MCP,
                    category=(
                        ToolCategory.NETWORK_READ
                        if et.read_only
                        else ToolCategory.NETWORK_WRITE
                    ),
                    risk=RiskLevel.MEDIUM if et.read_only else RiskLevel.HIGH,
                    side_effect=(
                        SideEffectKind.READ_ONLY
                        if et.read_only
                        else SideEffectKind.NAMESPACE_MUTATING
                    ),
                    feature=ref,
                    metadata={"raw_name": et.raw_name, **dict(et.metadata)},
                )
                for et in exposed_tools
            )
            server_definitions = tuple(
                ToolDefinition(
                    descriptor=d,
                    handler=self._handler(et, discovery.connection_ref),
                    input_schema=et.parameters_json_schema,
                    description=et.description,
                )
                for d, et in zip(descs, exposed_tools)
            )
            definitions.extend(server_definitions)
        detect_mcp_conflicts(final_names_by_server)
        return AgentContribution(tools=tuple(definitions))

    async def _target_ids(
        self, ref: AgentFeatureRef, context: AgentFeatureContext
    ) -> "tuple[str, ...]":
        if ref.name == "*":
            # Wildcard exposes EVERY server's tools -- deployment-level opt-in
            # only. The Runtime gate is authoritative; a tool ref's own config
            # must NOT be able to self-grant the wildcard.
            if not self.policy.allow_wildcard:
                raise AgentAssemblyError(
                    f"agent {context.agent_id}: mcp:* requires allow_wildcard=True"
                )
            return await self.specs.list_ids()
        return (ref.name,)

    async def _spec(self, server_id: str):
        try:
            return await self.specs.get(server_id)
        except (KeyError, LookupError):
            raise MCPServerNotFoundError(f"mcp server not found: {server_id}") from None

    async def _discover(self, spec) -> MCPDiscoveryResult:
        result_getter = getattr(self.connections, "list_tools_result", None)
        if result_getter is None:
            from ...errors import MCPDiscoveryUnsupportedError

            return MCPDiscoveryResult(
                (),
                False,
                MCPDiscoveryUnsupportedError(
                    "MCP manager must implement list_tools_result"
                ),
            )
        return await result_getter(spec)

    def _handler(
        self, exposed: MCPExposedTool, connection_ref: MCPConnectionRef | None
    ):
        async def call(**arguments: Any) -> Any:
            if connection_ref is None:
                raise AgentAssemblyError(
                    "MCP discovery did not return a connection reference"
                )
            return await self.connections.call_tool(
                connection_ref=connection_ref,
                tool_name=exposed.raw_name,
                arguments=arguments,
            )

        return call
