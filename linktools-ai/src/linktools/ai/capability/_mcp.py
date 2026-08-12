#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP capability binding and execution-scoped toolset materialization."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.toolsets import AbstractToolset

from ..asset import AssetDiscoveryStatus, AssetRef, AssetRepository
from ..core import (
    JsonValue,
    Principal,
    ResourceRef,
    canonical_json_bytes,
    canonical_sha256,
)
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef, MCPServerSpec
from ._contract import (
    CapabilityBinding,
    CapabilityProvider,
    CapabilityRefResolution,
    CapabilityRuntimeContext,
    validate_fingerprint,
)
from ._skill import AssetSkillProvider


def mcp_server_namespace(server_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", server_id)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value or "__" in value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return value


def mcp_server_selector(server_id: str) -> str:
    return f"mcp__{mcp_server_namespace(server_id)}"


def mcp_tool_name(server_id: str, tool_name: str) -> str:
    if not tool_name or tool_name != tool_name.strip() or "*" in tool_name:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return f"{mcp_server_selector(server_id)}__{tool_name}"


@dataclass(frozen=True, slots=True)
class MCPCallRequest:
    principal: Principal
    execution: ResourceRef
    operation_id: str
    server_id: str
    tool_name: str
    arguments: "Mapping[str, JsonValue]"
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


class MCPRuntimeProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def connect(self, server_id: str) -> MCPConnectionPool: ...

    async def toolsets(
        self,
        servers: Sequence[MCPServerSpec],
        *,
        principal: Principal,
        execution: ResourceRef,
    ) -> "tuple[AbstractToolset[None], ...]": ...


@dataclass(frozen=True, slots=True)
class MCPServerCapabilityBinding:
    resolutions: "tuple[CapabilityRefResolution, ...]"
    servers: "tuple[MCPServerSpec, ...]"
    runtime: MCPRuntimeProvider
    fingerprint: str
    inherit_to_subagents: bool = False

    @property
    def id(self) -> str:
        return "mcp"

    @property
    def provider(self) -> str:
        return "mcp"

    async def materialize(self, context: CapabilityRuntimeContext) -> "tuple[MCP[None], ...]":
        selected = tuple(server for server in self.servers if _server_selected(server.id, context.allow_tools))
        if not selected:
            return ()
        toolsets = await self.runtime.toolsets(
            selected,
            principal=context.principal,
            execution=context.execution,
        )
        _validate_mcp_toolsets(toolsets, len(selected))
        return tuple(
            MCP(local=toolset.prefixed(f"mcp__{mcp_server_namespace(server.id)}_"), id=toolset.id)
            for server, toolset in zip(selected, toolsets)
        )


class AssetMCPProvider:
    """Resolve MCP server declarations through the logical AssetRepository."""

    provider = "mcp"

    def __init__(self, assets: AssetRepository, runtime: MCPRuntimeProvider) -> None:
        self._assets = assets
        self._runtime = runtime

    async def bootstrap_refs(self) -> "tuple[AgentCapabilityRef, ...]":
        entries = await _list_asset_entries(self._assets, "mcp")
        refs: list[AgentCapabilityRef] = []
        namespaces: set[str] = set()
        for entry in entries:
            if entry.status is AssetDiscoveryStatus.CONFLICT:
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
            namespace = mcp_server_namespace(entry.ref.id)
            if namespace in namespaces:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            namespaces.add(namespace)
            refs.append(AgentCapabilityRef(self.provider, entry.ref.id, required=False))
        return tuple(refs)

    async def bind(self, refs: "tuple[AgentCapabilityRef, ...]") -> CapabilityBinding:
        values: list[MCPServerSpec | None] = []
        for ref in refs:
            try:
                resolved = await self._assets.resolve(AssetRef("mcp", ref.id))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND and not ref.required:
                    values.append(None)
                    continue
                raise
            if not isinstance(resolved.spec, MCPServerSpec):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            values.append(resolved.spec)
        return bind_mcp_capability(refs, values, self._runtime)


def build_builtin_capability_providers(
    assets: AssetRepository,
    *,
    mcp_runtime: "MCPRuntimeProvider | None",
) -> "tuple[CapabilityProvider, ...]":
    providers: list[CapabilityProvider] = [AssetSkillProvider(assets)]
    if mcp_runtime is not None:
        providers.append(AssetMCPProvider(assets, mcp_runtime))
    return tuple(providers)


def bind_mcp_capability(
    refs: "Sequence[AgentCapabilityRef]",
    servers: "Sequence[MCPServerSpec | None]",
    runtime: MCPRuntimeProvider,
) -> MCPServerCapabilityBinding:
    """Compile resolved MCP declarations with their execution provider."""
    if len(refs) != len(servers):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(runtime.fingerprint)
    resolutions: list[CapabilityRefResolution] = []
    resolved: list[MCPServerSpec] = []
    namespaces: set[str] = set()
    for ref, server in zip(refs, servers):
        if server is None:
            if ref.required:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
            resolutions.append(CapabilityRefResolution(ref.id, ref.revision, None, False, "unresolved", None))
            continue
        if server.id != ref.id or (ref.revision is not None and server.revision != ref.revision):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        namespace = mcp_server_namespace(server.id)
        if namespace in namespaces:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        namespaces.add(namespace)
        fingerprint = canonical_sha256(
            {
                "id": server.id,
                "revision": server.revision,
                "command": server.command,
                "args": list(server.args),
            }
        )
        resolutions.append(CapabilityRefResolution(ref.id, ref.revision, server.revision, ref.required, "resolved", fingerprint))
        resolved.append(server)
    binding_resolutions = tuple(resolutions)
    return MCPServerCapabilityBinding(
        binding_resolutions,
        tuple(resolved),
        runtime,
        canonical_sha256(
            {
                "provider": "mcp",
                "runtime_fingerprint": runtime.fingerprint,
                "inherit_to_subagents": False,
                "configs": [dict(ref.config) for ref in refs],
                "resolutions": [_resolution_payload(item) for item in binding_resolutions],
            }
        ),
    )


def _validate_mcp_toolsets(toolsets: Sequence[AbstractToolset[None]], expected: int | None = None) -> None:
    if not isinstance(toolsets, tuple):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if expected is not None and len(toolsets) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        ids = [toolset.id for toolset in toolsets]
    except (AttributeError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if any(not isinstance(toolset_id, str) or not toolset_id.strip() for toolset_id in ids):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "MCP toolset id is required")
    if len(set(ids)) != len(ids):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "MCP toolset ids must be unique per run")


def _server_selected(server_id: str, allow_tools: "tuple[str, ...]") -> bool:
    selector = mcp_server_selector(server_id)
    return "*" in allow_tools or selector in allow_tools or f"{selector}__*" in allow_tools or any(
        item.startswith(selector + "__") for item in allow_tools
    )


async def _list_asset_entries(assets: AssetRepository, kind: str) -> "tuple[object, ...]":
    cursor = None
    entries = []
    while True:
        page = await assets.list(kind=kind, cursor=cursor)
        entries.extend(page.items)
        if page.next_cursor is None:
            return tuple(entries)
        cursor = page.next_cursor


def _resolution_payload(resolution: CapabilityRefResolution) -> "dict[str, object]":
    return {
        "id": resolution.id,
        "requested_revision": resolution.requested_revision,
        "resolved_revision": resolution.resolved_revision,
        "required": resolution.required,
        "status": resolution.status,
        "fingerprint": resolution.fingerprint,
    }


__all__ = [
    "AssetMCPProvider", "MCPCallRequest", "MCPConnectionPool", "MCPRuntimeProvider", "MCPServerCapabilityBinding",
    "bind_mcp_capability", "mcp_server_namespace", "mcp_server_selector", "mcp_tool_name", "validate_mcp_response",
]
