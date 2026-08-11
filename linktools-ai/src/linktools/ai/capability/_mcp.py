#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP capability binding and execution-scoped toolset materialization."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.toolsets import AbstractToolset

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
    CapabilityFeature,
    CapabilityRefResolution,
    CapabilityRuntimeContext,
    validate_fingerprint,
)


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

    @property
    def features(self) -> "frozenset[CapabilityFeature]":
        return frozenset({CapabilityFeature.TOOLS})

    async def materialize(self, context: CapabilityRuntimeContext) -> "tuple[MCP[None], ...]":
        if not self.servers:
            return ()
        toolsets = await self.runtime.toolsets(
            self.servers,
            principal=context.principal,
            execution=context.execution,
        )
        _validate_mcp_toolsets(toolsets)
        return tuple(MCP(local=toolset, id=toolset.id) for toolset in toolsets)


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
    for ref, server in zip(refs, servers):
        if server is None:
            if ref.required:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
            resolutions.append(CapabilityRefResolution(ref.id, ref.revision, None, False, "unresolved", None))
            continue
        if server.id != ref.id or (ref.revision is not None and server.revision != ref.revision):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
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


def _validate_mcp_toolsets(toolsets: Sequence[AbstractToolset[None]]) -> None:
    if not toolsets:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    ids = [toolset.id for toolset in toolsets]
    if any(not isinstance(toolset_id, str) or not toolset_id.strip() for toolset_id in ids):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "MCP toolset id is required")
    if len(set(ids)) != len(ids):
        raise AIError(ErrorCode.STORAGE_CONFLICT, "MCP toolset ids must be unique per run")


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
    "MCPCallRequest", "MCPConnectionPool", "MCPRuntimeProvider", "MCPServerCapabilityBinding",
    "bind_mcp_capability", "validate_mcp_response",
]
