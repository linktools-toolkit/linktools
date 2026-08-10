#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP capability resolution and execution-scoped toolset materialization."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities import Toolset as PydanticToolset
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


class MCPToolProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve_ref(self, server_id: str, revision: "int | None" = None) -> MCPServerSpec: ...
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
    source: MCPToolProvider
    fingerprint: str
    inherit_to_subagents: bool = False

    @property
    def provider(self) -> str:
        return "mcp"

    async def materialize(self, context: CapabilityRuntimeContext) -> "tuple[PydanticToolset[None], ...]":
        if not self.servers:
            return ()
        toolsets = await self.source.toolsets(
            self.servers,
            principal=context.principal,
            execution=context.execution,
        )
        _validate_mcp_toolsets(toolsets)
        return tuple(PydanticToolset(toolset) for toolset in toolsets)


class MCPServerCapabilityResolver:
    provider = "mcp"

    def __init__(self, provider: MCPToolProvider) -> None:
        self._source = provider
        self._fingerprint = provider.manifest()
        validate_fingerprint(self._fingerprint)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def resolve(self, refs: "tuple[AgentCapabilityRef, ...]") -> MCPServerCapabilityBinding:
        resolutions: list[CapabilityRefResolution] = []
        servers: list[MCPServerSpec] = []
        for ref in refs:
            try:
                server = self._source.resolve_ref(ref.id, ref.revision)
            except (KeyError, LookupError):
                server = None
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND:
                    server = None
                else:
                    raise
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
            servers.append(server)
        binding_resolutions = tuple(resolutions)
        return MCPServerCapabilityBinding(
            binding_resolutions,
            tuple(servers),
            self._source,
            canonical_sha256(
                {
                    "provider": self.provider,
                    "resolver_fingerprint": self.fingerprint,
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
    "MCPCallRequest", "MCPConnectionPool", "MCPServerCapabilityBinding", "MCPServerCapabilityResolver",
    "MCPToolProvider", "validate_mcp_response",
]
