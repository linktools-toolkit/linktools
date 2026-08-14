#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP capability provider and execution-scoped runtime contract."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.toolsets import AbstractToolset

from ..asset import AssetRef, AssetRepository
from ..core import Principal, ResourceRef, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef, MCPServerSpec
from ._contract import CapabilityBinding, CapabilityMaterializationContext, CapabilityRefResolution, validate_fingerprint


class MCPRuntime(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def toolsets(
        self,
        servers: Sequence[MCPServerSpec],
        *,
        principal: Principal,
        execution: ResourceRef,
        execution_root: str,
    ) -> "tuple[AbstractToolset[None], ...]": ...


def mcp_server_namespace(server_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", server_id)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return value


def mcp_server_selector(server_id: str) -> str:
    return f"mcp__{mcp_server_namespace(server_id)}"


def mcp_tool_name(server_id: str, tool_name: str) -> str:
    if not tool_name or tool_name != tool_name.strip() or "*" in tool_name:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return f"{mcp_server_selector(server_id)}__{tool_name}"


@dataclass(frozen=True, slots=True)
class MCPServerCapabilityBinding:
    resolutions: "tuple[CapabilityRefResolution, ...]"
    servers: "tuple[MCPServerSpec, ...]"
    runtime: MCPRuntime
    fingerprint: str

    @property
    def id(self) -> str:
        return "mcp"

    @property
    def provider(self) -> str:
        return "mcp"

    async def materialize(self, context: CapabilityMaterializationContext) -> "tuple[MCP[None], ...]":
        selected = tuple(server for server in self.servers if _server_selected(server.id, context.allow_tools))
        if not selected:
            return ()
        toolsets = await self.runtime.toolsets(
            selected,
            principal=context.principal,
            execution=context.execution,
            execution_root=context.execution_root,
        )
        if len(toolsets) != len(selected) or len({toolset.id for toolset in toolsets}) != len(toolsets):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(MCP(local=toolset.prefixed(f"mcp__{mcp_server_namespace(server.id)}_"), id=toolset.id) for server, toolset in zip(selected, toolsets))


class MCPCapabilityProvider:
    provider = "mcp"

    def __init__(self, runtime: MCPRuntime) -> None:
        self._runtime = runtime

    async def bind(self, refs: "tuple[AgentCapabilityRef, ...]", *, assets: AssetRepository) -> CapabilityBinding:
        values: list[MCPServerSpec | None] = []
        for ref in refs:
            try:
                resolved = await assets.resolve(AssetRef("mcp", ref.id))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND and not ref.required:
                    values.append(None)
                    continue
                raise
            if not isinstance(resolved.spec, MCPServerSpec):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            values.append(resolved.spec)
        return bind_mcp_capability(refs, values, self._runtime)


def bind_mcp_capability(
    refs: Sequence[AgentCapabilityRef],
    servers: Sequence[MCPServerSpec | None],
    runtime: MCPRuntime,
) -> MCPServerCapabilityBinding:
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
        if server.id != ref.id or ref.revision is not None and server.revision != ref.revision:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        namespace = mcp_server_namespace(server.id)
        if namespace in namespaces:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        namespaces.add(namespace)
        resolved.append(server)
        resolutions.append(CapabilityRefResolution(ref.id, ref.revision, server.revision, ref.required, "resolved", canonical_sha256({"id": server.id, "revision": server.revision, "command": server.command, "args": list(server.args)})))
    return MCPServerCapabilityBinding(
        tuple(resolutions),
        tuple(resolved),
        runtime,
        canonical_sha256({"provider": "mcp", "runtime_fingerprint": runtime.fingerprint, "resolutions": [_resolution_payload(item) for item in resolutions]}),
    )


def _server_selected(server_id: str, allow_tools: "tuple[str, ...]") -> bool:
    selector = mcp_server_selector(server_id)
    return "*" in allow_tools or selector in allow_tools or f"{selector}__*" in allow_tools or any(item.startswith(selector + "__") for item in allow_tools)


def _resolution_payload(resolution: CapabilityRefResolution) -> dict[str, object]:
    return {"id": resolution.id, "requested_revision": resolution.requested_revision, "resolved_revision": resolution.resolved_revision, "required": resolution.required, "status": resolution.status, "fingerprint": resolution.fingerprint}


__all__ = ["MCPRuntime", "MCPCapabilityProvider", "MCPServerCapabilityBinding", "bind_mcp_capability", "mcp_server_namespace", "mcp_server_selector", "mcp_tool_name"]
