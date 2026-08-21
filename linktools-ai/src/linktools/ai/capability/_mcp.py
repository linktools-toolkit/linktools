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
from ..spec import MCPServerSpec
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
            execution_root=str(context.execution_root),
        )
        if len(toolsets) != len(selected) or len({toolset.id for toolset in toolsets}) != len(toolsets):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(
            MCP(
                local=toolset.prefixed(f"mcp__{mcp_server_namespace(server.id)}__"),
                id=toolset.id,
            )
            for server, toolset in zip(selected, toolsets)
        )


class MCPCapabilityProvider:
    provider = "mcp"
    value_type = MCPServerSpec

    def __init__(self, runtime: MCPRuntime) -> None:
        self._runtime = runtime

    async def bind(
        self,
        refs: "tuple[AssetRef, ...]",
        *,
        assets: AssetRepository,
    ) -> CapabilityBinding:
        values: list[MCPServerSpec] = []
        for ref in refs:
            resolved = await assets.resolve(ref)
            if type(resolved.spec) is not MCPServerSpec:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            values.append(resolved.spec)
        return bind_mcp_capability(refs, values, self._runtime)

    def select(
        self,
        binding: CapabilityBinding,
        refs: "tuple[AssetRef, ...]",
    ) -> CapabilityBinding:
        if not isinstance(binding, MCPServerCapabilityBinding) or binding.runtime is not self._runtime or not refs:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        values = {resolution.ref: server for resolution, server in zip(binding.resolutions, binding.servers)}
        ordered = tuple(sorted(refs, key=lambda ref: (ref.kind, ref.id)))
        if len(ordered) != len(set(ordered)) or any(ref not in values for ref in ordered):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return bind_mcp_capability(ordered, tuple(values[ref] for ref in ordered), self._runtime)


def bind_mcp_capability(
    refs: Sequence[AssetRef],
    servers: Sequence[MCPServerSpec],
    runtime: MCPRuntime,
) -> MCPServerCapabilityBinding:
    if len(refs) != len(servers):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(runtime.fingerprint)
    resolutions: list[CapabilityRefResolution] = []
    resolved: list[MCPServerSpec] = []
    namespaces: set[str] = set()
    ids: set[str] = set()
    for ref, server in zip(refs, servers):
        if server.id != ref.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if server.id in ids:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        ids.add(server.id)
        namespace = mcp_server_namespace(server.id)
        if namespace in namespaces:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        namespaces.add(namespace)
        resolved.append(server)
        resolutions.append(
            CapabilityRefResolution(
                ref,
                server.revision,
                canonical_sha256(
                    {
                        "id": server.id,
                        "revision": server.revision,
                        "command": server.command,
                        "args": list(server.args),
                    }
                ),
            )
        )
    return MCPServerCapabilityBinding(
        tuple(resolutions),
        tuple(resolved),
        runtime,
        canonical_sha256(
            {
                "provider": "mcp",
                "runtime_fingerprint": runtime.fingerprint,
                "resolutions": [_resolution_payload(item) for item in resolutions],
            }
        ),
    )


def _server_selected(server_id: str, allow_tools: "tuple[str, ...]") -> bool:
    selector = mcp_server_selector(server_id)
    return (
        "*" in allow_tools
        or selector in allow_tools
        or f"{selector}__*" in allow_tools
        or any(item.startswith(selector + "__") for item in allow_tools)
    )


def _resolution_payload(resolution: CapabilityRefResolution) -> dict[str, object]:
    return {
        "kind": resolution.ref.kind,
        "id": resolution.ref.id,
        "resolved_revision": resolution.resolved_revision,
        "fingerprint": resolution.fingerprint,
    }


__all__ = [
    "MCPRuntime",
    "MCPCapabilityProvider",
    "MCPServerCapabilityBinding",
    "bind_mcp_capability",
    "mcp_server_namespace",
    "mcp_server_selector",
    "mcp_tool_name",
]
