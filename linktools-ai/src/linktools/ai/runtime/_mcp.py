#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize compiler-selected stdio MCP as Runtime capabilities."""

import asyncio
import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import (
    AbstractToolset,
    ToolsetTool,
    WrapperToolset,
)
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..capability import AgentContext, mcp_server_namespace, tool_semantic_metadata
from ..asset import AssetStore
from ..core import Principal, ResourceRef
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, parse_mcp_tool_selector
from ..storage import ObjectStore
from ._tool import ToolOperationBridge
from ._mcp_resources import validate_resource_path, validate_resource_tree
from ._tool_boundary import (
    RuntimeToolBoundaryToolset,
    managed_tool_descriptor_from_metadata,
)
from ._tool_metrics import _ToolMetricContext

_logger = environ.get_logger("ai.runtime.mcp")
_MCP_TOOL_METADATA = tool_semantic_metadata(
    effect="non_replay_safe",
    tool_class="mcp",
)


def _mcp_tool_metadata(base: Mapping[str, object] | None) -> dict[str, object]:
    metadata = tool_semantic_metadata(base=base)
    metadata.update(_MCP_TOOL_METADATA)
    return metadata


class _MCPSemanticToolset(WrapperToolset[object]):
    """Attach MCP effect and class semantics at the materialization boundary."""

    async def get_tools(
        self,
        ctx: PydanticRunContext[object],
    ) -> dict[str, ToolsetTool[object]]:
        tools = await self.wrapped.get_tools(ctx)
        for tool in tools.values():
            tool.tool_def = replace(
                tool.tool_def,
                metadata=_mcp_tool_metadata(tool.tool_def.metadata),
            )
        return tools


class _MCPRuntimeCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        capability_id: str,
        toolset: AbstractToolset[AgentContext[object]],
        resource_directory: "tempfile.TemporaryDirectory[str] | None" = None,
    ) -> None:
        self.id = capability_id
        self._toolset = toolset
        self._resource_directory = resource_directory

    def get_toolset(self) -> AbstractToolset[AgentContext[object]]:
        return self._toolset

    async def close_resources(self) -> None:
        directory = self._resource_directory
        self._resource_directory = None
        if directory is not None:
            await asyncio.to_thread(directory.cleanup)


async def close_mcp_resources(
    capabilities: Sequence[AbstractCapability[AgentContext[object]]],
) -> None:
    """Release temporary directories used by frozen MCP resources."""
    for capability in capabilities:
        if isinstance(capability, _MCPRuntimeCapability):
            await capability.close_resources()


async def materialize_mcp_capabilities(
    servers: Sequence[MCPServerSpec],
    selectors: Sequence[str],
    *,
    principal: Principal,
    execution: ResourceRef,
    execution_root: "str | None",
    tool_operations: "ToolOperationBridge | None",
    tool_metrics: "_ToolMetricContext | None",
    background_tasks: set[asyncio.Task[object]],
    resource_objects: ObjectStore | None = None,
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Materialize only compiler-selected stdio MCP servers."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai.mcp import MCPToolset

    if servers and execution_root is None:
        raise AIError(
            ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
            safe_details={"reason": "mcp_cwd_unavailable"},
        )
    policy = _selector_policy(selectors)
    descriptor = managed_tool_descriptor_from_metadata(_MCP_TOOL_METADATA)
    root = str(Path(cast(str, execution_root)).expanduser().resolve())
    values: list[AbstractCapability[AgentContext[object]]] = []
    seen_namespaces: set[str] = set()
    resource_directory: tempfile.TemporaryDirectory[str] | None = None
    try:
        for server in servers:
            namespace = mcp_server_namespace(server.id)
            if namespace in seen_namespaces:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            seen_namespaces.add(namespace)
            allowed = policy.get(namespace)
            if allowed is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            server_args, resource_directory = await _materialize_server_args(
                server,
                resource_objects=resource_objects,
            )
            client = Client(
                StdioTransport(
                    server.command,
                    server_args,
                    cwd=root,
                )
            )
            toolset = MCPToolset(client, id=f"mcp:{server.id}")
            if allowed:
                names = frozenset(allowed)
                toolset = toolset.filtered(
                    lambda _ctx, tool_def, selected=names: tool_def.name in selected
                )
            prefixed = _MCPSemanticToolset(
                toolset.prefixed(f"mcp__{namespace}__")
            )
            boundary = RuntimeToolBoundaryToolset(
                (
                    cast(
                        "AbstractToolset[AgentContext[object]]",
                        prefixed,
                    ),
                ),
                {},
                id=f"linktools.mcp.{namespace}",
                descriptor=descriptor,
                tool_operations=tool_operations,
                tool_metrics=tool_metrics,
                background_tasks=background_tasks,
            )
            values.append(
                _MCPRuntimeCapability(
                    f"linktools.ai.mcp.{namespace}",
                    boundary,
                    resource_directory,
                )
            )
            resource_directory = None
            _logger.debug(
                "MCP server materialized: server=%s namespace=%s selected_tools=%s",
                server.id,
                namespace,
                tuple(sorted(allowed or ("*",))),
            )
    except BaseException:
        await close_mcp_resources(values)
        if resource_directory is not None:
            await asyncio.to_thread(resource_directory.cleanup)
        raise
    return tuple(values)


async def _materialize_server_args(
    server: MCPServerSpec,
    *,
    resource_objects: ObjectStore | None,
) -> tuple[list[str], "tempfile.TemporaryDirectory[str] | None"]:
    if server.resource_root is None:
        return list(server.args), None
    if server.resource_snapshot is None or resource_objects is None:
        raise AIError(
            ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"kind": "mcp_resource", "server_id": server.id},
        )
    if server.resource_snapshot.store_id != resource_objects.store_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    snapshot = AssetStore.from_snapshot(
        server.resource_snapshot,
        object_store=resource_objects,
    )
    await snapshot.initialize()
    directory = tempfile.TemporaryDirectory(
        prefix=f"linktools-mcp-{server.id}-",
    )
    try:
        infos = await snapshot.metadata_snapshot()
        prefix = f"{server.resource_root.id}/"
        values = {
            info.key.id[len(prefix) :]: info
            for info in infos
            if info.key.kind == server.resource_root.kind
            and info.key.id.startswith(prefix)
        }
        validate_resource_tree(values)
        for relative, info in values.items():
            target = _resource_target(directory.name, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            body = await snapshot.get(info.key)
            if body is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if len(body) != info.size or hashlib.sha256(body).hexdigest() != info.etag:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target.write_bytes(body)
        arguments: list[str] = []
        for argument in server.args:
            if not argument.startswith("resource:"):
                arguments.append(argument)
                continue
            relative = argument[len("resource:") :]
            info = values.get(relative)
            if info is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            target = _resource_target(directory.name, relative)
            arguments.append(str(target))
        await snapshot.close()
        return arguments, directory
    except BaseException:
        directory.cleanup()
        await snapshot.close()
        raise


def _resource_target(root: str, relative: str) -> Path:
    validate_resource_path(relative)
    target = os.path.abspath(os.path.join(root, *relative.split("/")))
    if not target.startswith(os.path.abspath(root) + os.sep):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return Path(target)


def _selector_policy(selectors: Sequence[str]) -> "dict[str, frozenset[str]]":
    result: dict[str, set[str] | None] = {}
    for selector in selectors:
        parsed = parse_mcp_tool_selector(selector)
        if parsed is None:
            continue
        namespace, tool = parsed
        if tool is None:
            result[namespace] = None
            continue
        current = result.get(namespace)
        if current is None and namespace in result:
            continue
        if current is None:
            current = set()
            result[namespace] = current
        current.add(tool)
    return {
        namespace: frozenset() if tools is None else frozenset(tools)
        for namespace, tools in result.items()
    }


__all__ = []
