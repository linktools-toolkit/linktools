#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize compiler-selected stdio MCP as Runtime capabilities."""

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import (
    AbstractToolset,
    ToolsetTool,
    WrapperToolset,
)
from pydantic_ai.tools import RunContext as PydanticRunContext
from mcp.types import Tool as MCPTool

from ..capability import (
    AgentContext,
    tool_semantic_metadata,
    validate_resource_path,
    validate_resource_tree,
)
from ..asset import AssetStore
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import (
    MCPServerSpec,
    mcp_tool_selector,
    parse_mcp_tool_selector,
)
from ..storage import ObjectRef, ObjectStore
from ..workspace import (
    SandboxResource,
    SandboxResourcePath,
    SandboxStdioProcess,
    StdioSandbox,
    StdioSandboxSession,
    Workspace,
)
from ._tool import ToolOperationBridge
from ._tool_boundary import (
    RuntimeToolBoundaryToolset,
    managed_tool_descriptor_from_metadata,
)
from ._tool_metrics import _ToolMetricContext
from ._mcp_transport import _SandboxMCPTransport

_logger = environ.get_logger("ai.runtime.mcp")
_MCP_TOOL_METADATA = tool_semantic_metadata(
    effect="non_replay_safe",
    tool_class="mcp",
)


def _mcp_tool_metadata(base: Mapping[str, object] | None) -> dict[str, object]:
    metadata = tool_semantic_metadata(base=base)
    metadata.update(_MCP_TOOL_METADATA)
    return metadata


@dataclass(frozen=True, slots=True)
class _FrozenMCPResources:
    snapshot: ObjectRef | None
    resource_semantic_digest: str | None
    execution_policy: Mapping[str, JsonValue]


@dataclass(slots=True)
class _MCPResourceProjection:
    server_id: str
    args: tuple[str | SandboxResourcePath, ...]
    resources: tuple[SandboxResource, ...]
    directory: "tempfile.TemporaryDirectory[str] | None"

    async def close(self) -> None:
        directory = self.directory
        if directory is not None:
            await _cleanup_resource_directory(directory)
            self.directory = None


class _MCPModelToolset(WrapperToolset[object]):
    """Apply stable model names while routing calls to original MCP names."""

    def __init__(
        self,
        wrapped: AbstractToolset[object],
        server_id: str,
        allowed_tools: frozenset[str] | None,
        required_tools: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(wrapped)
        self._server_id = server_id
        self._allowed_tools = allowed_tools
        self._required_tools = required_tools
        self._published: dict[str, str] = {}

    async def get_tools(
        self,
        ctx: PydanticRunContext[object],
    ) -> dict[str, ToolsetTool[object]]:
        tools = await self.wrapped.get_tools(ctx)
        upstream_names = tuple(tool.tool_def.name for tool in tools.values())
        if len(upstream_names) != len(set(upstream_names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        available = frozenset(upstream_names)
        if (
            self._allowed_tools is not None
            and not self._allowed_tools.issubset(available)
        ) or not self._required_tools.issubset(available):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        result: dict[str, ToolsetTool[object]] = {}
        for tool in tools.values():
            upstream_name = tool.tool_def.name
            if (
                self._allowed_tools is not None
                and upstream_name not in self._allowed_tools
            ):
                continue
            model_name = _model_tool_name(self._server_id, upstream_name)
            previous = self._published.get(model_name)
            if previous is not None and previous != upstream_name:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            if model_name in result:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            self._published[model_name] = upstream_name
            identity = json.dumps(
                {"server_id": self._server_id, "tool_name": upstream_name},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            description = tool.tool_def.description or ""
            description = f"[MCP identity: {identity}]\n{description}"
            result[model_name] = replace(
                tool,
                toolset=self,
                tool_def=replace(
                    tool.tool_def,
                    name=model_name,
                    description=description,
                    metadata=_mcp_tool_metadata(tool.tool_def.metadata),
                ),
            )
        return result

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: PydanticRunContext[object],
        tool: ToolsetTool[object],
    ) -> Any:
        upstream_name = self._published.get(name)
        if upstream_name is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        upstream_tool = replace(
            tool,
            toolset=self.wrapped,
            tool_def=replace(tool.tool_def, name=upstream_name),
        )
        upstream_context = replace(ctx, tool_name=upstream_name)
        return await self.wrapped.call_tool(
            upstream_name,
            tool_args,
            upstream_context,
            upstream_tool,
        )


class _MCPDiscoveryToolset(MCPToolset[object]):
    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        names = tuple(tool.name for tool in tools)
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        return tools


class _MCPRuntimeCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        capability_id: str,
        toolset: AbstractToolset[AgentContext[object]],
        client: object,
    ) -> None:
        self.id = capability_id
        self._toolset = toolset
        self._client: object | None = client

    def get_toolset(self) -> AbstractToolset[AgentContext[object]]:
        return self._toolset

    async def close_resources(self) -> None:
        client = self._client
        if client is not None:
            await cast(Any, client).close()
            self._client = None


async def close_mcp_resources(
    capabilities: Sequence[AbstractCapability[AgentContext[object]]],
) -> None:
    """Release temporary directories used by frozen MCP resources."""
    failure: BaseException | None = None
    for capability in capabilities:
        if isinstance(capability, _MCPRuntimeCapability):
            try:
                await capability.close_resources()
            except BaseException as error:
                if failure is None:
                    failure = error
    if failure is not None:
        _raise_cleanup_failure(failure)


async def close_mcp_projections(
    projections: Mapping[str, _MCPResourceProjection],
) -> None:
    failure: BaseException | None = None
    for projection in projections.values():
        try:
            await projection.close()
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        _raise_cleanup_failure(failure)


def _raise_cleanup_failure(error: BaseException) -> None:
    if isinstance(error, (AIError, asyncio.CancelledError)):
        raise error
    raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error


async def _cleanup_resource_directory(
    directory: tempfile.TemporaryDirectory[str],
) -> None:
    task = asyncio.create_task(asyncio.to_thread(directory.cleanup))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            await asyncio.shield(task)
        except BaseException as cleanup_error:
            raise cancellation from cleanup_error
        raise cancellation


def _raise_primary_after_cleanup(
    primary_error: BaseException,
    cleanup_error: BaseException,
) -> NoReturn:
    if cleanup_error is primary_error:
        raise primary_error
    try:
        _raise_cleanup_failure(cleanup_error)
    except BaseException as typed_cleanup_error:
        raise primary_error from typed_cleanup_error


def validate_frozen_mcp_policy(
    resources: Mapping[str, _FrozenMCPResources],
    workspace: Workspace | None,
) -> None:
    current_policy = _current_execution_policy(workspace)
    if any(
        dict(frozen.execution_policy) != dict(current_policy)
        for frozen in resources.values()
    ):
        raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)


async def prepare_mcp_resource_projections(
    servers: Sequence[MCPServerSpec],
    resources: Mapping[str, _FrozenMCPResources],
    *,
    object_store: ObjectStore | None,
    sandboxed: bool,
) -> dict[str, _MCPResourceProjection]:
    """Verify and materialize each selected server's frozen resource tree."""
    projections: dict[str, _MCPResourceProjection] = {}
    try:
        for server in servers:
            frozen = resources.get(server.id)
            if frozen is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if server.resource_root is None:
                if frozen.snapshot is not None or frozen.resource_semantic_digest is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if any(argument.startswith("resource:") for argument in server.args):
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                projections[server.id] = _MCPResourceProjection(
                    server.id,
                    tuple(server.args),
                    (),
                    None,
                )
                continue
            if object_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            directory = await _materialize_resource_snapshot(
                server,
                frozen,
                object_store,
            )
            try:
                arguments: list[str | SandboxResourcePath] = []
                for argument in server.args:
                    if not argument.startswith("resource:"):
                        arguments.append(argument)
                        continue
                    relative = argument[len("resource:") :]
                    target = _resource_target(directory.name, relative)
                    arguments.append(
                        SandboxResourcePath(server.id, relative)
                        if sandboxed
                        else str(target)
                    )
                projections[server.id] = _MCPResourceProjection(
                    server.id,
                    tuple(arguments),
                    (SandboxResource(server.id, Path(directory.name)),)
                    if sandboxed
                    else (),
                    directory,
                )
            except BaseException as primary_error:
                try:
                    await _cleanup_resource_directory(directory)
                except BaseException as cleanup_error:
                    _raise_primary_after_cleanup(primary_error, cleanup_error)
                raise
    except BaseException as primary_error:
        try:
            await close_mcp_projections(projections)
        except BaseException as cleanup_error:
            _raise_primary_after_cleanup(primary_error, cleanup_error)
        raise
    return projections


async def materialize_mcp_capabilities(
    servers: Sequence[MCPServerSpec],
    selectors: Sequence[str],
    *,
    workspace: Workspace | None,
    sandbox_session: object | None,
    host_cwd: "str | None",
    frozen_resources: Mapping[str, _FrozenMCPResources],
    projections: Mapping[str, _MCPResourceProjection],
    tool_operations: "ToolOperationBridge | None",
    tool_metrics: "_ToolMetricContext | None",
    background_tasks: set[asyncio.Task[object]],
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Materialize only compiler-selected stdio MCP servers."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    policy, required = _selector_policy(selectors)
    descriptor = managed_tool_descriptor_from_metadata(_MCP_TOOL_METADATA)
    values: list[AbstractCapability[AgentContext[object]]] = []
    current_policy = _current_execution_policy(workspace)
    if workspace is not None and not isinstance(
        sandbox_session,
        StdioSandboxSession,
    ):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    if policy and workspace is None and host_cwd is None:
        raise AIError(
            ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
            safe_details={"reason": "mcp_cwd_unavailable"},
        )
    try:
        for server in servers:
            if server.id not in policy:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            frozen = frozen_resources.get(server.id)
            projection = projections.get(server.id)
            if frozen is None or projection is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if dict(frozen.execution_policy) != dict(current_policy):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            allowed = policy[server.id]
            if workspace is None:
                transport = StdioTransport(
                    server.command,
                    list(cast("Sequence[str]", projection.args)),
                    cwd=host_cwd,
                )
            else:
                transport = _SandboxMCPTransport(
                    cast(StdioSandboxSession, sandbox_session),
                    server.command,
                    projection.args,
                    projection.resources,
                )
            client = Client(transport)
            toolset = _MCPDiscoveryToolset(
                client,
                id=f"mcp:{server.id}",
                cache_tools=True,
            )
            mapped = _MCPModelToolset(
                cast("AbstractToolset[object]", toolset),
                server.id,
                allowed,
                required.get(server.id, frozenset()),
            )
            boundary = RuntimeToolBoundaryToolset(
                (
                    cast(
                        "AbstractToolset[AgentContext[object]]",
                        mapped,
                    ),
                ),
                {},
                id=f"linktools.mcp.{server.id}",
                descriptor=descriptor,
                tool_operations=tool_operations,
                tool_metrics=tool_metrics,
                background_tasks=background_tasks,
            )
            values.append(
                _MCPRuntimeCapability(
                    f"linktools.ai.mcp.{server.id}",
                    boundary,
                    client,
                )
            )
            _logger.debug(
                "MCP server materialized: server=%s selected_tools=%s",
                server.id,
                "*" if allowed is None else tuple(sorted(allowed)),
            )
    except BaseException as primary_error:
        try:
            await close_mcp_resources(values)
        except BaseException as cleanup_error:
            _raise_primary_after_cleanup(primary_error, cleanup_error)
        raise
    return tuple(values)


async def _materialize_resource_snapshot(
    server: MCPServerSpec,
    frozen: _FrozenMCPResources,
    resource_objects: ObjectStore,
) -> "tempfile.TemporaryDirectory[str]":
    if frozen.snapshot is None or frozen.resource_semantic_digest is None:
        raise AIError(
            ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"kind": "mcp_resource", "server_id": server.id},
        )
    if frozen.snapshot.store_id != resource_objects.store_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    snapshot = AssetStore.from_snapshot(
        frozen.snapshot,
        object_store=resource_objects,
    )
    await snapshot.initialize()
    directory = tempfile.TemporaryDirectory(
        prefix="linktools-mcp-",
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
        files: list[dict[str, str]] = []
        for relative, info in values.items():
            target = _resource_target(directory.name, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            body = await snapshot.get(info.key)
            if body is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if len(body) != info.size or hashlib.sha256(body).hexdigest() != info.etag:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target.write_bytes(body)
            files.append({"path": relative, "sha256": info.etag})
        files.sort(key=lambda item: item["path"])
        actual_digest = canonical_sha256(
            {
                "version": 1,
                "kind": "mcp-resource-semantics",
                "files": files,
            }
        )
        if actual_digest != frozen.resource_semantic_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for argument in server.args:
            if argument.startswith("resource:"):
                relative = argument[len("resource:") :]
                if relative not in values:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        await snapshot.close()
        return directory
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        try:
            await _cleanup_resource_directory(directory)
        except BaseException as error:
            cleanup_error = error
        try:
            await snapshot.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            _raise_primary_after_cleanup(primary_error, cleanup_error)
        raise


def _resource_target(root: str, relative: str) -> Path:
    validate_resource_path(relative)
    target = os.path.abspath(os.path.join(root, *relative.split("/")))
    if not target.startswith(os.path.abspath(root) + os.sep):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return Path(target)


def _selector_policy(
    selectors: Sequence[str],
) -> "tuple[dict[str, frozenset[str] | None], dict[str, frozenset[str]]]":
    result: dict[str, set[str] | None] = {}
    required: dict[str, set[str]] = {}
    for selector in selectors:
        parsed = parse_mcp_tool_selector(selector)
        if parsed is None:
            continue
        server_id, tool = parsed
        if tool is None:
            result[server_id] = None
            continue
        required.setdefault(server_id, set()).add(tool)
        current = result.get(server_id)
        if current is None and server_id in result:
            continue
        if current is None:
            current = set()
            result[server_id] = current
        current.add(tool)
    return (
        {
            server_id: None if tools is None else frozenset(tools)
            for server_id, tools in result.items()
        },
        {
            server_id: frozenset(tools)
            for server_id, tools in required.items()
        },
    )


def _model_tool_name(server_id: str, tool_name: str) -> str:
    mcp_tool_selector(server_id, tool_name)
    server_token = canonical_sha256(
        {
            "version": 1,
            "kind": "mcp-server-name",
            "server_id": server_id,
        }
    )[:24]
    tool_token = canonical_sha256(
        {
            "version": 1,
            "kind": "mcp-tool-name",
            "server_id": server_id,
            "tool_name": tool_name,
        }
    )[:24]
    return f"mcp__{server_token}__{tool_token}"


def _current_execution_policy(
    workspace: Workspace | None,
) -> Mapping[str, JsonValue]:
    if workspace is None:
        return {"version": 1, "boundary": "host-stdio"}
    backend = workspace.sandbox
    if not isinstance(backend, StdioSandbox):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return backend.stdio_execution_policy()


__all__ = []
