#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize compiler-selected MCP servers as Runtime capabilities."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Any, NoReturn

import anyio
import httpx
from fastmcp.client.transports import ClientTransport
from fastmcp.exceptions import McpError

from httpx2 import HTTPError as HTTPErrorV2, HTTPStatusError as HTTPStatusErrorV2
from linktools.core import environ
from pydantic import ValidationError
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
    tool_metadata,
    validate_resource_path,
    validate_resource_tree,
)
from ..asset import AssetStoreReader, AssetVersionRef
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import (
    MCPServerSpec,
    mcp_tool_selector,
    parse_mcp_tool_selector,
)
from ..workspace import (
    Sandbox,
    SandboxResource,
    SandboxResourcePath,
    StdioSandbox,
)
from ._tool import ToolOperationBridge
from ._tool_boundary import (
    BoundaryToolset,
    managed_tool_descriptor_from_metadata,
)
from ._tool_metrics import _ToolMetricContext
from ._mcp_transport import _create_mcp_transport

_logger = environ.get_logger("ai.runtime.mcp")
_MCP_TOOL_METADATA = tool_metadata(
    effect_policy="non_replay_safe",
    tool_class="mcp",
)


def _mcp_tool_metadata(base: Mapping[str, object] | None) -> dict[str, object]:
    metadata = tool_metadata(base=base)
    metadata.update(_MCP_TOOL_METADATA)
    return metadata


@dataclass(frozen=True, slots=True)
class _MCPBinding:
    versions: "tuple[AssetVersionRef, ...] | None"
    asset_source_id: "str | None"
    execution_policy: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _MCPProjection:
    server_id: str
    args: tuple[str | SandboxResourcePath, ...]
    resources: tuple[SandboxResource, ...]


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
    def __init__(self, transport: ClientTransport, *, server: MCPServerSpec) -> None:
        super().__init__(
            transport,
            id=f"mcp:{server.id}",
            cache_tools=True,
            init_timeout=server.init_timeout,
            read_timeout=server.read_timeout,
        )
        self._server_id = server.id
        self._transport_kind = server.transport
        self._cleanup_error: BaseException | None = None

    def _details(self, phase: str) -> dict[str, JsonValue]:
        return {
            "server_id": self._server_id,
            "transport": self._transport_kind,
            "phase": phase,
        }

    async def __aenter__(self) -> "_MCPDiscoveryToolset":
        try:
            await super().__aenter__()
        except BaseException as error:
            _raise_connection_failure(error, self._details("connect"))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            return await super().__aexit__(exc_type, exc_value, traceback)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            try:
                _raise_cleanup_failure(error, self._details("close"))
            except BaseException as cleanup_error:
                # Upstream wrapper exits may discard the primary exception.
                # Final owned cleanup still has it and can report both safely.
                if self._cleanup_error is None:
                    self._cleanup_error = cleanup_error
            return None

    async def list_tools(self) -> list[MCPTool]:
        try:
            tools = await super().list_tools()
        except BaseException as error:
            _raise_connection_failure(error, self._details("tools/list"))
        names = tuple(tool.name for tool in tools)
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        return tools

    async def close_resources(self) -> None:
        try:
            await self.client.close()
        except BaseException as error:
            try:
                _raise_cleanup_failure(error, self._details("close"))
            except BaseException as cleanup_error:
                if self._cleanup_error is None:
                    self._cleanup_error = cleanup_error
        if self._cleanup_error is not None:
            raise self._cleanup_error


class _MCPCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        capability_id: str,
        toolset: AbstractToolset[AgentContext[object]],
        owned_toolset: _MCPDiscoveryToolset,
    ) -> None:
        self.id = capability_id
        self._toolset = toolset
        self._owned_toolset: _MCPDiscoveryToolset | None = owned_toolset

    def get_toolset(self) -> AbstractToolset[AgentContext[object]]:
        return self._toolset

    async def close_resources(self) -> None:
        toolset = self._owned_toolset
        if toolset is not None:
            await toolset.close_resources()
            self._owned_toolset = None


async def close_mcp_resources(
    capabilities: Sequence[AbstractCapability[AgentContext[object]]],
) -> None:
    """Finish owned MCP cleanup before propagating caller cancellation."""
    async def close_owned() -> BaseException | None:
        failure: BaseException | None = None
        for capability in capabilities:
            if isinstance(capability, _MCPCapability):
                try:
                    await capability.close_resources()
                except BaseException as error:
                    if failure is None:
                        failure = error
        return failure

    task = asyncio.create_task(close_owned())
    cancelled: asyncio.CancelledError | None = None
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
    failure = task.result()
    if cancelled is not None:
        if failure is not None:
            _raise_primary_after_cleanup(cancelled, failure)
        raise cancelled
    if failure is not None:
        _raise_cleanup_failure(failure)


def _raise_cleanup_failure(
    error: BaseException,
    details: Mapping[str, JsonValue] | None = None,
) -> NoReturn:
    if isinstance(error, (AIError, asyncio.CancelledError)):
        raise error
    raise AIError(ErrorCode.MCP_CLEANUP_FAILED, safe_details=details) from None


def _connection_errors(error: BaseException) -> tuple[BaseException, ...] | None:
    if isinstance(error, (AIError, asyncio.CancelledError)):
        raise error
    nested = getattr(error, "exceptions", None)
    if isinstance(nested, tuple):
        values: list[BaseException] = []
        for child in nested:
            result = _connection_errors(child)
            if result is None:
                return None
            values.extend(result)
        return tuple(values) if values else None
    if isinstance(error, (
        OSError, TimeoutError, httpx.HTTPError, HTTPErrorV2, McpError, ValidationError,
        anyio.EndOfStream, anyio.ClosedResourceError, anyio.BrokenResourceError,
    )):
        return (error,)
    if type(error) is RuntimeError:
        if error.__cause__ is not None:
            return _connection_errors(error.__cause__)
        # SDK negotiation failures have no public error type, and transport
        # exception-group unwrapping can remove their original timeout cause.
        message = str(error)
        if message in {
            "Failed to initialize server session",
            "Server session was closed unexpectedly",
        } or message.startswith("Unsupported protocol version from the server:"):
            return (error,)
    return None


def _raise_connection_failure(
    error: BaseException,
    details: Mapping[str, JsonValue],
) -> NoReturn:
    failures = _connection_errors(error)
    if failures is None:
        raise error
    safe_details = dict(details)
    for failure in failures:
        if isinstance(failure, (httpx.HTTPStatusError, HTTPStatusErrorV2)):
            safe_details["status_code"] = failure.response.status_code
            break
    raise AIError(ErrorCode.MCP_CONNECTION_FAILED, safe_details=safe_details) from None


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


async def prepare_mcp_projections(
    servers: Sequence[MCPServerSpec],
    bindings: Mapping[str, _MCPBinding],
    *,
    asset_readers: Mapping[str, AssetStoreReader],
    sandboxed: bool,
) -> dict[str, _MCPProjection]:
    """Verify selected Asset versions and expose existing local resource files."""
    projections: dict[str, _MCPProjection] = {}
    for server in servers:
        if server.transport != "stdio":
            continue
        binding = bindings.get(server.id)
        if binding is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if server.resource is None:
            if binding.versions is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            projections[server.id] = _MCPProjection(
                server.id,
                tuple(server.args),
                (),
            )
            continue
        if binding.asset_source_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        reader = asset_readers.get(binding.asset_source_id)
        if reader is None:
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={
                    "kind": "mcp_asset_source",
                    "asset_source_id": binding.asset_source_id,
                    "server_id": server.id,
                },
            )
        versions = _bound_resource_versions(server, binding)
        resource = await SandboxResource.from_asset_versions(
            server.id,
            reader,
            versions,
        )
        local_files = None if resource is None else resource.files
        arguments: list[str | SandboxResourcePath] = []
        for argument in server.args:
            if not argument.startswith("resource:"):
                arguments.append(argument)
                continue
            relative = argument[len("resource:") :]
            if local_files is None:
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={"kind": "mcp_local_resource", "server_id": server.id},
                )
            arguments.append(
                SandboxResourcePath(server.id, relative)
                if sandboxed
                else str(local_files[relative])
            )
        projections[server.id] = _MCPProjection(
            server.id,
            tuple(arguments),
            (resource,) if sandboxed and resource is not None else (),
        )
    return projections


async def materialize_mcp_capabilities(
    servers: Sequence[MCPServerSpec],
    selectors: Sequence[str],
    *,
    sandbox: Sandbox | None,
    sandbox_session: object | None,
    host_cwd: "str | None",
    bindings: Mapping[str, _MCPBinding],
    projections: Mapping[str, _MCPProjection],
    tool_operations: "ToolOperationBridge | None",
    tool_metrics: "_ToolMetricContext | None",
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Materialize compiler-selected MCP servers."""
    policy, required = _selector_policy(selectors)
    descriptor = managed_tool_descriptor_from_metadata(_MCP_TOOL_METADATA)
    values: list[AbstractCapability[AgentContext[object]]] = []
    try:
        for server in servers:
            if server.id not in policy:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            binding = bindings.get(server.id)
            projection = projections.get(server.id)
            if binding is None or (server.transport == "stdio" and projection is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current_policy = _mcp_execution_policy(server, sandbox)
            if dict(binding.execution_policy) != dict(current_policy):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            allowed = policy[server.id]

            transport = _create_mcp_transport(
                server,
                sandboxed=sandbox is not None,
                sandbox_session=sandbox_session,
                host_cwd=host_cwd,
                args=() if projection is None else projection.args,
                resources=() if projection is None else projection.resources,
            )
            toolset = _MCPDiscoveryToolset(transport, server=server)
            mapped = _MCPModelToolset(
                toolset,
                server.id,
                allowed,
                required.get(server.id, frozenset()),
            )
            boundary = BoundaryToolset(
                (mapped,),
                {},
                id=f"linktools.mcp.{server.id}",
                descriptor=descriptor,
                tool_operations=tool_operations,
                tool_metrics=tool_metrics,
            )
            values.append(
                _MCPCapability(
                    f"linktools.ai.mcp.{server.id}",
                    boundary,
                    toolset,
                )
            )
            _logger.debug(
                "MCP server materialized: server=%s transport=%s selected_tools=%s",
                server.id,
                server.transport,
                "*" if allowed is None else tuple(sorted(allowed)),
            )
    except BaseException as primary_error:
        try:
            await close_mcp_resources(values)
        except BaseException as cleanup_error:
            _raise_primary_after_cleanup(primary_error, cleanup_error)
        raise
    return tuple(values)


def _bound_resource_versions(
    server: MCPServerSpec,
    binding: _MCPBinding,
) -> "dict[str, AssetVersionRef]":
    if binding.versions is None:
        raise AIError(
            ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"kind": "mcp_resource", "server_id": server.id},
        )
    if server.resource is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    prefix = f"{server.resource.id}/"
    values: dict[str, AssetVersionRef] = {}
    for version in binding.versions:
        if (
            version.key.kind != server.resource.kind
            or not version.key.id.startswith(prefix)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        relative = version.key.id[len(prefix) :]
        validate_resource_path(relative)
        if relative in values:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values[relative] = version
    validate_resource_tree(values)
    for argument in server.args:
        if argument.startswith("resource:"):
            relative = argument[len("resource:") :]
            if relative not in values:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return {relative: values[relative] for relative in sorted(values)}


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


def _mcp_execution_policy(
    server: MCPServerSpec,
    sandbox: Sandbox | None,
) -> Mapping[str, JsonValue]:
    if server.transport != "stdio":
        return {"version": 1, "boundary": "host-network"}
    if sandbox is None:
        return {"version": 1, "boundary": "host-stdio"}
    if not isinstance(sandbox, StdioSandbox):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return sandbox.stdio_execution_policy()


__all__ = []
