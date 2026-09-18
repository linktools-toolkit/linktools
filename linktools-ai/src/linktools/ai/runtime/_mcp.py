#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize compiler-selected stdio MCP as Runtime capabilities."""

import asyncio
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
from ..core import Principal, ResourceRef
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, parse_mcp_tool_selector
from ._tool import ToolOperationBridge
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
    ) -> None:
        self.id = capability_id
        self._toolset = toolset

    def get_toolset(self) -> AbstractToolset[AgentContext[object]]:
        return self._toolset


async def materialize_mcp_capabilities(
    servers: Sequence[MCPServerSpec],
    selectors: Sequence[str],
    *,
    principal: Principal,
    execution: ResourceRef,
    execution_root: str,
    tool_operations: "ToolOperationBridge | None",
    tool_metrics: "_ToolMetricContext | None",
    background_tasks: set[asyncio.Task[object]],
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Materialize only compiler-selected stdio MCP servers."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai.mcp import MCPToolset

    if principal.tenant_id != execution.tenant_id:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    policy = _selector_policy(selectors)
    descriptor = managed_tool_descriptor_from_metadata(_MCP_TOOL_METADATA)
    root = str(Path(execution_root).expanduser().resolve())
    values: list[AbstractCapability[AgentContext[object]]] = []
    seen_namespaces: set[str] = set()
    for server in servers:
        namespace = mcp_server_namespace(server.id)
        if namespace in seen_namespaces:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        seen_namespaces.add(namespace)
        allowed = policy.get(namespace)
        if allowed is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        client = Client(
            StdioTransport(
                server.command,
                list(server.args),
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
            )
        )
        _logger.debug(
            "MCP server materialized: server=%s namespace=%s selected_tools=%s",
            server.id,
            namespace,
            tuple(sorted(allowed or ("*",))),
        )
    return tuple(values)


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
