#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permission enforcement shared by native workspace tool compositions."""

from dataclasses import dataclass
from typing import Any

from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ApprovalRequiredToolset, ToolsetTool, WrapperToolset

from ..workspace import ToolPermissionPolicy
from ._context import AgentContext
from ._tool_metadata import tool_class_from_metadata
from ._tool_signal import ToolCallFailed


@dataclass
class _WorkspacePermissionToolset(WrapperToolset[AgentContext[object]]):
    policy: ToolPermissionPolicy

    def requires_approval(
        self, ctx: RunContext[AgentContext[object]], tool: ToolDefinition, arguments: dict[str, Any],
    ) -> bool:
        return self.policy.decide(
            tool_name=tool.name, tool_class=tool_class_from_metadata(tool.metadata),
        ) == "ask"

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentContext[object]],
        tool: ToolsetTool[AgentContext[object]],
    ) -> Any:
        if self.policy.decide(
            tool_name=tool.tool_def.name, tool_class=tool_class_from_metadata(tool.tool_def.metadata),
        ) == "deny":
            raise ToolCallFailed("Workspace policy does not allow this tool.")
        return await super().call_tool(name, tool_args, ctx, tool)


def workspace_policy_toolset(
    toolset: AbstractToolset[Any], policy: ToolPermissionPolicy,
) -> AbstractToolset[Any]:
    if policy.default == "allow" and all(rule.decision == "allow" for rule in policy.rules):
        return toolset
    permissions = _WorkspacePermissionToolset(toolset, policy)
    return ApprovalRequiredToolset(permissions, approval_required_func=permissions.requires_approval)
