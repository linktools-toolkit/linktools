#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public capability extension contracts."""

from ._context import AgentContext
from ._group import (
    CapabilityContribution,
    CapabilityGroup,
    CapabilityLoadContext,
    CapabilityLoadEntry,
    CapabilityLoader,
    PLAN_SAFE_METADATA_KEY,
)
from ._mcp import (
    mcp_selector_server,
    mcp_server_namespace,
    mcp_server_selector,
)
from ._names import SKILL_TOOL_NAMES, SUBAGENT_TOOL_NAMES
from ._skill import LinkToolsSkills, SkillDefinition
from ._skill_source import (
    AssetSkillResourceSource,
    LocalSkillResourceSource,
    SkillLocation,
    SkillResourceSource,
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
)
from ._subagent import LinkToolsSubagents, SubagentDelegate
from ._task import TaskExpander, TaskExpansionContext
from ._workspace import (
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
    WorkspaceAccess,
    workspace_capabilities,
    workspace_tool_class,
    workspace_tool_contributions,
)

__all__ = [
    "AgentContext",
    "CapabilityContribution",
    "CapabilityGroup",
    "CapabilityLoadContext",
    "CapabilityLoadEntry",
    "CapabilityLoader",
    "AssetSkillResourceSource",
    "LinkToolsSkills",
    "LinkToolsSubagents",
    "LocalSkillResourceSource",
    "PLAN_SAFE_METADATA_KEY",
    "SkillDefinition",
    "SkillLocation",
    "SkillResourceSource",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "SKILL_TOOL_NAMES",
    "SUBAGENT_TOOL_NAMES",
    "SubagentDelegate",
    "TaskExpander",
    "TaskExpansionContext",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "WorkspaceAccess",
    "mcp_selector_server",
    "mcp_server_namespace",
    "mcp_server_selector",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_contributions",
]
