#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability loading, tool adaptation, and runtime materialization contracts."""

from ._context import RunContext
from ._group import (
    CapabilityContribution,
    CapabilityGroup,
    CapabilityLoadContext,
    CapabilityLoader,
    capability_fingerprint,
    contribution_semantic_contract,
)
from ._mcp import (
    materialize_mcp_servers,
    mcp_selector_server,
    mcp_server_namespace,
    mcp_server_selector,
    mcp_tool_name,
)
from ._names import SKILL_TOOL_NAMES, SUBAGENT_TOOL_NAMES
from ._skill import SkillCapability, SkillDefinition
from ._skill_source import (
    AssetSkillResourceSource,
    LocalSkillResourceSource,
    SkillLocation,
    SkillResourceSource,
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
    normalize_skill_resource_path,
)
from ._subagent import SubagentCapability, SubagentDelegate
from ._workspace import (
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
    workspace_capabilities,
    workspace_tool_class,
    workspace_tool_contributions,
)

__all__ = [
    "AssetSkillResourceSource",
    "CapabilityContribution",
    "CapabilityGroup",
    "CapabilityLoadContext",
    "CapabilityLoader",
    "LocalSkillResourceSource",
    "RunContext",
    "SKILL_TOOL_NAMES",
    "SUBAGENT_TOOL_NAMES",
    "SkillCapability",
    "SkillDefinition",
    "SkillLocation",
    "SkillResourceSource",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "SubagentCapability",
    "SubagentDelegate",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "capability_fingerprint",
    "contribution_semantic_contract",
    "materialize_mcp_servers",
    "mcp_selector_server",
    "mcp_server_namespace",
    "mcp_server_selector",
    "mcp_tool_name",
    "normalize_skill_resource_path",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_contributions",
]
