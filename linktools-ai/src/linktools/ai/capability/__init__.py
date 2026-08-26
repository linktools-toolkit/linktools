#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability loading, tool adaptation, and Pydantic materialization boundary."""

from ._context import RunContext
from ._group import (
    CapabilityContribution,
    CapabilityGroup,
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
from ._names import SKILL_TOOL_NAMES
from ._skill import SkillCapability
from ._workspace import (
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
    workspace_capabilities,
    workspace_tool_class,
    workspace_tool_contributions,
)

__all__ = [
    "CapabilityContribution",
    "CapabilityGroup",
    "CapabilityLoader",
    "capability_fingerprint",
    "contribution_semantic_contract",
    "RunContext",
    "SKILL_TOOL_NAMES",
    "SkillCapability",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "materialize_mcp_servers",
    "mcp_selector_server",
    "mcp_server_namespace",
    "mcp_server_selector",
    "mcp_tool_name",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_contributions",
]
