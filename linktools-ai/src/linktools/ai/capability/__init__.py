#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public capability extension contracts."""

from ._context import AgentContext
from ._declaration import AgentDeclarationLoader
from ._resource_path import (
    mcp_resource_path,
    validate_resource_path,
    validate_resource_tree,
)
from ._group import (
    CapabilityContribution,
    CapabilityGroup,
    CapabilityGroupSnapshot,
    CapabilityLoadContext,
    CapabilityLoadEntry,
    CapabilityLoader,
)
from ._skill import SkillCapability, SkillDefinition
from ._skill_source import (
    AssetSkillResourceSource,
    AssetVersionSkillResourceSource,
    LocalSkillResourceSource,
    SkillLocation,
    SkillResourceSource,
    SkillResourceVersion,
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
)
from ._subagent import SubagentCapability, SubagentDelegate
from ._task import TaskExpander, TaskExpansionContext
from ._tool_signal import ToolCallFailed, ToolCallRetry
from ._tool_semantic import (
    TOOL_CLASS_METADATA_KEY,
    TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY,
    TOOL_CONTEXT_DEDUPE_METADATA_KEY,
    TOOL_EFFECT_METADATA_KEY,
    TOOL_PATH_FIELDS_METADATA_KEY,
    TOOL_PLAN_SAFE_METADATA_KEY,
    tool_class_from_metadata,
    tool_compaction_keep_result_from_metadata,
    tool_context_dedupe_from_metadata,
    tool_effect_from_metadata,
    tool_path_fields_from_metadata,
    tool_plan_safe_from_metadata,
    tool_semantic_metadata,
    validate_tool_semantic_metadata,
)
from ._workspace import (
    ToolDeclaration,
    WorkspaceAccess,
    workspace_capabilities,
    workspace_tool_declarations,
)
from ..spec import mcp_server_selector, mcp_tool_selector

__all__ = [
    "AgentContext",
    "AgentDeclarationLoader",
    "CapabilityContribution",
    "CapabilityGroup",
    "CapabilityGroupSnapshot",
    "CapabilityLoadContext",
    "CapabilityLoadEntry",
    "CapabilityLoader",
    "AssetSkillResourceSource",
    "AssetVersionSkillResourceSource",
    "SkillCapability",
    "SubagentCapability",
    "LocalSkillResourceSource",
    "TOOL_CLASS_METADATA_KEY",
    "TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY",
    "TOOL_CONTEXT_DEDUPE_METADATA_KEY",
    "TOOL_EFFECT_METADATA_KEY",
    "TOOL_PATH_FIELDS_METADATA_KEY",
    "TOOL_PLAN_SAFE_METADATA_KEY",
    "ToolDeclaration",
    "SkillDefinition",
    "SkillLocation",
    "SkillResourceSource",
    "SkillResourceVersion",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "SubagentDelegate",
    "TaskExpander",
    "TaskExpansionContext",
    "ToolCallFailed",
    "ToolCallRetry",
    "WorkspaceAccess",
    "mcp_resource_path",
    "mcp_server_selector",
    "mcp_tool_selector",
    "workspace_capabilities",
    "workspace_tool_declarations",
    "validate_resource_path",
    "validate_resource_tree",
    "tool_class_from_metadata",
    "tool_compaction_keep_result_from_metadata",
    "tool_context_dedupe_from_metadata",
    "tool_effect_from_metadata",
    "tool_path_fields_from_metadata",
    "tool_plan_safe_from_metadata",
    "tool_semantic_metadata",
    "validate_tool_semantic_metadata",
]
