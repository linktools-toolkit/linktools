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
)
from ._mcp import mcp_server_namespace, mcp_server_selector
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
    WorkspaceAccess,
    workspace_capabilities,
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
    "TOOL_CLASS_METADATA_KEY",
    "TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY",
    "TOOL_CONTEXT_DEDUPE_METADATA_KEY",
    "TOOL_EFFECT_METADATA_KEY",
    "TOOL_PATH_FIELDS_METADATA_KEY",
    "TOOL_PLAN_SAFE_METADATA_KEY",
    "SkillDefinition",
    "SkillLocation",
    "SkillResourceSource",
    "SkillResourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "SubagentDelegate",
    "TaskExpander",
    "TaskExpansionContext",
    "WorkspaceAccess",
    "mcp_server_namespace",
    "mcp_server_selector",
    "workspace_capabilities",
    "workspace_tool_contributions",
    "tool_class_from_metadata",
    "tool_compaction_keep_result_from_metadata",
    "tool_context_dedupe_from_metadata",
    "tool_effect_from_metadata",
    "tool_path_fields_from_metadata",
    "tool_plan_safe_from_metadata",
    "tool_semantic_metadata",
    "validate_tool_semantic_metadata",
]
