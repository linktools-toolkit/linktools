#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public capability extension contracts."""

from ._persistence import CheckpointCapability, TraceCapability
from ._tool_responses import ToolResponseFixture, ToolResponsesCapability
from ._tool_results import ToolResultsCapability
from ._run_callbacks import RunCallbacks, RunCallbackScope, bind_run_callbacks
from ._referenced_inputs import ReferencedInputsCapability
from ._asset import AssetCapability
from ._context import AgentContext
from ._declaration import AgentDeclarationLoader, capture_contributions
from ._resource_path import (
    mcp_resource_path,
    validate_resource_path,
    validate_resource_tree,
)
from ._contribution import CapabilityContribution
from ._loading import CapabilityLoadContext, CapabilityLoadEntry, CapabilityLoader
from ._skill import SkillCapability, SkillDefinition
from ._skill_source import (
    AssetSkillSource,
    SkillLocation,
    SkillSource,
    SkillResource,
    SkillSourceView,
    SkillSourceRef,
    SkillSourceRegistry,
)
from ._subagent import SubagentCapability, SubagentDelegate
from ._tool_signal import ToolCallFailed, ToolCallRetry
from ._pydantic_tool_control import PydanticToolControlCapability
from ._tool_metadata import (
    ToolClass,
    ToolEffectPolicy,
    is_tool_class,
    is_tool_effect_policy,
    TOOL_CLASS_METADATA_KEY,
    TOOL_MCP_SELECTOR_METADATA_KEY,
    TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY,
    TOOL_CONTEXT_DEDUPE_METADATA_KEY,
    TOOL_EFFECT_POLICY_METADATA_KEY,
    TOOL_PATH_FIELDS_METADATA_KEY,
    TOOL_PLAN_SAFE_METADATA_KEY,
    tool_class_from_metadata,
    tool_compaction_keep_result_from_metadata,
    tool_context_dedupe_from_metadata,
    tool_effect_policy_from_metadata,
    tool_path_fields_from_metadata,
    tool_plan_safe_from_metadata,
    tool_metadata,
    validate_tool_metadata,
)
from ._shell import ShellCapability
from ._filesystem import (
    FileSystem,
    FileSystemToolDeclaration,
    filesystem_capabilities,
    filesystem_tool_declarations,
)
from ..spec import mcp_server_selector, mcp_tool_selector

from ._session import SessionCapability
from ._execution import ExecutionCapability

__all__ = [
    "ReferencedInputsCapability",
    "SessionCapability",
    "ExecutionCapability",
    "ToolResponseFixture",
    "CheckpointCapability", "TraceCapability",
    "ToolResponsesCapability",
    "ToolResultsCapability",
    "RunCallbacks", "RunCallbackScope", "bind_run_callbacks",
    "AssetCapability",
    "FileSystem",
    "ShellCapability",
    "ToolClass",
    "ToolEffectPolicy",
    "is_tool_class",
    "is_tool_effect_policy",
    "AgentContext",
    "AgentDeclarationLoader",
    "capture_contributions",
    "CapabilityContribution",
    "CapabilityLoadContext",
    "CapabilityLoadEntry",
    "CapabilityLoader",
    "AssetSkillSource",
    "SkillCapability",
    "SubagentCapability",
    "TOOL_CLASS_METADATA_KEY",
    "TOOL_MCP_SELECTOR_METADATA_KEY",
    "TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY",
    "TOOL_CONTEXT_DEDUPE_METADATA_KEY",
    "TOOL_EFFECT_POLICY_METADATA_KEY",
    "TOOL_PATH_FIELDS_METADATA_KEY",
    "TOOL_PLAN_SAFE_METADATA_KEY",
    "FileSystemToolDeclaration",
    "SkillDefinition",
    "SkillLocation",
    "SkillSource",
    "SkillResource",
    "SkillSourceView",
    "SkillSourceRef",
    "SkillSourceRegistry",
    "SubagentDelegate",
    "ToolCallFailed",
    "ToolCallRetry",
    "PydanticToolControlCapability",
    "mcp_resource_path",
    "mcp_server_selector",
    "mcp_tool_selector",
    "filesystem_capabilities",
    "filesystem_tool_declarations",
    "validate_resource_path",
    "validate_resource_tree",
    "tool_class_from_metadata",
    "tool_compaction_keep_result_from_metadata",
    "tool_context_dedupe_from_metadata",
    "tool_effect_policy_from_metadata",
    "tool_path_fields_from_metadata",
    "tool_plan_safe_from_metadata",
    "tool_metadata",
    "validate_tool_metadata",
]
