#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility proxy for LinkTools-only capability support."""

from __future__ import annotations

import linktools.ai.runtime._capability_support as _support

MEMORY_READ_TOOL_NAMES = _support.MEMORY_READ_TOOL_NAMES
MEMORY_TOOL_NAMES = _support.MEMORY_TOOL_NAMES
PLANNING_TOOL_NAMES = _support.PLANNING_TOOL_NAMES
PLAN_SAFE_METADATA_KEY = _support.PLAN_SAFE_METADATA_KEY
SUBAGENT_TOOL_NAMES = _support.SUBAGENT_TOOL_NAMES
WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = _support.WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
WORKSPACE_FILESYSTEM_TOOL_NAMES = _support.WORKSPACE_FILESYSTEM_TOOL_NAMES
WORKSPACE_SHELL_TOOL_NAMES = _support.WORKSPACE_SHELL_TOOL_NAMES
ToolOperationBridge = _support.ToolOperationBridge
ToolOperationDecision = _support.ToolOperationDecision
select_runtime_tool_names = _support.select_runtime_tool_names
tool_allowed_in_planning = _support.tool_allowed_in_planning
tool_is_control = _support.tool_is_control
tool_name_allowed = _support.tool_name_allowed


def __getattr__(name: str) -> object:
    return getattr(_support, name)


__all__ = [
    "MEMORY_READ_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "PLANNING_TOOL_NAMES",
    "PLAN_SAFE_METADATA_KEY",
    "SUBAGENT_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "ToolOperationBridge",
    "ToolOperationDecision",
    "select_runtime_tool_names",
    "tool_allowed_in_planning",
    "tool_is_control",
    "tool_name_allowed",
]
