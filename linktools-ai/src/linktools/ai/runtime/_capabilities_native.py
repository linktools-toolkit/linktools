#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility proxy for LinkTools-only capability support."""

from __future__ import annotations

import linktools.ai.runtime._capability_support as _support

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

for _name in __all__:
    globals()[_name] = getattr(_support, _name)


def __getattr__(name: str) -> object:
    return getattr(_support, name)
