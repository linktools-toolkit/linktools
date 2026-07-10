#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolDescriptor: structured metadata classifying a tool for policy decisions.
Avoids guessing risk from function names. Each tool exposed to the model must
have a descriptor so the governance chain (policy, pipeline, baseline) can make
decisions based on category/risk, not name patterns."""

from dataclasses import dataclass, field
from typing import Any, Mapping

# Standard category -> default risk mapping. Unknown categories
# default to "high" (conservative).
_CATEGORY_RISK: "dict[str, str]" = {
    "discovery": "low",
    "file-read": "low",
    "network-read": "medium",
    "file-write": "medium",
    "mcp-read": "medium",
    "subagent": "medium",
    "terminal": "high",
    "network-write": "high",
    "mcp-write": "high",
    "package-execute": "high",
    "package-read": "low",
}


def default_risk_for_category(category: str) -> str:
    """Conservative risk for a category. Unknown -> high."""
    return _CATEGORY_RISK.get(category, "high")


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    source: str               # "builtin" | "mcp" | "skill" | "subagent" | "package"
    category: str             # stable security classification
    risk: str                 # "low" | "medium" | "high" | "critical"
    mutating: bool
    capability_kind: str = ""     # ToolRef kind that produced this tool
    capability_name: str = ""     # capability instance name
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
