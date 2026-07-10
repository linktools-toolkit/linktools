#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compat layer: auto-generate ToolContribution from a raw toolset when a
Provider does not yet return explicit descriptors. Uses conservative defaults
(custom / high / mutating) for every tool — never infers category from the tool
name. Providers should migrate to returning ToolContribution directly."""

from typing import Any

from ..security.descriptor import ToolDescriptor
from .contribution import ToolContribution


def _toolset_tool_names(toolset: Any) -> "tuple[str, ...]":
    """Best-effort tool-name extraction from a pydantic-ai toolset."""
    tools = getattr(toolset, "tools", None)
    if isinstance(tools, dict):
        return tuple(str(k) for k in tools.keys())
    return ()


def auto_contribute(
    toolset: Any,
    *,
    source: str = "builtin",
    capability_kind: str = "",
    capability_name: str = "",
) -> ToolContribution:
    """Wrap a raw toolset into a ToolContribution with conservative descriptors
    for every tool. Every auto-generated descriptor uses category=custom,
    risk=high, mutating=True — the safest default until the Provider supplies
    proper classification."""
    names = _toolset_tool_names(toolset)
    descriptors = tuple(
        ToolDescriptor(
            name=n,
            source=source,
            category="custom",
            risk="high",
            mutating=True,
            capability_kind=capability_kind,
            capability_name=capability_name,
        )
        for n in names
    )
    return ToolContribution(toolset=toolset, descriptors=descriptors)


def extract_handler(toolset: Any, name: str) -> "Any | None":
    """Extract the raw callable for a named tool from a FunctionToolset."""
    tools = getattr(toolset, "tools", None)
    if isinstance(tools, dict) and name in tools:
        return tools[name].function
    return None
