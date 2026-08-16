#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability resolution and materialization contracts."""

from ._contract import (
    CapabilityBinding,
    CapabilityMaterializationContext,
    CapabilityProvider,
    CapabilityRefResolution,
    RuntimeCapability,
    group_capability_refs,
    unresolved_binding,
    validate_fingerprint,
)
from ._mcp import MCPCapabilityProvider, MCPRuntime
from ._names import SKILL_TOOL_NAMES
from ._skill import SkillCapabilityProvider

__all__ = [
    "SKILL_TOOL_NAMES",
    "CapabilityBinding",
    "CapabilityMaterializationContext",
    "CapabilityProvider",
    "CapabilityRefResolution",
    "RuntimeCapability",
    "MCPCapabilityProvider",
    "MCPRuntime",
    "SkillCapabilityProvider",
    "group_capability_refs",
    "unresolved_binding",
    "validate_fingerprint",
]
