#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability resolution and materialization contracts."""

from ._contract import (
    CapabilityBinding,
    CapabilityGrant,
    CapabilityMaterializationContext,
    CapabilityProvider,
    CapabilityRefResolution,
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
    "CapabilityGrant",
    "CapabilityMaterializationContext",
    "CapabilityProvider",
    "CapabilityRefResolution",
    "MCPCapabilityProvider",
    "MCPRuntime",
    "SkillCapabilityProvider",
    "group_capability_refs",
    "unresolved_binding",
    "validate_fingerprint",
]
