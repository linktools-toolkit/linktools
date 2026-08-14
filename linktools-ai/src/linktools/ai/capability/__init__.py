#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability resolution and materialization contracts."""

from ._contract import (
    CapabilityBinding,
    CapabilityGrant,
    CapabilityMaterializationContext,
    CapabilityProvider,
    CapabilityRefResolution,
    UnresolvedCapabilityBinding,
    group_capability_refs,
    unresolved_binding,
    validate_fingerprint,
)
from ._mcp import MCPRuntime, MCPCapabilityProvider, MCPServerCapabilityBinding, bind_mcp_capability, mcp_server_namespace, mcp_server_selector, mcp_tool_name
from ._skill import SKILL_TOOL_NAMES, SkillCapability, SkillCapabilityBinding, SkillCapabilityProvider, SkillCatalogSnapshot, SkillCatalogView, SkillDescriptor, bind_skill_capability, merge_skill_catalogs, snapshot_skill_catalog

__all__ = [
    "CapabilityBinding",
    "CapabilityGrant",
    "CapabilityMaterializationContext",
    "CapabilityProvider",
    "CapabilityRefResolution",
    "MCPRuntime",
    "MCPCapabilityProvider",
    "SkillCapabilityProvider",
]
