#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability binding and Pydantic AI execution boundaries."""

from ._contract import (
    CapabilityBinding,
    CapabilityFeature,
    CapabilityProvider,
    CapabilityRefResolution,
    CapabilityRuntimeContext,
    StaticCapabilityBinding,
    UnresolvedCapabilityBinding,
    group_capability_refs,
    unresolved_binding,
    validate_fingerprint,
)
from ._mcp import (
    MCPCallRequest,
    MCPConnectionPool,
    MCPRuntimeProvider,
    MCPServerCapabilityBinding,
    bind_mcp_capability,
    validate_mcp_response,
)
from ._mcp_runtime import PydanticMCPRuntimeProvider
from ._retrieval import RetrievalProvider
from ._skill import (
    SkillCapability,
    SkillCapabilityBinding,
    SkillCatalogSnapshot,
    SkillCatalogView,
    SkillDescriptor,
    bind_skill_capability,
    merge_skill_catalogs,
    snapshot_skill_catalog,
)

__all__ = [
    "CapabilityBinding", "CapabilityFeature", "CapabilityProvider", "CapabilityRefResolution", "CapabilityRuntimeContext",
    "MCPCallRequest", "MCPConnectionPool", "MCPRuntimeProvider", "MCPServerCapabilityBinding", "PydanticMCPRuntimeProvider",
    "RetrievalProvider", "SkillCapability", "SkillCapabilityBinding", "SkillCatalogSnapshot", "SkillCatalogView",
    "SkillDescriptor", "UnresolvedCapabilityBinding", "bind_mcp_capability", "bind_skill_capability",
    "group_capability_refs", "merge_skill_catalogs", "snapshot_skill_catalog", "unresolved_binding", "StaticCapabilityBinding",
    "validate_fingerprint", "validate_mcp_response",
]
