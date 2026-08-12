#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability binding and Pydantic AI execution boundaries."""

from ._contract import (
    CapabilityBinding,
    CapabilityProvider,
    CapabilityRefResolution,
    CapabilityRuntimeContext,
    StaticCapabilityBinding,
    UnresolvedCapabilityBinding,
    canonical_bootstrap_refs,
    group_capability_refs,
    unresolved_binding,
    validate_fingerprint,
)
from ._mcp import (
    AssetMCPProvider,
    MCPCallRequest,
    MCPConnectionPool,
    MCPRuntimeProvider,
    MCPServerCapabilityBinding,
    bind_mcp_capability,
    build_builtin_capability_providers,
    mcp_server_namespace,
    mcp_server_selector,
    mcp_tool_name,
    validate_mcp_response,
)
from ._mcp_runtime import PydanticMCPRuntimeProvider
from ._retrieval import RetrievalProvider
from ._skill import (
    SKILL_TOOL_NAMES,
    AssetSkillProvider,
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
    "CapabilityBinding", "CapabilityProvider", "CapabilityRefResolution", "CapabilityRuntimeContext", "canonical_bootstrap_refs",
    "MCPCallRequest", "MCPConnectionPool", "MCPRuntimeProvider", "MCPServerCapabilityBinding", "AssetMCPProvider", "PydanticMCPRuntimeProvider",
    "build_builtin_capability_providers", "mcp_server_namespace", "mcp_server_selector", "mcp_tool_name",
    "RetrievalProvider", "SkillCapability", "SkillCapabilityBinding", "SkillCatalogSnapshot", "SkillCatalogView",
    "SkillDescriptor", "AssetSkillProvider", "SKILL_TOOL_NAMES", "UnresolvedCapabilityBinding", "bind_mcp_capability", "bind_skill_capability",
    "group_capability_refs", "merge_skill_catalogs", "snapshot_skill_catalog", "unresolved_binding", "StaticCapabilityBinding",
    "validate_fingerprint", "validate_mcp_response",
]
