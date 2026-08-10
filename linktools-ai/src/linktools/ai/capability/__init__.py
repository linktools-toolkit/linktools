#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability resolver and Pydantic AI execution boundaries."""

from ._contract import (
    CapabilityBinding,
    CapabilityInjection,
    CapabilityRefResolution,
    CapabilityResolver,
    CapabilityRuntimeContext,
    UnresolvedCapabilityBinding,
    unresolved_binding,
    validate_fingerprint,
)
from ._mcp import (
    MCPCallRequest,
    MCPConnectionPool,
    MCPServerCapabilityBinding,
    MCPServerCapabilityResolver,
    MCPToolProvider,
    validate_mcp_response,
)
from ._registry import CapabilityResolverRegistry
from ._retrieval import RetrievalProvider
from ._skill import (
    SkillCapability,
    SkillCapabilityBinding,
    SkillCapabilityResolver,
    SkillCatalogSnapshot,
    SkillCatalogView,
    SkillDescriptor,
    SkillProvider,
    snapshot_skill_catalog,
)

__all__ = [
    "CapabilityBinding", "CapabilityInjection", "CapabilityRefResolution", "CapabilityResolver",
    "CapabilityResolverRegistry", "CapabilityRuntimeContext", "MCPCallRequest", "MCPConnectionPool",
    "MCPServerCapabilityBinding", "MCPServerCapabilityResolver", "MCPToolProvider", "RetrievalProvider",
    "SkillCapability", "SkillCapabilityBinding", "SkillCapabilityResolver", "SkillCatalogSnapshot",
    "SkillCatalogView", "SkillDescriptor", "SkillProvider", "UnresolvedCapabilityBinding",
    "snapshot_skill_catalog", "unresolved_binding", "validate_fingerprint", "validate_mcp_response",
]
