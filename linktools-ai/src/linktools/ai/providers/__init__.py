#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider Protocols: the configuration-source-agnostic surfaces the Runtime
consumes. A Registry is merely the default file/resource-backed implementation
of these Protocols -- business systems may substitute any backend that returns
the standard Specs."""

from .agent import AgentSpecProvider
from .bundle import ProviderBundle, ProviderPrefixes
from .mcp import MCPServerSpecProvider
from .package import PackageResourceProvider, PackageSpecProvider
from .skill import SkillSpecProvider
from .subagent import AgentBackedSubagentSpecProvider, SubagentSpecProvider
from .swarm import SwarmSpecProvider
from .tool_policy import ToolPolicyProvider

__all__ = [
    "AgentSpecProvider",
    "SkillSpecProvider",
    "MCPServerSpecProvider",
    "ToolPolicyProvider",
    "SwarmSpecProvider",
    "SubagentSpecProvider",
    "AgentBackedSubagentSpecProvider",
    "PackageSpecProvider",
    "PackageResourceProvider",
    "ProviderBundle",
    "ProviderPrefixes",
]
