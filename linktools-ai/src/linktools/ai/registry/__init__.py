#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spec registry package: shared spec-loading primitives plus the per-domain
registries (agent/swarm/tool/skill/mcp) that build on them.

Each registry is the DEFAULT file/resource-backed implementation of the
matching Provider Protocol (see ``linktools.ai.providers``) -- a recommended
format implementation, not a mandatory configuration system. Business systems
may substitute any backend that returns the standard Specs. The
``Markdown*``/``Yaml*`` aliases make that role explicit without breaking the
existing class names."""

from .agent import AgentRegistry, parse_agent_spec
from .mcp import MCPRegistry, MCPServerSpec, parse_mcp_spec
from .parser import SpecLoader, parse_markdown_text, parse_yaml_text
from .skill import SkillRegistry, SkillSpec, parse_skill_spec
from .swarm import SwarmRegistry, parse_swarm_spec
from .tool import ToolRegistry, ToolSpec

# Format-clarifying aliases: these names state which default format each
# registry parses. The original names remain the canonical, stable identifiers.
MarkdownAgentRegistry = AgentRegistry
MarkdownSkillRegistry = SkillRegistry
YamlMCPRegistry = MCPRegistry
YamlToolPolicyRegistry = ToolRegistry
YamlSwarmRegistry = SwarmRegistry

__all__ = [
    "AgentRegistry", "MarkdownAgentRegistry", "parse_agent_spec",
    "SkillRegistry", "MarkdownSkillRegistry", "SkillSpec", "parse_skill_spec",
    "MCPRegistry", "YamlMCPRegistry", "MCPServerSpec", "parse_mcp_spec",
    "ToolRegistry", "YamlToolPolicyRegistry", "ToolSpec",
    "SwarmRegistry", "YamlSwarmRegistry", "parse_swarm_spec",
    "SpecLoader", "parse_markdown_text", "parse_yaml_text",
]
