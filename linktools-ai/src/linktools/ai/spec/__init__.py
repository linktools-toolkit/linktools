#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Agent, Skill, and MCP declaration contracts."""

from ._codec import (
    AgentSpecCodec,
    MCPServerSpecCodec,
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpecCodec,
    SpecCodec,
)
from ._agent_markdown import AgentMarkdownSpecCodec
from ._contract import (
    AgentSpec,
    AgentUsageLimits,
    MCPServerSpec,
    SkillSpec,
    SubagentRef,
    ThinkingEffort,
    ThinkingValue,
    canonical_selectors,
    normalize_thinking,
    mcp_server_selector,
    mcp_tool_selector,
    parse_mcp_tool_selector,
)
from ..core import validate_logical_id
from ._identity import (
    agent_spec_identity_payload,
    bound_agent_spec_identity_payload,
    binding_identity_payload,
    capability_identity_payload,
)
from ._schema import canonicalize_json_schema, canonicalize_pydantic_model_schema

__all__ = [
    "AgentSpec",
    "AgentSpecCodec",
    "AgentMarkdownSpecCodec",
    "AgentUsageLimits",
    "agent_spec_identity_payload",
    "bound_agent_spec_identity_payload",
    "binding_identity_payload",
    "canonicalize_json_schema",
    "canonicalize_pydantic_model_schema",
    "validate_logical_id",
    "capability_identity_payload",
    "MCPServerSpec",
    "MCPServerSpecCodec",
    "SkillMarkdownSpecAdapter",
    "SkillMarkdownSpecCodec",
    "SkillSpec",
    "SkillSpecCodec",
    "SubagentRef",
    "SpecCodec",
    "ThinkingEffort",
    "ThinkingValue",
    "canonical_selectors",
    "normalize_thinking",
    "mcp_server_selector",
    "mcp_tool_selector",
    "parse_mcp_tool_selector",
]
