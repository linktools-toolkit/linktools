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
    agent_ref_payload,
    binding_digest_payload,
    capability_ref_payload,
)
from ._schema import canonicalize_json_schema, canonicalize_pydantic_model_schema

__all__ = [
    "AgentSpec",
    "AgentSpecCodec",
    "AgentMarkdownSpecCodec",
    "AgentUsageLimits",
    "agent_ref_payload",
    "binding_digest_payload",
    "canonicalize_json_schema",
    "canonicalize_pydantic_model_schema",
    "validate_logical_id",
    "capability_ref_payload",
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
