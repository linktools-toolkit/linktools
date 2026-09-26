#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Agent, Skill, and MCP declaration contracts."""

from ._adapter import AgentSpecAdapter, MCPServerSpecAdapter, SkillSpecAdapter
from ._codec import AgentSpecCodec, MCPServerSpecCodec, SkillSpecCodec, SpecCodec
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
from ._instructions import (
    RuleInstructionResolver,
    DEFAULT_REPOSITORY_INSTRUCTION_BYTES,
    DEFAULT_REPOSITORY_INSTRUCTION_DOCUMENTS,
    RepositoryInstructionDocument,
    RepositoryInstructionResolver,
    RepositoryInstructions,
)
from ._identity import (
    agent_ref_payload,
    binding_digest_payload,
    capability_ref_payload,
)
from ._schema import canonicalize_json_schema, canonicalize_pydantic_model_schema

__all__ = [
    "AgentSpec",
    "AgentSpecCodec",
    "AgentSpecAdapter",
    "AgentUsageLimits",
    "RuleInstructionResolver",
    "DEFAULT_REPOSITORY_INSTRUCTION_BYTES",
    "DEFAULT_REPOSITORY_INSTRUCTION_DOCUMENTS",
    "agent_ref_payload",
    "binding_digest_payload",
    "canonicalize_json_schema",
    "canonicalize_pydantic_model_schema",
    "validate_logical_id",
    "capability_ref_payload",
    "MCPServerSpec",
    "MCPServerSpecAdapter",
    "MCPServerSpecCodec",
    "RepositoryInstructionDocument",
    "RepositoryInstructionResolver",
    "RepositoryInstructions",
    "SkillSpecAdapter",
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
