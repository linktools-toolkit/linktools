#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable declaration contracts for Agent, Skill, and MCP specifications."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ..core import (
    ThinkingEffort,
    ThinkingValue,
    normalize_thinking,
)
from ..errors import AIError, ErrorCode

_MCP_NAMESPACE = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_mcp_tool_selector(selector: str) -> "tuple[str, str | None] | None":
    """Parse one MCP selector into its namespace and optional exact tool."""
    if not isinstance(selector, str) or not selector.startswith("mcp__"):
        return None
    parts = selector[5:].split("__")
    if len(parts) == 1:
        namespace = parts[0]
        tool = None
    elif len(parts) == 2:
        namespace, tool = parts
        if not tool or tool == "*":
            if tool != "*":
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            tool = None
    else:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not namespace or _MCP_NAMESPACE.fullmatch(namespace) is None:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if tool is not None and (
        not tool
        or tool != tool.strip()
        or "*" in tool
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return namespace, tool


def canonical_selectors(
    value: Sequence[str],
    *,
    field_name: str,
    mcp: bool = False,
) -> "tuple[str, ...]":
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} must be an array of strings")
    selectors: set[str] = set()
    for raw in value:
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        if raw == "*":
            return ("*",)
        if not mcp and "*" in raw:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        selector = raw
        parsed = parse_mcp_tool_selector(raw) if mcp else None
        if parsed is not None:
            namespace, tool = parsed
            selector = (
                f"mcp__{namespace}__*"
                if tool is None
                else f"mcp__{namespace}__{tool}"
            )
        elif "*" in raw:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        selectors.add(selector)
    if mcp:
        wildcard_servers = {
            parsed[0]
            for selector in selectors
            if (parsed := parse_mcp_tool_selector(selector)) is not None
            and parsed[1] is None
        }
        selectors = {
            selector
            for selector in selectors
            if (
                (parsed := parse_mcp_tool_selector(selector)) is None
                or parsed[0] not in wildcard_servers
                or parsed[1] is None
            )
        }
    return tuple(sorted(selectors))


@dataclass(frozen=True, slots=True)
class AgentUsageLimits:
    model_requests: "int | None" = None
    tool_calls: "int | None" = None
    input_tokens: "int | None" = None
    output_tokens: "int | None" = None
    total_tokens: "int | None" = None

    def __post_init__(self) -> None:
        values = (
            self.model_requests,
            self.tool_calls,
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
        )
        if all(value is None for value in values):
            raise ValueError("usage limits must define at least one limit")
        if any(value is not None and (not isinstance(value, int) or isinstance(value, bool)) for value in values):
            raise TypeError("usage limits must contain integers or None")
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("usage limits must be positive integers")


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Durable, runtime-independent declaration of one Agent."""

    id: str
    model: str = "default"
    system_prompt: str = ""
    instructions: "tuple[str, ...]" = ()
    allow_tools: "tuple[str, ...]" = ("*",)
    allow_skills: "tuple[str, ...]" = ("*",)
    allow_subagents: "tuple[str, ...]" = ("*",)
    allow_capabilities: "tuple[str, ...]" = ("*",)
    usage_limits: "AgentUsageLimits | None" = None
    planning: bool = False
    thinking: ThinkingValue = False
    tool_retries: int = 10000
    output_retries: int = 3
    description: "str | None" = None
    preload_skills: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("agent id must be a string")
        if not self.id.strip():
            raise ValueError("agent id must be non-empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent model must be a non-empty string")
        if not isinstance(self.system_prompt, str):
            raise TypeError("agent system_prompt must be a string")
        if isinstance(self.instructions, (str, bytes, bytearray)) or not isinstance(self.instructions, Sequence):
            raise TypeError("agent instructions must be a string array")
        instructions = tuple(self.instructions)
        if any(not isinstance(item, str) for item in instructions):
            raise TypeError("agent instructions must be strings")
        if self.usage_limits is not None and not isinstance(self.usage_limits, AgentUsageLimits):
            raise TypeError("agent usage_limits must be AgentUsageLimits or None")
        if not isinstance(self.planning, bool):
            raise TypeError("agent planning must be bool")
        for name, value in (
            ("tool_retries", self.tool_retries),
            ("output_retries", self.output_retries),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"agent {name} must be an integer")
            if value < 0:
                raise ValueError(f"agent {name} cannot be negative")
        if self.description is not None and (
            not isinstance(self.description, str) or not 1 <= len(self.description) <= 1024
        ):
            raise ValueError("agent description must contain 1..1024 characters")
        thinking = normalize_thinking(self.thinking)
        allow_skills = canonical_selectors(self.allow_skills, field_name="allow_skills")
        preload_skills = canonical_selectors(self.preload_skills, field_name="preload_skills")
        allow_capabilities = canonical_selectors(
            self.allow_capabilities,
            field_name="allow_capabilities",
        )
        if "*" in preload_skills:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                "preload_skills requires exact skill ids",
            )
        if allow_skills != ("*",) and any(
            skill_id not in allow_skills for skill_id in preload_skills
        ):
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                "preload_skills must be selected by allow_skills",
            )
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "allow_tools", canonical_selectors(self.allow_tools, field_name="allow_tools", mcp=True))
        object.__setattr__(self, "allow_skills", allow_skills)
        object.__setattr__(self, "allow_subagents", canonical_selectors(self.allow_subagents, field_name="allow_subagents"))
        object.__setattr__(self, "allow_capabilities", allow_capabilities)
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "preload_skills", preload_skills)


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    content: str
    description: "str | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("skill id must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("skill content must be a string")
        if self.description is not None and (
            not isinstance(self.description, str) or not 1 <= len(self.description) <= 1024
        ):
            raise ValueError("skill description must contain 1..1024 characters")


@dataclass(frozen=True, slots=True)
class SubagentRef:
    """Durable logical reference to one allowed child Agent."""

    kind: Literal["agent"]
    id: str
    description: "str | None" = None

    def __post_init__(self) -> None:
        if self.kind != "agent" or not isinstance(self.id, str) or not self.id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.description is not None and (
            not isinstance(self.description, str) or not 1 <= len(self.description) <= 1024
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def to_payload(self) -> "dict[str, object]":
        payload: dict[str, object] = {"kind": "agent", "id": self.id}
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_payload(cls, value: object) -> "SubagentRef":
        if (
            not isinstance(value, Mapping)
            or not {"kind", "id"}.issubset(value)
            or value.get("kind") != "agent"
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = value.get("id")
        description = value.get("description")
        if not isinstance(identity, str) or not identity.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if description is not None and not isinstance(description, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls("agent", identity, description)


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    command: str
    args: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("MCP server id must be non-empty")
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("MCP server command must be non-empty")
        if isinstance(self.args, (str, bytes, bytearray)) or not isinstance(self.args, Sequence):
            raise TypeError("MCP server args must be a string sequence")
        args = tuple(self.args)
        if any(not isinstance(item, str) for item in args):
            raise TypeError("MCP server args must be strings")
        object.__setattr__(self, "args", args)


__all__ = [
    "AgentSpec",
    "AgentUsageLimits",
    "MCPServerSpec",
    "SkillSpec",
    "SubagentRef",
    "ThinkingEffort",
    "ThinkingValue",
    "canonical_selectors",
    "normalize_thinking",
    "parse_mcp_tool_selector",
]
