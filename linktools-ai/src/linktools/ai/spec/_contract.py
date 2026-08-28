#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable declaration contracts for Agent, Skill, and MCP specifications."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

from ..core import (
    ImmutableJsonMapping,
    JsonValue,
    ThinkingEffort,
    ThinkingValue,
    normalize_thinking,
)
from ..errors import AIError, ErrorCode


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
        if mcp and raw.startswith("mcp__"):
            parts = raw.split("__")
            if len(parts) == 2 and parts[1]:
                selector = f"{raw}__*"
            elif len(parts) != 3 or not parts[1] or not parts[2]:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid MCP selector")
            elif "*" in parts[2] and parts[2] != "*":
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid MCP selector")
        elif "*" in raw:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        selectors.add(selector)
    if mcp:
        wildcard_servers = {
            selector.removesuffix("__*")
            for selector in selectors
            if selector.startswith("mcp__") and selector.endswith("__*")
        }
        selectors = {
            selector
            for selector in selectors
            if not (
                selector.startswith("mcp__")
                and any(selector.startswith(f"{server}__") for server in wildcard_servers)
                and selector not in {f"{server}__*" for server in wildcard_servers}
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
    usage_limits: "AgentUsageLimits | None" = None
    planning: bool = False
    thinking: ThinkingValue = False
    description: "str | None" = None
    _extensions: Mapping[str, JsonValue] = field(default_factory=dict, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("agent id must be a string")
        if not self.id.strip():
            raise ValueError("agent id must be non-empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent model must be a non-empty string")
        if not isinstance(self.system_prompt, str):
            raise TypeError("agent system prompt must be a string")
        if isinstance(self.instructions, (str, bytes, bytearray)) or not isinstance(self.instructions, Sequence):
            raise TypeError("agent instructions must be a string array")
        instructions = tuple(self.instructions)
        if any(not isinstance(item, str) for item in instructions):
            raise TypeError("agent instructions must be strings")
        if self.usage_limits is not None and not isinstance(self.usage_limits, AgentUsageLimits):
            raise TypeError("agent usage_limits must be AgentUsageLimits or None")
        if not isinstance(self.planning, bool):
            raise TypeError("agent planning must be bool")
        if self.description is not None and (
            not isinstance(self.description, str) or not 1 <= len(self.description) <= 1024
        ):
            raise ValueError("agent description must contain 1..1024 characters")
        thinking = normalize_thinking(self.thinking)
        try:
            extensions = ImmutableJsonMapping(self._extensions)
        except (TypeError, ValueError) as error:
            raise TypeError("agent extensions must be JSON values") from error
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "allow_tools", canonical_selectors(self.allow_tools, field_name="allow_tools", mcp=True))
        object.__setattr__(self, "allow_skills", canonical_selectors(self.allow_skills, field_name="allow_skills"))
        object.__setattr__(self, "allow_subagents", canonical_selectors(self.allow_subagents, field_name="allow_subagents"))
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "_extensions", extensions)


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    content: str
    description: "str | None" = None
    _extensions: Mapping[str, JsonValue] = field(default_factory=dict, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("skill id must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("skill content must be a string")
        if self.description is not None and (
            not isinstance(self.description, str) or not 1 <= len(self.description) <= 1024
        ):
            raise ValueError("skill description must contain 1..1024 characters")
        object.__setattr__(self, "_extensions", ImmutableJsonMapping(self._extensions))


@dataclass(frozen=True, slots=True)
class SubagentRef:
    """Durable logical reference to one allowed child Agent."""

    kind: Literal["agent"]
    id: str
    description: "str | None" = None
    _extensions: Mapping[str, JsonValue] = field(
        default_factory=dict,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if self.kind != "agent" or not isinstance(self.id, str) or not self.id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.description is not None and (
            not isinstance(self.description, str) or not 1 <= len(self.description) <= 1024
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            extensions = ImmutableJsonMapping(self._extensions)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "_extensions", extensions)

    def to_payload(self) -> "dict[str, JsonValue]":
        payload = dict(self._extensions)
        payload.update({"kind": "agent", "id": self.id})
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_payload(cls, value: object) -> "SubagentRef":
        if not isinstance(value, Mapping) or value.get("kind") != "agent":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = value.get("id")
        description = value.get("description")
        if not isinstance(identity, str) or not identity.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if description is not None and not isinstance(description, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        extensions: dict[str, JsonValue] = {}
        for key, raw in value.items():
            if key in {"kind", "id", "description"}:
                continue
            if not isinstance(key, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                extensions[key] = cast(JsonValue, raw)
                ImmutableJsonMapping(extensions)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls("agent", identity, description, extensions)


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    command: str
    args: "tuple[str, ...]" = ()
    _extensions: Mapping[str, JsonValue] = field(default_factory=dict, repr=False, compare=False, hash=False)

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
        object.__setattr__(self, "_extensions", ImmutableJsonMapping(self._extensions))


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
]