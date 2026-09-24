#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable declaration contracts for Agent, Skill, and MCP specifications."""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from ..asset import AssetKey
from ..core import (
    ThinkingEffort,
    ThinkingValue,
    normalize_thinking,
    validate_logical_id,
)
from ..errors import AIError, ErrorCode

_SELECTOR_SAFE_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


def parse_mcp_tool_selector(selector: str) -> "tuple[str, str | None] | None":
    """Parse one canonical MCP selector into its server and optional tool."""
    if not isinstance(selector, str) or not selector.startswith("mcp:"):
        return None
    parts = selector.split(":")
    if len(parts) != 3 or parts[0] != "mcp":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    server_id = _decode_selector_component(parts[1])
    if not server_id or not server_id.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if parts[2] == "*":
        return server_id, None
    tool_name = _decode_selector_component(parts[2])
    if not tool_name or tool_name != tool_name.strip() or _has_control(tool_name):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return server_id, tool_name


def mcp_server_selector(server_id: str) -> str:
    """Return the selector for every tool from one MCP server."""
    if not isinstance(server_id, str) or not server_id.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return f"mcp:{_encode_selector_component(server_id)}:*"


def mcp_tool_selector(server_id: str, tool_name: str) -> str:
    """Return the selector for one exact upstream MCP tool."""
    if not isinstance(server_id, str) or not server_id.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(tool_name, str) or not tool_name:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if tool_name != tool_name.strip() or _has_control(tool_name):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return (
        f"mcp:{_encode_selector_component(server_id)}:"
        f"{_encode_selector_component(tool_name)}"
    )


def canonical_selectors(
    value: Sequence[str],
    *,
    field_name: str,
    mcp: bool = False,
) -> "tuple[str, ...]":
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} must be an array of strings")
    selectors: set[str] = set()
    has_all = False
    for raw in value:
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        if raw == "*":
            has_all = True
            continue
        workspace_selector = raw in {
            "file:*",
            "file:read",
            "file:write",
            "terminal:*",
        }
        if (
            raw.startswith(("file:", "terminal:"))
            and not workspace_selector
        ):
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                f"{field_name} contains an unsupported selector",
            )
        if workspace_selector and not mcp:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                f"{field_name} contains an invalid selector",
            )
        if not mcp and "*" in raw:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        selector = raw
        parsed = parse_mcp_tool_selector(raw) if mcp else None
        if parsed is not None:
            server_id, tool_name = parsed
            selector = (
                mcp_server_selector(server_id)
                if tool_name is None
                else mcp_tool_selector(server_id, tool_name)
            )
        elif raw.startswith("mcp__") or ("*" in raw and not workspace_selector):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, f"{field_name} contains an invalid selector")
        selectors.add(selector)
    ordered = tuple(sorted(selectors))
    return ("*", *ordered) if has_all else ordered


def _encode_selector_component(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    try:
        payload = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    return "".join(
        chr(byte) if byte in _SELECTOR_SAFE_BYTES else f"%{byte:02X}"
        for byte in payload
    )


def _decode_selector_component(value: str) -> str:
    if not value:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    payload = bytearray()
    index = 0
    while index < len(value):
        character = value[index]
        if character == "%":
            if index + 2 >= len(value):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            encoded = value[index + 1 : index + 3]
            if re.fullmatch(r"[0-9A-Fa-f]{2}", encoded) is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            payload.append(int(encoded, 16))
            index += 3
            continue
        try:
            encoded = character.encode("ascii")
        except UnicodeEncodeError as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if encoded[0] not in _SELECTOR_SAFE_BYTES:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        payload.extend(encoded)
        index += 1
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error


def _has_control(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


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

    DEFAULT_TOOL_RETRIES: ClassVar[int] = 10
    DEFAULT_OUTPUT_RETRIES: ClassVar[int] = 3

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
    tool_retries: int = DEFAULT_TOOL_RETRIES
    output_retries: int = DEFAULT_OUTPUT_RETRIES
    description: "str | None" = None
    preload_skills: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        validate_logical_id(self.id)
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
        if "*" not in allow_skills and any(
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
        validate_logical_id(self.id)
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
        if self.kind != "agent":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            validate_logical_id(self.id)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
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
        if description is not None and not isinstance(description, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls("agent", identity, description)


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    command: str
    args: "tuple[str, ...]" = ()
    resource_root: "AssetKey | None" = None

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
        if self.resource_root is not None and not isinstance(self.resource_root, AssetKey):
            raise TypeError("MCP resource root must be an AssetKey")


__all__ = [
    "AgentSpec",
    "AgentUsageLimits",
    "MCPServerSpec",
    "SkillSpec",
    "SubagentRef",
    "ThinkingEffort",
    "ThinkingValue",
    "canonical_selectors",
    "mcp_server_selector",
    "mcp_tool_selector",
    "normalize_thinking",
    "parse_mcp_tool_selector",
]
