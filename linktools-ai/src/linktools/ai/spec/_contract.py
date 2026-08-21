#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable declaration contracts for Agent and Asset specifications."""

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import cast

from ..core import JsonValue, canonical_json_bytes, canonical_string_tuple


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
        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool) or value <= 0)
            for value in values
        ):
            raise ValueError("usage limits must be positive integers or None")


class _ImmutableJsonMapping(Mapping[str, JsonValue]):
    """Store a JSON object as canonical bytes and decode fresh values on read."""

    __slots__ = ("_payload",)

    def __init__(self, value: Mapping[str, JsonValue]) -> None:
        normalized = _normalize_mapping(value)
        self._payload = canonical_json_bytes(normalized)

    def __getitem__(self, key: str) -> JsonValue:
        return self._decode()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decode())

    def __len__(self) -> int:
        return len(self._decode())

    def _decode(self) -> "dict[str, JsonValue]":
        return cast("dict[str, JsonValue]", json.loads(self._payload.decode("utf-8")))


def _normalize_mapping(value: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("JSON object keys must be non-empty strings")
        normalized[key] = _normalize_value(item)
    return normalized


def _normalize_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return _normalize_mapping(cast("Mapping[str, JsonValue]", value))
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Durable, runtime-independent declaration of one Agent."""

    id: str
    revision: int
    model: str
    system_prompt: str = ""
    instructions: "tuple[str, ...]" = ()
    allow_tools: "tuple[str, ...]" = ("*",)
    allow_skills: "tuple[str, ...]" = ("*",)
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)
    usage_limits: "AgentUsageLimits | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("agent id must be a non-empty string")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("agent revision must be a positive integer")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent model must be a non-empty string")
        if not isinstance(self.system_prompt, str):
            raise ValueError("agent system prompt must be a string")
        instructions = tuple(self.instructions)
        if any(not isinstance(item, str) for item in instructions):
            raise ValueError("agent instructions must be strings")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "allow_tools", canonical_string_tuple(self.allow_tools, field="allow_tools"))
        object.__setattr__(self, "allow_skills", canonical_string_tuple(self.allow_skills, field="allow_skills"))
        object.__setattr__(self, "metadata", _ImmutableJsonMapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    revision: int
    content: str

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1:
            raise ValueError("skill spec is incomplete")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    revision: int
    command: str
    args: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1 or not self.command.strip():
            raise ValueError("MCP server spec is incomplete")
        object.__setattr__(self, "args", tuple(self.args))


__all__ = ["AgentSpec", "AgentUsageLimits", "MCPServerSpec", "SkillSpec"]
