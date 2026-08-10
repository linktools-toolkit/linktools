#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable declaration contracts for Agents, Prompts, and capabilities."""

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import cast

from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode


class _ImmutableJsonMapping(Mapping[str, JsonValue]):
    """Store a JSON object as canonical bytes and decode fresh values on read."""

    __slots__ = ("_payload",)

    def __init__(self, value: Mapping[str, JsonValue], *, reject_secrets: bool = False) -> None:
        normalized = _normalize_mapping(value, reject_secrets=reject_secrets)
        self._payload = canonical_json_bytes(normalized)

    def __getitem__(self, key: str) -> JsonValue:
        return self._decode()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decode())

    def __len__(self) -> int:
        return len(self._decode())

    def _decode(self) -> "dict[str, JsonValue]":
        return cast("dict[str, JsonValue]", json.loads(self._payload.decode("utf-8")))


def _normalize_mapping(value: Mapping[str, JsonValue], *, reject_secrets: bool) -> "dict[str, JsonValue]":
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("JSON object keys must be non-empty strings")
        if reject_secrets and _looks_secret(key):
            raise ValueError("declaration metadata cannot contain secret fields")
        normalized[key] = _normalize_value(item, reject_secrets=reject_secrets)
    return normalized


def _normalize_value(value: object, *, reject_secrets: bool) -> JsonValue:
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [_normalize_value(item, reject_secrets=reject_secrets) for item in value]
    if isinstance(value, Mapping):
        return _normalize_mapping(cast("Mapping[str, JsonValue]", value), reject_secrets=reject_secrets)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _looks_secret(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in ("secret", "credential", "token", "password", "api_key", "apikey"))


def _mapping_json(value: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    return _normalize_mapping(value, reject_secrets=False)


@dataclass(frozen=True, slots=True)
class AgentCapabilityRef:
    provider: str
    id: str
    revision: "int | None" = None
    required: bool = True
    config: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.id.strip() or (self.revision is not None and self.revision < 1):
            raise ValueError("agent capability reference is invalid")
        object.__setattr__(self, "config", _ImmutableJsonMapping(self.config))


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    revision: int
    model: str
    capabilities: "tuple[AgentCapabilityRef, ...]"
    output_schema: str
    output_schema_revision: int
    instructions: "tuple[str, ...]" = ()
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1 or not self.model.strip() or not self.output_schema.strip() or self.output_schema_revision < 1:
            raise ValueError("agent spec is incomplete")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "metadata", _ImmutableJsonMapping(self.metadata, reject_secrets=True))
        unique: dict[tuple[str, str], AgentCapabilityRef] = {}
        for capability in self.capabilities:
            key = capability.provider, capability.id
            previous = unique.get(key)
            if previous is not None:
                if _capability_digest(previous) != _capability_digest(capability):
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                continue
            unique[key] = capability
        object.__setattr__(self, "capabilities", tuple(unique.values()))


@dataclass(frozen=True, slots=True)
class PromptSpec:
    id: str
    revision: int
    system: str
    instructions: "tuple[str, ...]"
    variables: "tuple[str, ...]"

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1:
            raise ValueError("prompt spec is incomplete")
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "variables", tuple(self.variables))


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


def _capability_digest(capability: AgentCapabilityRef) -> str:
    return canonical_sha256(
        {
            "provider": capability.provider,
            "id": capability.id,
            "revision": capability.revision,
            "required": capability.required,
            "config": _mapping_json(capability.config),
        }
    )


__all__ = ["AgentCapabilityRef", "AgentSpec", "MCPServerSpec", "PromptSpec", "SkillSpec"]
