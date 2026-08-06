#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable Agent and Prompt asset DTOs."""

from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
from typing import Literal

from ..core.json import JsonValue
from ..asset.model import AssetValue


@dataclass(frozen=True, slots=True)
class AgentFeatureRef:
    kind: "Literal['tool', 'skill', 'mcp', 'subagent', 'sandbox', 'middleware']"
    id: str
    revision: "int | None" = None
    required: bool = True
    config: "Mapping[str, JsonValue]" = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.id.strip() or (self.revision is not None and self.revision < 1):
            raise ValueError("agent feature reference is invalid")
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True, slots=True)
class AgentSpec(AssetValue):
    id: str
    revision: int
    model: str
    features: "tuple[AgentFeatureRef, ...]"
    output_schema: str
    instructions: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1 or not self.model.strip() or not self.output_schema.strip():
            raise ValueError("agent spec is incomplete")
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "instructions", tuple(self.instructions))

    @property
    def asset_kind(self) -> str:
        return "agent"

    @property
    def asset_id(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class PromptSpec(AssetValue):
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

    @property
    def asset_kind(self) -> str:
        return "prompt"

    @property
    def asset_id(self) -> str:
        return self.id


__all__ = ["AgentFeatureRef", "AgentSpec", "PromptSpec"]
