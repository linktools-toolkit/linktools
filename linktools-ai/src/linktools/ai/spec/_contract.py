#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable Agent and Prompt contracts and their asset codecs."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from ..asset import AssetCodec
from ..asset import AssetKey, AssetValue
from ..errors import ErrorCode, AIError
from ..core import canonical_sha256
from ..core import JsonValue


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
    output_schema_revision: int
    instructions: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1 or not self.model.strip() or not self.output_schema.strip() or self.output_schema_revision < 1:
            raise ValueError("agent spec is incomplete")
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "instructions", tuple(self.instructions))
        unique: dict[tuple[str, str], AgentFeatureRef] = {}
        for feature in self.features:
            key = (feature.kind, feature.id)
            previous = unique.get(key)
            if previous is not None:
                if _feature_digest(previous) != _feature_digest(feature):
                    raise AIError(ErrorCode.FEATURE_CONFLICT)
                continue
            unique[key] = feature
        object.__setattr__(self, "features", tuple(unique.values()))

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


def _feature_digest(feature: AgentFeatureRef) -> str:
    return canonical_sha256({"kind": feature.kind, "id": feature.id, "revision": feature.revision, "required": feature.required, "config": dict(feature.config)})


class AgentSpecCodec(AssetCodec[AgentSpec]):
    @property
    def kind(self) -> str:
        return "agent"

    @property
    def primary_path(self) -> str:
        return "agent.md"

    @property
    def value_type(self) -> 'type[AgentSpec]':
        return AgentSpec

    @property
    def fingerprint(self) -> str:
        return "agent-spec"

    def encode(self, value: AgentSpec) -> bytes:
        return json.dumps({"id": value.id, "revision": value.revision, "model": value.model, "features": [{"kind": feature.kind, "id": feature.id, "revision": feature.revision, "required": feature.required, "config": dict(feature.config)} for feature in value.features], "output_schema": value.output_schema, "output_schema_revision": value.output_schema_revision, "instructions": list(value.instructions)}, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, data: bytes) -> AgentSpec:
        raw = json.loads(data.decode("utf-8"))
        if "output_schema_revision" not in raw:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_REVISION_REQUIRED)
        return AgentSpec(raw["id"], raw["revision"], raw["model"], tuple(AgentFeatureRef(item["kind"], item["id"], item.get("revision"), item.get("required", True), item.get("config", {})) for item in raw["features"]), raw["output_schema"], int(raw["output_schema_revision"]), tuple(raw.get("instructions", ())))

    def validate_key(self, key: AssetKey, value: AgentSpec) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise ValueError("agent spec key mismatch")


class PromptSpecCodec(AssetCodec[PromptSpec]):
    @property
    def kind(self) -> str:
        return "prompt"

    @property
    def primary_path(self) -> str:
        return "prompt.md"

    @property
    def value_type(self) -> 'type[PromptSpec]':
        return PromptSpec

    @property
    def fingerprint(self) -> str:
        return "prompt-spec"

    def encode(self, value: PromptSpec) -> bytes:
        return json.dumps({"id": value.id, "revision": value.revision, "system": value.system, "instructions": list(value.instructions), "variables": list(value.variables)}, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, data: bytes) -> PromptSpec:
        raw = json.loads(data.decode("utf-8"))
        return PromptSpec(raw["id"], raw["revision"], raw["system"], tuple(raw["instructions"]), tuple(raw["variables"]))

    def validate_key(self, key: AssetKey, value: PromptSpec) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise ValueError("prompt spec key mismatch")


__all__ = ["AgentFeatureRef", "AgentSpec", "AgentSpecCodec", "PromptSpec", "PromptSpecCodec"]
