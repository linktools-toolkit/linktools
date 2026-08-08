#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset codecs for the Spec DTOs."""

import json

from ..asset.codec import AssetCodec
from ..asset.model import AssetKey
from ..core.errors import ErrorCode, AIError
from .model import AgentFeatureRef, AgentSpec, PromptSpec


class AgentSpecCodec(AssetCodec[AgentSpec]):
    @property
    def kind(self) -> str:
        return "agent"

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

    def validate_key(self, key: "AssetKey", value: AgentSpec) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise ValueError("agent spec key mismatch")


class PromptSpecCodec(AssetCodec[PromptSpec]):
    @property
    def kind(self) -> str:
        return "prompt"

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

    def validate_key(self, key: "AssetKey", value: PromptSpec) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise ValueError("prompt spec key mismatch")


__all__ = ["AgentSpecCodec", "PromptSpecCodec"]
