#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bytes codecs for declaration DTOs."""

import json
from typing import Protocol, TypeVar, cast

from ..errors import AIError, ErrorCode
from ._contract import (
    AgentCapabilityRef,
    AgentSpec,
    MCPServerSpec,
    PromptSpec,
    SkillSpec,
)

SpecT = TypeVar("SpecT")


class SpecCodec(Protocol[SpecT]):
    def encode(self, value: SpecT) -> bytes: ...
    def decode(self, data: bytes) -> SpecT: ...


class AgentSpecCodec:
    def encode(self, value: AgentSpec) -> bytes:
        return _encode(
            {
                "id": value.id,
                "revision": value.revision,
                "model": value.model,
                "capabilities": [
                    {
                        "provider": item.provider,
                        "id": item.id,
                        "revision": item.revision,
                        "required": item.required,
                        "config": dict(item.config),
                    }
                    for item in value.capabilities
                ],
                "output_schema": value.output_schema,
                "output_schema_revision": value.output_schema_revision,
                "instructions": list(value.instructions),
                "metadata": dict(value.metadata),
            }
        )

    def decode(self, data: bytes) -> AgentSpec:
        raw = _decode(data)
        if "output_schema_revision" not in raw:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_REVISION_REQUIRED)
        capabilities = tuple(
            AgentCapabilityRef(
                str(item["provider"]),
                str(item["id"]),
                item.get("revision"),
                bool(item.get("required", True)),
                cast("dict[str, object]", item.get("config", {})),
            )
            for item in cast("list[dict[str, object]]", raw["capabilities"])
        )
        return AgentSpec(
            str(raw["id"]),
            int(raw["revision"]),
            str(raw["model"]),
            capabilities,
            str(raw["output_schema"]),
            int(raw["output_schema_revision"]),
            tuple(str(item) for item in cast("list[object]", raw.get("instructions", []))),
            cast("dict[str, object]", raw.get("metadata", {})),
        )


class PromptSpecCodec:
    def encode(self, value: PromptSpec) -> bytes:
        return _encode(
            {
                "id": value.id,
                "revision": value.revision,
                "system": value.system,
                "instructions": list(value.instructions),
            }
        )

    def decode(self, data: bytes) -> PromptSpec:
        raw = _decode(data)
        if "variables" in raw:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "PromptSpec.variables is unsupported")
        return PromptSpec(
            str(raw["id"]),
            int(raw["revision"]),
            str(raw["system"]),
            tuple(str(item) for item in cast("list[object]", raw["instructions"])),
        )


class SkillSpecCodec:
    def encode(self, value: SkillSpec) -> bytes:
        return _encode({"id": value.id, "revision": value.revision, "content": value.content})

    def decode(self, data: bytes) -> SkillSpec:
        raw = _decode(data)
        return SkillSpec(str(raw["id"]), int(raw["revision"]), str(raw["content"]))


class MCPServerSpecCodec:
    def encode(self, value: MCPServerSpec) -> bytes:
        return _encode({"id": value.id, "revision": value.revision, "command": value.command, "args": list(value.args)})

    def decode(self, data: bytes) -> MCPServerSpec:
        raw = _decode(data)
        return MCPServerSpec(
            str(raw["id"]),
            int(raw["revision"]),
            str(raw["command"]),
            tuple(str(item) for item in cast("list[object]", raw.get("args", []))),
        )


def _encode(value: "dict[str, object]") -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _decode(data: bytes) -> "dict[str, object]":
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("spec payload must be a JSON object")
    return value


__all__ = ["AgentSpecCodec", "MCPServerSpecCodec", "PromptSpecCodec", "SkillSpecCodec", "SpecCodec"]
