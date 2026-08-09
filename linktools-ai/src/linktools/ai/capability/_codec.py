#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serializers for capability specifications."""

import json
from typing import Protocol, TypeVar

from ._skill import SkillSpec
from ._mcp import MCPServerSpec

CapabilityAsset = TypeVar("CapabilityAsset", SkillSpec, MCPServerSpec)


class CapabilityCodec(Protocol[CapabilityAsset]):
    def encode(self, value: CapabilityAsset) -> bytes: ...
    def decode(self, data: bytes) -> CapabilityAsset: ...


class SkillSpecCodec:
    def encode(self, value: SkillSpec) -> bytes:
        return json.dumps(
            {"id": value.id, "revision": value.revision, "content": value.content},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode(self, data: bytes) -> SkillSpec:
        raw = json.loads(data.decode("utf-8"))
        return SkillSpec(str(raw["id"]), int(raw["revision"]), str(raw["content"]))


class MCPServerSpecCodec:
    def encode(self, value: MCPServerSpec) -> bytes:
        return json.dumps(
            {"id": value.id, "revision": value.revision, "command": value.command, "args": list(value.args)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode(self, data: bytes) -> MCPServerSpec:
        raw = json.loads(data.decode("utf-8"))
        return MCPServerSpec(
            str(raw["id"]),
            int(raw["revision"]),
            str(raw["command"]),
            tuple(str(argument) for argument in raw.get("args", ())),
        )


__all__ = ["CapabilityCodec", "MCPServerSpecCodec", "SkillSpecCodec"]
