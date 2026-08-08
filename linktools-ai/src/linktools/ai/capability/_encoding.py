#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen codecs for capability assets."""

import json
from typing import Protocol, TypeVar

from ..asset import AssetCodec
from ..asset import AssetKey
from ._skill import SkillSpec
from ._mcp import MCPServerSpec

CapabilityAsset = TypeVar("CapabilityAsset", SkillSpec, MCPServerSpec)


class CapabilityCodec(AssetCodec[CapabilityAsset], Protocol[CapabilityAsset]):
    pass


class SkillSpecCodec(AssetCodec[SkillSpec]):
    @property
    def kind(self) -> str:
        return "skill"

    @property
    def value_type(self) -> 'type[SkillSpec]':
        return SkillSpec

    @property
    def fingerprint(self) -> str:
        return "skill-spec"

    def encode(self, value: SkillSpec) -> bytes:
        return json.dumps(
            {"id": value.id, "revision": value.revision, "content": value.content},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def decode(self, data: bytes) -> SkillSpec:
        raw = json.loads(data.decode("utf-8"))
        return SkillSpec(str(raw["id"]), int(raw["revision"]), str(raw["content"]))

    def validate_key(self, key: "AssetKey", value: SkillSpec) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise ValueError("skill spec key mismatch")


class MCPServerSpecCodec(AssetCodec[MCPServerSpec]):
    @property
    def kind(self) -> str:
        return "mcp"

    @property
    def value_type(self) -> 'type[MCPServerSpec]':
        return MCPServerSpec

    @property
    def fingerprint(self) -> str:
        return "mcp-server-spec"

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

    def validate_key(self, key: "AssetKey", value: MCPServerSpec) -> None:
        if value.asset_kind != key.kind or value.asset_id != key.id:
            raise ValueError("MCP server spec key mismatch")


__all__ = ["CapabilityCodec", "MCPServerSpecCodec", "SkillSpecCodec"]
