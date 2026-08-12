#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bytes codecs for declaration DTOs."""

import json
import re
from collections.abc import Mapping
from typing import Protocol, TypeVar, cast

import yaml

from ..core import canonical_string_tuple
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
                "allow_tools": value.allow_tools,
                "allow_skills": value.allow_skills,
                "allow_subagents": value.allow_subagents,
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
                _strict_bool(item, "required", True),
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
            _strict_allowlist(raw, "allow_tools", ("*",)),
            _strict_allowlist(raw, "allow_skills", ("*",)),
            _strict_allowlist(raw, "allow_subagents", ()),
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


class SkillMarkdownSpecCodec:
    """Decode standard SKILL.md documents without rewriting their text."""

    def encode(self, value: SkillSpec) -> bytes:
        try:
            frontmatter, revision = _parse_skill_markdown(value.content)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if frontmatter["name"] != value.id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        if (
            not isinstance(value.revision, int)
            or isinstance(value.revision, bool)
            or value.revision < 1
            or revision != value.revision
        ):
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        try:
            return value.content.encode("utf-8")
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error

    def decode(self, data: bytes) -> SkillSpec:
        try:
            content = data.decode("utf-8")
            frontmatter, revision = _parse_skill_markdown(content)
            return SkillSpec(str(frontmatter["name"]), revision, content)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


class SkillMarkdownSpecAdapter:
    """Translate standard local skill names to complete logical ids."""

    def to_logical(self, logical_id: str, value: SkillSpec) -> SkillSpec:
        local_name = logical_id.rsplit("/", 1)[-1]
        if value.id != local_name:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return SkillSpec(logical_id, value.revision, value.content)

    def to_storage(self, logical_id: str, value: SkillSpec) -> SkillSpec:
        if value.id != logical_id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return SkillSpec(logical_id.rsplit("/", 1)[-1], value.revision, value.content)


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


def _strict_bool(raw: Mapping[str, object], name: str, default: bool) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, f"{name} must be a boolean")
    return value


def _strict_allowlist(raw: Mapping[str, object], name: str, default: "tuple[str, ...]") -> "tuple[str, ...]":
    value = raw.get(name, list(default))
    if not isinstance(value, list):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, f"{name} must be a JSON array")
    return canonical_string_tuple(value, field=name)


_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError("duplicate YAML key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _parse_skill_markdown(content: str) -> tuple[dict[str, object], int]:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing is None:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    raw = yaml.load("".join(lines[1:closing]), Loader=_StrictSafeLoader)
    if not isinstance(raw, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    frontmatter = dict(raw)
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or _SKILL_NAME.fullmatch(name) is None:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if not isinstance(description, str) or not 1 <= len(description) <= 1024:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    for key, maximum in (("compatibility", 500),):
        if key in frontmatter and (
            not isinstance(frontmatter[key], str) or not 1 <= len(cast(str, frontmatter[key])) <= maximum
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    for key in ("license", "allowed-tools"):
        if key in frontmatter and not isinstance(frontmatter[key], str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    metadata_present = "metadata" in frontmatter
    metadata = frontmatter.get("metadata")
    if metadata_present and (
        not isinstance(metadata, Mapping)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    revision_value = None if not metadata_present else cast(Mapping[str, str], metadata).get("linktools-revision")
    if revision_value is None:
        revision = 1
    elif (
        not isinstance(revision_value, str)
        or not revision_value.isdecimal()
        or int(revision_value) < 1
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    else:
        revision = int(revision_value)
    return frontmatter, revision


__all__ = [
    "AgentSpecCodec",
    "MCPServerSpecCodec",
    "PromptSpecCodec",
    "SkillMarkdownSpecAdapter",
    "SkillMarkdownSpecCodec",
    "SkillSpecCodec",
    "SpecCodec",
]
