#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned codecs for durable declaration DTOs."""

import json
import re
from collections.abc import Mapping
from typing import Protocol, TypeVar, cast

import yaml

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._contract import AgentSpec, AgentUsageLimits, MCPServerSpec, SkillSpec, normalize_thinking

SpecT = TypeVar("SpecT")
_VERSION = 1
_AGENT_FIELDS = frozenset(
    {
        "version",
        "id",
        "model",
        "system_prompt",
        "instructions",
        "allow_tools",
        "allow_skills",
        "allow_subagents",
        "usage_limits",
        "planning",
        "thinking",
    }
)
_SKILL_FIELDS = frozenset({"version", "id", "content"})
_MCP_FIELDS = frozenset({"version", "id", "command", "args"})
_USAGE_LIMIT_FIELDS = (
    "model_requests",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


class SpecCodec(Protocol[SpecT]):
    def encode(self, value: SpecT) -> bytes: ...
    def decode(self, data: bytes) -> SpecT: ...


class AgentSpecCodec:
    def to_payload(self, value: AgentSpec) -> "dict[str, JsonValue]":
        """Return the canonical semantic v1 payload used by durable identity."""
        if not isinstance(value, AgentSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent spec is invalid")
        return {
            "version": _VERSION,
            "id": value.id,
            "model": value.model,
            "system_prompt": value.system_prompt,
            "instructions": list(value.instructions),
            "allow_tools": list(value.allow_tools),
            "allow_skills": list(value.allow_skills),
            "allow_subagents": list(value.allow_subagents),
            "usage_limits": None
            if value.usage_limits is None
            else {
                "model_requests": value.usage_limits.model_requests,
                "tool_calls": value.usage_limits.tool_calls,
                "input_tokens": value.usage_limits.input_tokens,
                "output_tokens": value.usage_limits.output_tokens,
                "total_tokens": value.usage_limits.total_tokens,
            },
            "planning": value.planning,
            "thinking": value.thinking,
        }

    def to_wire_payload(self, value: AgentSpec) -> "dict[str, JsonValue]":
        payload = dict(value._extensions)
        payload.update(self.to_payload(value))
        return payload

    def from_payload(self, raw: Mapping[str, object]) -> AgentSpec:
        _require_v1(raw)
        identity = raw.get("id")
        model = raw.get("model", "default")
        system_prompt = raw.get("system_prompt", "")
        instructions = raw.get("instructions", [])
        allow_tools = raw.get("allow_tools", ["*"])
        allow_skills = raw.get("allow_skills", ["*"])
        allow_subagents = raw.get("allow_subagents", ["*"])
        planning = raw.get("planning", False)
        thinking = raw.get("thinking", False)
        if not isinstance(identity, str) or not identity.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent id must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent model must be a non-empty string")
        if not isinstance(system_prompt, str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "system_prompt must be a string")
        if not isinstance(instructions, list) or any(not isinstance(item, str) for item in instructions):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "instructions must be a string array")
        for name, value in (
            ("allow_tools", allow_tools),
            ("allow_skills", allow_skills),
            ("allow_subagents", allow_subagents),
        ):
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, f"{name} must be a string array")
        if not isinstance(planning, bool):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "planning must be bool")
        try:
            normalized_thinking = normalize_thinking(thinking)
            return AgentSpec(
                id=identity,
                model=model,
                system_prompt=system_prompt,
                instructions=tuple(instructions),
                allow_tools=tuple(cast("list[str]", allow_tools)),
                allow_skills=tuple(cast("list[str]", allow_skills)),
                allow_subagents=tuple(cast("list[str]", allow_subagents)),
                usage_limits=_decode_usage_limits(raw.get("usage_limits")),
                planning=planning,
                thinking=normalized_thinking,
                _extensions=_extensions(raw, _AGENT_FIELDS),
            )
        except AIError as error:
            if error.code in {ErrorCode.STORAGE_INTEGRITY_ERROR, ErrorCode.STORAGE_VERSION_UNSUPPORTED}:
                raise
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent spec is invalid") from error
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent spec is invalid") from error

    def encode(self, value: AgentSpec) -> bytes:
        return _encode(cast("dict[str, object]", self.to_wire_payload(value)))

    def decode(self, data: bytes) -> AgentSpec:
        return self.from_payload(_decode(data))


class SkillSpecCodec:
    def to_payload(self, value: SkillSpec) -> "dict[str, JsonValue]":
        if not isinstance(value, SkillSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid")
        return {"version": _VERSION, "id": value.id, "content": value.content}

    def to_wire_payload(self, value: SkillSpec) -> "dict[str, JsonValue]":
        payload = dict(value._extensions)
        payload.update(self.to_payload(value))
        return payload

    def from_payload(self, raw: Mapping[str, object]) -> SkillSpec:
        _require_v1(raw)
        identity = raw.get("id")
        content = raw.get("content")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(content, str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid")
        try:
            return SkillSpec(identity, content, _extensions(raw, _SKILL_FIELDS))
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid") from error

    def encode(self, value: SkillSpec) -> bytes:
        return _encode(cast("dict[str, object]", self.to_wire_payload(value)))

    def decode(self, data: bytes) -> SkillSpec:
        return self.from_payload(_decode(data))


class SkillMarkdownSpecCodec:
    """Decode standard SKILL.md documents without rewriting their text."""

    def encode(self, value: SkillSpec) -> bytes:
        try:
            frontmatter = _parse_skill_markdown(value.content)
        except Exception as error:
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if frontmatter["name"] != value.id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        try:
            return value.content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error

    def decode(self, data: bytes) -> SkillSpec:
        try:
            content = data.decode("utf-8")
            frontmatter = _parse_skill_markdown(content)
            return SkillSpec(cast(str, frontmatter["name"]), content)
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
        return SkillSpec(logical_id, value.content, value._extensions)

    def to_storage(self, logical_id: str, value: SkillSpec) -> SkillSpec:
        if value.id != logical_id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return SkillSpec(logical_id.rsplit("/", 1)[-1], value.content, value._extensions)


def retarget_skill_markdown(content: str, local_name: str) -> str:
    """Rewrite the canonical frontmatter name during a logical rename."""
    lines = content.splitlines(keepends=True)
    closing = next((index for index, line in enumerate(lines[1:], 1) if line.rstrip("\r\n") == "---"), None)
    if closing is None:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    frontmatter = _parse_skill_markdown(content)
    frontmatter["name"] = local_name
    encoded = yaml.safe_dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=True)
    return f"---\n{encoded}---\n{''.join(lines[closing + 1:])}"


class MCPServerSpecCodec:
    def to_payload(self, value: MCPServerSpec) -> "dict[str, JsonValue]":
        if not isinstance(value, MCPServerSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server spec is invalid")
        return {"version": _VERSION, "id": value.id, "command": value.command, "args": list(value.args)}

    def to_wire_payload(self, value: MCPServerSpec) -> "dict[str, JsonValue]":
        payload = dict(value._extensions)
        payload.update(self.to_payload(value))
        return payload

    def from_payload(self, raw: Mapping[str, object]) -> MCPServerSpec:
        _require_v1(raw)
        identity = raw.get("id")
        command = raw.get("command")
        args = raw.get("args", [])
        if not isinstance(identity, str) or not identity.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server id must be a non-empty string")
        if not isinstance(command, str) or not command.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server command must be a non-empty string")
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server args must be a string array")
        try:
            return MCPServerSpec(identity, command, tuple(cast("list[str]", args)), _extensions(raw, _MCP_FIELDS))
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server spec is invalid") from error

    def encode(self, value: MCPServerSpec) -> bytes:
        return _encode(cast("dict[str, object]", self.to_wire_payload(value)))

    def decode(self, data: bytes) -> MCPServerSpec:
        return self.from_payload(_decode(data))


def _encode(value: "dict[str, object]") -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _decode(data: bytes) -> "dict[str, object]":
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _require_v1(raw: Mapping[str, object]) -> None:
    version = raw.get("version")
    if version is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _extensions(raw: Mapping[str, object], known: frozenset[str]) -> "dict[str, JsonValue]":
    result: dict[str, JsonValue] = {}
    for key, value in raw.items():
        if key in known:
            continue
        if not isinstance(key, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        result[key] = cast(JsonValue, value)
    return result


def _decode_usage_limits(value: object) -> "AgentUsageLimits | None":
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "usage_limits must be an object or null")
    unknown = set(value).difference(_USAGE_LIMIT_FIELDS)
    if unknown:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "usage_limits contains unknown fields")
    kwargs = {name: value[name] for name in _USAGE_LIMIT_FIELDS if name in value}
    try:
        return AgentUsageLimits(**kwargs)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "usage_limits values are invalid") from error


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


def _parse_skill_markdown(content: str) -> dict[str, object]:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    closing = next((index for index, line in enumerate(lines[1:], 1) if line.rstrip("\r\n") == "---"), None)
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
        if key in frontmatter and (not isinstance(frontmatter[key], str) or not 1 <= len(cast(str, frontmatter[key])) <= maximum):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    for key in ("license", "allowed-tools"):
        if key in frontmatter and not isinstance(frontmatter[key], str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    metadata = frontmatter.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, Mapping)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if isinstance(metadata, Mapping) and "linktools-revision" in metadata:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return frontmatter


__all__ = [
    "AgentSpecCodec",
    "MCPServerSpecCodec",
    "SkillMarkdownSpecAdapter",
    "SkillMarkdownSpecCodec",
    "SkillSpecCodec",
    "SpecCodec",
    "retarget_skill_markdown",
]
