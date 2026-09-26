#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoring adapters for Agent, Skill, and MCP declarations."""

from collections.abc import Mapping

from ..core import JsonValue, normalize_json_value, validate_logical_id
from ..errors import AIError, ErrorCode
from ._codec import (
    AgentSpecCodec,
    MCPServerSpecCodec,
    SkillSpecCodec,
    _decode_author_revision,
    _decode_mcp_author_server,
    _parse_skill_markdown,
    _skill_revision,
    decode_author_json_mapping,
    decode_author_yaml_mapping,
)
from ._contract import AgentSpec, MCPServerSpec, SkillSpec


class AgentSpecAdapter:
    """Adapt AGENT.md authoring input to AgentSpec."""

    def parse_markdown(self, data: bytes) -> dict[str, object]:
        if not isinstance(data, bytes):
            raise TypeError("Agent Markdown data must be bytes")
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        lines = _split_lines(text)
        if not lines or _without_line_ending(lines[0]) != "---":
            return {"system_prompt": text}
        closing = next(
            (
                index
                for index, line in enumerate(lines[1:], 1)
                if _without_line_ending(line) == "---"
            ),
            None,
        )
        if closing is None:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        frontmatter = decode_author_yaml_mapping(
            "".join(lines[1:closing]).encode("utf-8")
        )
        canonical = _canonicalize_agent_fields(frontmatter)
        canonical["system_prompt"] = "".join(lines[closing + 1 :])
        return canonical

    def from_mapping(
        self,
        payload: Mapping[str, object],
        *,
        logical_id: str,
        defaults: "Mapping[str, object] | None" = None,
    ) -> AgentSpec:
        validate_logical_id(logical_id)
        if not isinstance(payload, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if "id" in payload:
            declared_id = payload["id"]
            try:
                validate_logical_id(declared_id)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
            if declared_id != logical_id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        system_prompt = payload.get("system_prompt")
        if not isinstance(system_prompt, str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        merged = _validated_agent_defaults(defaults, logical_id)
        merged.update(payload)
        merged["id"] = logical_id
        merged["system_prompt"] = system_prompt
        return _decode_agent_mapping(merged)

    def decode_markdown(
        self,
        data: bytes,
        *,
        logical_id: str,
        defaults: "Mapping[str, object] | None" = None,
    ) -> AgentSpec:
        return self.from_mapping(
            self.parse_markdown(data),
            logical_id=logical_id,
            defaults=defaults,
        )


class SkillSpecAdapter:
    """Adapt Skill authoring input to logical SkillSpec values."""

    def from_mapping(
        self,
        raw: Mapping[str, object],
        *,
        logical_id: "str | None" = None,
    ) -> SkillSpec:
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        payload: dict[str, object] = {
            "version": raw.get("version", 1),
            "id": raw.get("id"),
            "revision": raw.get("revision", 1),
            "content": raw.get("content"),
        }
        if "description" in raw:
            payload["description"] = raw["description"]
        if "metadata" in raw:
            payload["metadata"] = raw["metadata"]
        value = SkillSpecCodec().from_payload(payload)
        return value if logical_id is None else _logical_skill(logical_id, value)

    def decode_json(
        self,
        data: bytes,
        *,
        logical_id: "str | None" = None,
    ) -> SkillSpec:
        return self.from_mapping(
            decode_author_json_mapping(data),
            logical_id=logical_id,
        )

    def decode_markdown(
        self,
        data: bytes,
        *,
        logical_id: "str | None" = None,
    ) -> SkillSpec:
        try:
            content = data.decode("utf-8")
            frontmatter = _parse_skill_markdown(content)
            metadata = dict(frontmatter.get("metadata", {}))
            revision = _skill_revision(metadata.pop("linktools-revision", 1))
            value = SkillSpec(
                frontmatter["name"],
                content,
                frontmatter["description"],
                metadata,
                revision=revision,
            )
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        return value if logical_id is None else _logical_skill(logical_id, value)

    def encode_markdown(
        self,
        value: SkillSpec,
        *,
        logical_id: "str | None" = None,
    ) -> bytes:
        local = value if logical_id is None else _storage_skill(logical_id, value)
        try:
            frontmatter = _parse_skill_markdown(local.content)
        except Exception as error:
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        metadata = dict(frontmatter.get("metadata", {}))
        revision = _skill_revision(metadata.pop("linktools-revision", 1))
        if (
            frontmatter["name"] != local.id
            or frontmatter["description"] != local.description
            or metadata != dict(local.metadata)
            or revision != local.revision
        ):
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        try:
            return local.content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


class MCPServerSpecAdapter:
    """Adapt MCP authoring formats to MCPServerSpec."""

    def decode_json(
        self,
        data: bytes,
        *,
        package_id: "str | None" = None,
    ) -> MCPServerSpec:
        return self._decode(
            decode_author_json_mapping(data),
            package_id=package_id,
        )

    def decode_yaml(
        self,
        data: bytes,
        *,
        package_id: "str | None" = None,
    ) -> MCPServerSpec:
        return self._decode(
            decode_author_yaml_mapping(data),
            package_id=package_id,
        )

    def decode_config(
        self,
        data: bytes,
        *,
        revision: int = 1,
    ) -> "tuple[MCPServerSpec, ...]":
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        raw = decode_author_json_mapping(data)
        servers = raw.get("mcpServers")
        if not isinstance(servers, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        result: list[MCPServerSpec] = []
        for identity in sorted(servers):
            value = servers[identity]
            if (
                not isinstance(identity, str)
                or not identity.strip()
                or not isinstance(value, Mapping)
            ):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            result.append(
                _decode_mcp_author_server(
                    value,
                    identity=identity,
                    revision=revision,
                    package=False,
                )
            )
        return tuple(result)

    def _decode(
        self,
        raw: Mapping[str, object],
        *,
        package_id: "str | None",
    ) -> MCPServerSpec:
        identity = raw.get("id")
        if package_id is None:
            if not isinstance(identity, str) or not identity.strip():
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            return _decode_mcp_author_server(
                raw,
                identity=identity,
                revision=_decode_author_revision(raw.get("revision", 1)),
                package=False,
            )
        if "id" in raw and identity != package_id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return _decode_mcp_author_server(
            raw,
            identity=package_id,
            revision=_decode_author_revision(raw.get("revision", 1)),
            package=True,
        )


def _decode_agent_mapping(raw: Mapping[str, object]) -> AgentSpec:
    payload: dict[str, object] = {
        "version": raw.get("version", 1),
        "id": raw.get("id"),
        "revision": raw.get("revision", 1),
        "model": raw.get("model", "default"),
        "system_prompt": raw.get("system_prompt", ""),
        "instructions": raw.get("instructions", []),
        "allow_tools": raw.get("allow_tools", ["*"]),
        "allow_skills": raw.get("allow_skills", ["*"]),
        "allow_subagents": raw.get("allow_subagents", ["*"]),
    }
    if "description" in raw:
        payload["description"] = raw["description"]
    if "metadata" in raw:
        payload["metadata"] = raw["metadata"]
    return AgentSpecCodec().from_payload(payload)


def _validated_agent_defaults(
    defaults: "Mapping[str, object] | None",
    logical_id: str,
) -> dict[str, object]:
    if defaults is None:
        return {}
    if not isinstance(defaults, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        normalized = normalize_json_value(dict(defaults))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if {"id", "version", "system_prompt"}.intersection(normalized):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    _decode_agent_mapping(
        {
            **normalized,
            "id": logical_id,
            "system_prompt": "",
        }
    )
    return normalized


def _logical_skill(logical_id: str, value: SkillSpec) -> SkillSpec:
    validate_logical_id(logical_id)
    if value.id != logical_id.rsplit("/", 1)[-1]:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
    return SkillSpec(
        logical_id,
        value.content,
        value.description,
        value.metadata,
        revision=value.revision,
    )


def _storage_skill(logical_id: str, value: SkillSpec) -> SkillSpec:
    validate_logical_id(logical_id)
    if value.id != logical_id:
        raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
    return SkillSpec(
        logical_id.rsplit("/", 1)[-1],
        value.content,
        value.description,
        value.metadata,
        revision=value.revision,
    )


def _canonicalize_agent_fields(
    payload: Mapping[str, object],
) -> dict[str, object]:
    aliases = {
        "allow-tools": "allow_tools",
        "allow-skills": "allow_skills",
        "allow-subagents": "allow_subagents",
    }
    result: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        canonical = aliases.get(key, key)
        if canonical in result:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        result[canonical] = value
    return normalize_json_value(result)


def _without_line_ending(value: str) -> str:
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith(("\r", "\n")):
        return value[:-1]
    return value


def _split_lines(value: str) -> list[str]:
    lines: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        if value[index] not in "\r\n":
            index += 1
            continue
        end = index + 1
        if value[index] == "\r" and value[end : end + 1] == "\n":
            end += 1
        lines.append(value[start:end])
        start = end
        index = end
    if start < len(value):
        lines.append(value[start:])
    return lines


__all__ = ["AgentSpecAdapter", "MCPServerSpecAdapter", "SkillSpecAdapter"]
