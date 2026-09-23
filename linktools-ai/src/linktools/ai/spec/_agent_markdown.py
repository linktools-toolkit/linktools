#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict authoring adapter for Agent Markdown declarations."""

from collections.abc import Mapping
from typing import cast

from ..core import normalize_json_value, validate_logical_id
from ..errors import AIError, ErrorCode
from ._codec import AgentSpecCodec, decode_author_yaml_mapping
from ._contract import AgentSpec

_AGENT_FIELDS = frozenset(
    {
        "allow_capabilities",
        "allow_skills",
        "allow_subagents",
        "allow_tools",
        "description",
        "id",
        "instructions",
        "model",
        "output_retries",
        "planning",
        "preload_skills",
        "thinking",
        "tool_retries",
        "usage_limits",
    }
)
_USAGE_LIMIT_FIELDS = frozenset(
    {
        "input_tokens",
        "model_requests",
        "output_tokens",
        "tool_calls",
        "total_tokens",
    }
)
_FORBIDDEN_FIELDS = frozenset({"system_prompt", "system-prompt", "version"})


class AgentMarkdownSpecCodec:
    """Parse and decode one `AGENT.md` authoring document."""

    def parse(self, data: bytes) -> dict[str, object]:
        """Return canonical standard fields and the original Markdown body."""
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
        frontmatter_bytes = "".join(lines[1:closing]).encode("utf-8")
        frontmatter = decode_author_yaml_mapping(frontmatter_bytes)
        canonical = _canonicalize_agent_fields(frontmatter)
        if _FORBIDDEN_FIELDS.intersection(canonical):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        canonical["system_prompt"] = "".join(lines[closing + 1 :])
        return canonical

    def from_payload(
        self,
        payload: Mapping[str, object],
        *,
        logical_id: str,
        defaults: "Mapping[str, object] | None" = None,
    ) -> AgentSpec:
        """Decode canonical partial fields with the given logical package ID."""
        validate_logical_id(logical_id)
        if not isinstance(payload, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if "version" in payload:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if "id" in payload:
            declared_id = payload["id"]
            try:
                validate_logical_id(cast(str, declared_id))
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
            if declared_id != logical_id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        system_prompt = payload.get("system_prompt")
        if not isinstance(system_prompt, str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if "usage_limits" in payload:
            _validate_usage_limit_fields(payload["usage_limits"])

        resolved_defaults = _validated_defaults(defaults, logical_id)
        merged = dict(resolved_defaults)
        merged.update(payload)
        merged["id"] = logical_id
        merged["system_prompt"] = system_prompt
        return AgentSpecCodec().from_author_payload(merged)

    def decode(
        self,
        data: bytes,
        *,
        logical_id: str,
        defaults: "Mapping[str, object] | None" = None,
    ) -> AgentSpec:
        """Parse and decode one Agent Markdown declaration."""
        return self.from_payload(
            self.parse(data),
            logical_id=logical_id,
            defaults=defaults,
        )


def _canonicalize_agent_fields(
    payload: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in payload.items():
        canonical = _canonical_field(key, _AGENT_FIELDS)
        if canonical in result:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        result[canonical] = value
    usage_limits = result.get("usage_limits")
    if isinstance(usage_limits, Mapping):
        result["usage_limits"] = _canonicalize_usage_limits(usage_limits)
    return cast(dict[str, object], normalize_json_value(result))


def _canonicalize_usage_limits(
    value: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        canonical = _canonical_field(key, _USAGE_LIMIT_FIELDS)
        if canonical in result:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        result[canonical] = item
    return result


def _canonical_field(key: object, fields: frozenset[str]) -> str:
    if not isinstance(key, str):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    for field in fields:
        if key == field or key == field.replace("_", "-"):
            return field
    return key


def _validated_defaults(
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
    probe: dict[str, object] = {
        **normalized,
        "id": logical_id,
        "system_prompt": "",
    }
    AgentSpecCodec().from_author_payload(probe)
    return normalized


def _validate_usage_limit_fields(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        return
    if any(
        not isinstance(key, str)
        or _canonical_field(key, _USAGE_LIMIT_FIELDS) not in _USAGE_LIMIT_FIELDS
        for key in value
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)


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


__all__ = ["AgentMarkdownSpecCodec"]
