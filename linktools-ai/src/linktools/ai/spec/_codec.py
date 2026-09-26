#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned codecs for durable declaration DTOs."""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, TypeVar

import yaml

from ..core import ImmutableJsonMapping, JsonValue, normalize_json_value
from ..errors import AIError, ErrorCode
from ..asset import AssetKey, AssetVersionRef
from ._contract import AgentSpec, AgentUsageLimits, MCPServerSpec, SkillSpec, normalize_thinking

SpecT = TypeVar("SpecT")
_VERSION = 1
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
        """Return the canonical resolved Agent declaration payload."""
        if not isinstance(value, AgentSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent spec is invalid")
        payload: dict[str, JsonValue] = {
            "version": 1,
            "id": value.id,
            "revision": value.revision,
            "model": value.model,
            "system_prompt": value.system_prompt,
            "instructions": list(value.instructions),
            "allow_tools": list(value.allow_tools),
            "allow_skills": list(value.allow_skills),
            "allow_subagents": list(value.allow_subagents),
            "allow_capabilities": list(value.allow_capabilities),
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
            "tool_retries": value.tool_retries,
            "output_retries": value.output_retries,
        }
        if value.preload_skills:
            payload["preload_skills"] = list(value.preload_skills)
        return payload

    def to_wire_payload(self, value: AgentSpec) -> "dict[str, JsonValue]":
        payload = self.to_payload(value)
        if value.description is not None:
            payload["description"] = value.description
        if value.metadata:
            payload["metadata"] = dict(value.metadata)
        return payload

    def from_payload(self, raw: Mapping[str, object]) -> AgentSpec:
        _require_v1(raw)
        identity = raw.get("id")
        revision = _decode_revision(raw)
        model = raw.get("model", "default")
        system_prompt = raw.get("system_prompt", "")
        instructions = raw.get("instructions", [])
        allow_tools = raw.get("allow_tools", ["*"])
        allow_skills = raw.get("allow_skills", ["*"])
        allow_subagents = raw.get("allow_subagents", ["*"])
        allow_capabilities = raw.get("allow_capabilities", ["*"])
        planning = raw.get("planning", False)
        thinking = raw.get("thinking", False)
        tool_retries = raw.get("tool_retries", AgentSpec.DEFAULT_TOOL_RETRIES)
        output_retries = raw.get("output_retries", AgentSpec.DEFAULT_OUTPUT_RETRIES)
        description = raw.get("description")
        metadata = raw.get("metadata", {})
        preload_skills: object = raw.get("preload_skills", [])
        if not isinstance(preload_skills, list) or any(
            not isinstance(item, str) for item in preload_skills
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "preload_skills must be a string array")
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
            ("allow_capabilities", allow_capabilities),
        ):
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, f"{name} must be a string array")
        if not isinstance(planning, bool):
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                "planning must be bool",
            )
        for name, value in (
            ("tool_retries", tool_retries),
            ("output_retries", output_retries),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AIError(
                    ErrorCode.OUTPUT_CONTRACT_INVALID,
                    f"{name} must be a non-negative integer",
                )
        if description is not None and (
            not isinstance(description, str) or not 1 <= len(description) <= 1024
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "description must contain 1..1024 characters")
        try:
            normalized_thinking = normalize_thinking(thinking)
            return AgentSpec(
                id=identity,
                model=model,
                system_prompt=system_prompt,
                instructions=tuple(instructions),
                allow_tools=tuple(allow_tools),
                allow_skills=tuple(allow_skills),
                allow_subagents=tuple(allow_subagents),
                allow_capabilities=tuple(allow_capabilities),
                usage_limits=_decode_usage_limits(raw.get("usage_limits")),
                planning=planning,
                thinking=normalized_thinking,
                tool_retries=tool_retries,
                output_retries=output_retries,
                description=description,
                preload_skills=tuple(preload_skills),
                metadata=metadata,
                revision=revision,
            )
        except AIError as error:
            if error.code in {ErrorCode.STORAGE_INTEGRITY_ERROR, ErrorCode.STORAGE_VERSION_UNSUPPORTED}:
                raise
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent spec is invalid") from error
        except (TypeError, ValueError, UnicodeError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "agent spec is invalid") from error

    def from_author_payload(self, raw: Mapping[str, object]) -> AgentSpec:
        """Decode the stable Agent authoring fields and ignore unrelated keys."""
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        payload: dict[str, object] = {
            "version": raw.get("version", _VERSION),
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
        return self.from_payload(payload)

    def decode_author_mapping(self, data: bytes) -> dict[str, object]:
        """Decode an author JSON mapping while rejecting malformed syntax and duplicate keys."""
        return decode_author_json_mapping(data)

    def encode(self, value: AgentSpec) -> bytes:
        return _encode(self.to_wire_payload(value))

    def decode(self, data: bytes) -> AgentSpec:
        return self.from_payload(_decode(data))


class SkillSpecCodec:
    def to_payload(self, value: SkillSpec) -> "dict[str, JsonValue]":
        """Return the canonical resolved Skill declaration payload."""
        if not isinstance(value, SkillSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid")
        payload: dict[str, JsonValue] = {
            "version": 1,
            "id": value.id,
            "revision": value.revision,
            "content": SkillMarkdownSpecCodec().model_content(value.content),
        }
        if value.description is not None:
            payload["description"] = value.description
        return payload

    def to_wire_payload(self, value: SkillSpec) -> "dict[str, JsonValue]":
        if not isinstance(value, SkillSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid")
        payload: dict[str, JsonValue] = {
            "version": 1,
            "id": value.id,
            "revision": value.revision,
            "content": value.content,
        }
        if value.description is not None:
            payload["description"] = value.description
        if value.metadata:
            payload["metadata"] = dict(value.metadata)
        return payload

    def from_payload(self, raw: Mapping[str, object]) -> SkillSpec:
        _require_v1(raw)
        identity = raw.get("id")
        revision = _decode_revision(raw)
        content = raw.get("content")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(content, str):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid")
        description = raw.get("description")
        metadata = raw.get("metadata", {})
        if description is not None and (
            not isinstance(description, str) or not 1 <= len(description) <= 1024
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill description must contain 1..1024 characters")
        try:
            return SkillSpec(
                identity,
                content,
                description,
                metadata,
                revision=revision,
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "skill spec is invalid") from error

    def from_author_payload(self, raw: Mapping[str, object]) -> SkillSpec:
        """Decode the stable Skill authoring fields and ignore unrelated keys."""
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        payload: dict[str, object] = {
            "version": raw.get("version", _VERSION),
            "id": raw.get("id"),
            "revision": raw.get("revision", 1),
            "content": raw.get("content"),
        }
        if "description" in raw:
            payload["description"] = raw["description"]
        if "metadata" in raw:
            payload["metadata"] = raw["metadata"]
        return self.from_payload(payload)

    def decode_author(self, data: bytes) -> SkillSpec:
        """Decode one strict JSON Skill declaration."""
        return self.from_author_payload(decode_author_json_mapping(data))

    def encode(self, value: SkillSpec) -> bytes:
        return _encode(self.to_wire_payload(value))

    def decode(self, data: bytes) -> SkillSpec:
        return self.from_payload(_decode(data))


class SkillMarkdownSpecCodec:
    """Decode standard SKILL.md documents without rewriting their text."""

    def model_content(self, content: str) -> str:
        """Return the model-visible Markdown with display metadata removed."""
        if not isinstance(content, str):
            raise TypeError("skill content must be a string")
        try:
            frontmatter = _parse_skill_markdown(content)
            lines = content.splitlines(keepends=True)
            closing = next(
                index
                for index, line in enumerate(lines[1:], 1)
                if line.rstrip("\r\n") == "---"
            )
            node = yaml.compose(
                "".join(lines[1:closing]), Loader=_StrictSafeLoader
            )
        except (AIError, yaml.YAMLError, TypeError, ValueError):
            return content
        if not isinstance(node, yaml.nodes.MappingNode):
            return content
        if not node.flow_style:
            if "metadata" not in frontmatter:
                return content
            for index, (key_node, _value_node) in enumerate(node.value):
                if (
                    isinstance(key_node, yaml.nodes.ScalarNode)
                    and key_node.tag == "tag:yaml.org,2002:str"
                    and key_node.value == "metadata"
                ):
                    start = key_node.start_mark.line
                    end = (
                        node.value[index + 1][0].start_mark.line
                        if index + 1 < len(node.value)
                        else closing - 1
                    )
                    return "".join(lines[: start + 1] + lines[end + 1 :])
            return content
        frontmatter.pop("metadata", None)
        encoded = yaml.safe_dump(
            frontmatter,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
        return f"---\n{encoded}---\n{''.join(lines[closing + 1:])}"

    def encode(self, value: SkillSpec) -> bytes:
        try:
            frontmatter = _parse_skill_markdown(value.content)
        except Exception as error:
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        metadata = dict(frontmatter.get("metadata", {}))
        revision = _skill_revision(metadata.pop("linktools-revision", 1))
        if (
            frontmatter["name"] != value.id
            or frontmatter["description"] != value.description
            or metadata != dict(value.metadata)
            or revision != value.revision
        ):
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        try:
            return value.content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error

    def decode(self, data: bytes) -> SkillSpec:
        try:
            content = data.decode("utf-8")
            frontmatter = _parse_skill_markdown(content)
            metadata = dict(frontmatter.get("metadata", {}))
            revision = _skill_revision(metadata.pop("linktools-revision", 1))
            return SkillSpec(
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


class SkillMarkdownSpecAdapter:
    """Translate standard local skill names to complete logical ids."""

    def to_logical(self, logical_id: str, value: SkillSpec) -> SkillSpec:
        local_name = logical_id.rsplit("/", 1)[-1]
        if value.id != local_name:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return SkillSpec(
            logical_id,
            value.content,
            value.description,
            value.metadata,
            revision=value.revision,
        )

    def to_storage(self, logical_id: str, value: SkillSpec) -> SkillSpec:
        if value.id != logical_id:
            raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
        return SkillSpec(
            logical_id.rsplit("/", 1)[-1],
            value.content,
            value.description,
            value.metadata,
            revision=value.revision,
        )


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
        """Return the complete declaration wire payload, including connection data."""
        if not isinstance(value, MCPServerSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server spec is invalid")
        payload: dict[str, JsonValue] = {
            "version": 1,
            "id": value.id,
            "revision": value.revision,
            "transport": value.transport,
        }
        if value.transport == "stdio":
            payload["command"] = value.command
            if value.args:
                payload["args"] = list(value.args)
            if value.env:
                payload["env"] = dict(value.env)
            if value.resource is not None:
                payload["resource"] = {"kind": value.resource.kind, "id": value.resource.id}
        else:
            payload["url"] = value.url
            if value.headers:
                payload["headers"] = dict(value.headers)
        return payload

    def to_contract_payload(self, value: MCPServerSpec) -> "dict[str, JsonValue]":
        """Return connection-free semantic MCP behavior for durable bindings."""
        if not isinstance(value, MCPServerSpec):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP server spec is invalid")
        payload: dict[str, JsonValue] = {
            "version": 1,
            "id": value.id,
            "revision": value.revision,
            "transport": value.transport,
        }
        if value.args:
            payload["args"] = list(value.args)
        if value.resource is not None:
            payload["resource"] = {"kind": value.resource.kind, "id": value.resource.id}
        return payload

    def to_binding_payload(
        self,
        value: MCPServerSpec,
        resource_versions: "Sequence[AssetVersionRef] | None",
        *,
        asset_source_id: "str | None" = None,
        execution_policy: "Mapping[str, JsonValue] | None" = None,
    ) -> "dict[str, JsonValue]":
        payload = self.to_contract_payload(value)
        if execution_policy is not None:
            payload["execution_policy"] = _execution_policy_payload(execution_policy)
        if value.resource is None:
            if resource_versions is not None or asset_source_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return payload
        if resource_versions is None or any(
            not isinstance(item, AssetVersionRef) for item in resource_versions
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        versions = tuple(sorted(resource_versions, key=lambda item: (item.key.kind, item.key.id)))
        if len({item.key for item in versions}) != len(versions):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(asset_source_id, str) or not asset_source_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload["resource_versions"] = [item.to_payload() for item in versions]
        payload["asset_source_id"] = asset_source_id
        return payload

    def from_payload(self, raw: Mapping[str, object]) -> MCPServerSpec:
        _require_v1(raw)
        if any(key in raw for key in ("resource_versions", "asset_source_id", "execution_policy")):
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                "MCP binding fields are Runtime-owned",
            )
        return _decode_mcp_wire_server(raw)

    def decode_author(
        self,
        data: bytes,
        *,
        format: Literal["json", "yaml"],
        package_id: "str | None" = None,
    ) -> MCPServerSpec:
        if format not in {"json", "yaml"}:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        raw = (
            decode_author_json_mapping(data)
            if format == "json"
            else decode_author_yaml_mapping(data)
        )
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
            if not isinstance(identity, str) or not identity.strip() or not isinstance(value, Mapping):
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

    def decode_binding_payload(
        self,
        raw: Mapping[str, object],
        *,
        declaration: MCPServerSpec,
    ) -> "tuple[AssetVersionRef, ...] | None":
        semantic, resource_versions = _decode_mcp_binding_contract(raw)
        if semantic != self.to_contract_payload(declaration):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return resource_versions

    def encode(self, value: MCPServerSpec) -> bytes:
        return _encode(self.to_payload(value))

    def decode(self, data: bytes) -> MCPServerSpec:
        return self.from_payload(_decode(data))


def _decode_mcp_wire_server(raw: Mapping[str, object]) -> MCPServerSpec:
    identity = raw.get("id")
    revision = _decode_revision(raw)
    transport = raw.get("transport")
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or transport not in {"stdio", "streamable-http", "sse"}
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)

    try:
        if transport == "stdio":
            if "url" in raw or "headers" in raw:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            command = raw.get("command")
            args = raw.get("args", [])
            env = raw.get("env", {})
            resource = _decode_asset_key(raw.get("resource"))
            if not isinstance(command, str) or not command.strip():
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            if not isinstance(args, list) or any(
                not isinstance(item, str) for item in args
            ):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            if not isinstance(env, Mapping) or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                for key, value in env.items()
            ):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            return MCPServerSpec(
                identity,
                command,
                tuple(args),
                resource,
                transport="stdio",
                env=dict(env),
                revision=revision,
            )

        if any(key in raw for key in ("command", "args", "env", "resource")):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        url = raw.get("url")
        headers = raw.get("headers", {})
        if not isinstance(url, str) or not url.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(headers, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in headers.items()
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return MCPServerSpec(
            identity,
            transport=transport,
            url=url,
            headers=dict(headers),
            revision=revision,
        )
    except AIError:
        raise
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


def _decode_mcp_author_server(
    raw: Mapping[str, object],
    *,
    identity: str,
    revision: int,
    package: bool,
) -> MCPServerSpec:
    transport_value = raw.get("type")
    has_command = "command" in raw
    has_url = "url" in raw
    if transport_value is None:
        if has_command == has_url:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        transport = "stdio" if has_command else "streamable-http"
    elif isinstance(transport_value, str):
        transport = "streamable-http" if transport_value in {"http", "streamable-http"} else transport_value
    else:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if transport not in {"stdio", "streamable-http", "sse"}:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if package and transport != "stdio":
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)

    if transport == "stdio":
        if "url" in raw or "headers" in raw:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        command = raw.get("command")
        args = raw.get("args", [])
        env = raw.get("env", {})
        if not isinstance(command, str) or not command.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(env, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in env.items()
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        try:
            return MCPServerSpec(
                identity,
                command,
                tuple(args),
                AssetKey("mcp", identity) if package else None,
                transport="stdio",
                env=dict(env),
                revision=revision,
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error

    if any(key in raw for key in ("command", "args", "env")):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    url = raw.get("url")
    headers = raw.get("headers", {})
    if not isinstance(url, str) or not url.strip():
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if not isinstance(headers, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in headers.items()
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        return MCPServerSpec(
            identity,
            transport=transport,
            url=url,
            headers=dict(headers),
            revision=revision,
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


def _decode_mcp_binding_contract(
    raw: Mapping[str, object],
) -> "tuple[dict[str, JsonValue], tuple[AssetVersionRef, ...] | None]":
    _require_v1(raw)
    if any(key in raw for key in ("command", "url", "env", "headers")):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    identity = raw.get("id")
    revision = _decode_revision(raw)
    transport = raw.get("transport")
    args = raw.get("args", [])
    resource = _decode_asset_key(raw.get("resource"))
    if (
        not isinstance(identity, str)
        or not identity.strip()
        or transport not in {"stdio", "streamable-http", "sse"}
        or not isinstance(args, list)
        or any(not isinstance(item, str) for item in args)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if transport != "stdio" and (args or resource is not None):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    semantic: dict[str, JsonValue] = {
        "version": 1,
        "id": identity,
        "revision": revision,
        "transport": transport,
    }
    if args:
        semantic["args"] = list(args)
    if resource is not None:
        semantic["resource"] = {"kind": resource.kind, "id": resource.id}

    resource_versions: tuple[AssetVersionRef, ...] | None = None
    raw_versions = raw.get("resource_versions")
    if raw_versions is not None:
        if not isinstance(raw_versions, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            parsed = tuple(AssetVersionRef.from_payload(item) for item in raw_versions)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        keys = tuple(item.key for item in parsed)
        if len(keys) != len(set(keys)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        resource_versions = tuple(sorted(parsed, key=lambda item: (item.key.kind, item.key.id)))
    asset_source_id = raw.get("asset_source_id")
    if resource is None:
        if resource_versions is not None or asset_source_id is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif resource_versions is None or not isinstance(asset_source_id, str) or not asset_source_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if "execution_policy" in raw:
        _execution_policy_payload(raw["execution_policy"])
    return semantic, resource_versions


def _decode_author_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return value


def _encode(value: "dict[str, object]") -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _execution_policy_payload(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    policy = dict(value)
    boundary = policy.get("boundary")
    if boundary in {"host-stdio", "host-network"}:
        expected = {"version": 1, "boundary": boundary}
        if policy != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return expected
    expected = {
        "version",
        "boundary",
        "workspace_access",
        "hidden_paths",
        "network",
    }
    workspace_access = policy.get("workspace_access")
    hidden_paths = policy.get("hidden_paths")
    if (
        set(policy) != expected
        or policy.get("version") != 1
        or isinstance(policy.get("version"), bool)
        or boundary != "workspace-stdio"
        or not isinstance(workspace_access, str)
        or workspace_access not in {"read", "read_write", "none"}
        or policy.get("network") != "isolated"
        or not isinstance(hidden_paths, list)
        or any(not isinstance(path, str) or not path for path in hidden_paths)
        or hidden_paths != sorted(set(hidden_paths))
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return {
        "version": 1,
        "boundary": "workspace-stdio",
        "workspace_access": workspace_access,
        "hidden_paths": list(hidden_paths),
        "network": "isolated",
    }


def _decode_asset_key(value: object) -> AssetKey | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP resource is invalid")
    kind = value.get("kind")
    identity = value.get("id")
    if not isinstance(kind, str) or not isinstance(identity, str):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP resource is invalid")
    try:
        return AssetKey(kind, identity)
    except ValueError as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "MCP resource is invalid") from error


def _decode(data: bytes) -> "dict[str, object]":
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def decode_author_json_mapping(data: bytes) -> dict[str, object]:
    """Decode an author JSON object while rejecting duplicate keys."""
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_json_mapping,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if not isinstance(value, dict):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return value


def decode_author_yaml_mapping(data: bytes) -> dict[str, object]:
    """Decode finite JSON-shaped author YAML while rejecting duplicate and merge keys."""
    try:
        text = data.decode("utf-8", errors="strict")
        value = yaml.load(text, Loader=_StrictSafeLoader)
        normalized = normalize_json_value(value)
    except AIError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return normalized


def _strict_json_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"unsupported JSON constant: {value}")


def _require_version(raw: Mapping[str, object], supported: set[int]) -> int:
    version = raw.get("version")
    if version is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version not in supported:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return version


def _require_v1(raw: Mapping[str, object]) -> None:
    _require_version(raw, {1})


def _decode_revision(raw: Mapping[str, object]) -> int:
    value = raw.get("revision", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AIError(
            ErrorCode.OUTPUT_CONTRACT_INVALID,
            "revision must be a positive integer",
        )
    return value


def _decode_usage_limits(value: object) -> "AgentUsageLimits | None":
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID, "usage_limits must be an object or null")
    if any(not isinstance(name, str) for name in value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "usage_limits field name is invalid")
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
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("YAML merge keys are not supported")
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError("duplicate YAML key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _skill_revision(value: object) -> int:
    if isinstance(value, bool):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if isinstance(value, int):
        revision = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None:
        revision = int(value)
    else:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if revision < 1:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return revision


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
    if "metadata" in frontmatter:
        metadata = frontmatter["metadata"]
        if not isinstance(metadata, Mapping):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        _skill_revision(metadata.get("linktools-revision", 1))
        try:
            normalized = normalize_json_value(dict(metadata))
            frontmatter["metadata"] = dict(
                ImmutableJsonMapping(
                    normalized,
                    allow_empty_keys=True,
                )
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
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
