#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-backed declaration loaders for CapabilityGroup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from ..core import ImmutableJsonMapping, JsonValue, validate_logical_id
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentMarkdownSpecCodec,
    AgentSpec,
    AgentSpecCodec,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpecCodec,
)
from ._resource_path import (
    mcp_resource_path,
    validate_resource_path,
    validate_resource_tree,
)
from ._skill import SkillDefinition
from ._skill_source import SkillSourceRef

if TYPE_CHECKING:
    from ._group import CapabilityLoadContext, CapabilityLoadEntry

_DECLARATION_SUFFIXES = {
    "agent": ("/AGENT.md",),
    "skill": ("/SKILL.md",),
    "mcp": ("/mcp.json", "/mcp.yaml"),
}


class AgentDeclarationLoader:
    """Load Agent declarations from one custom source kind."""

    def __init__(
        self,
        source_kind: str,
        defaults: "Mapping[str, object] | None" = None,
    ) -> None:
        if not isinstance(source_kind, str) or not source_kind.strip():
            raise ValueError("source_kind must be a non-empty string")
        self.source_kind = source_kind
        if defaults is None:
            self._defaults: Mapping[str, JsonValue] | None = None
        else:
            if not isinstance(defaults, Mapping):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            try:
                self._defaults = ImmutableJsonMapping(
                    cast(Mapping[str, JsonValue], defaults)
                )
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
            AgentMarkdownSpecCodec().from_payload(
                {"system_prompt": ""},
                logical_id="defaults-validation",
                defaults=self._defaults,
            )

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[AgentSpec]":
        return await _load_agents(
            context,
            source_kind=self.source_kind,
            defaults=self._defaults,
        )


class BuiltinDeclarationLoader:
    """Load the standard Agent, Skill, or MCP declaration layout."""

    def __init__(self, kind: str) -> None:
        if kind not in _DECLARATION_SUFFIXES:
            raise ValueError("unsupported declaration kind")
        self.source_kind = kind

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[AgentSpec | SkillDefinition | MCPServerSpec]":
        if self.source_kind == "agent":
            return await _load_agents(
                context,
                source_kind="agent",
                defaults=None,
            )
        if self.source_kind == "skill":
            return await _load_skills(context)
        return await _load_mcp(context)


async def _load_agents(
    context: CapabilityLoadContext,
    *,
    source_kind: str,
    defaults: "Mapping[str, object] | None",
) -> "Sequence[AgentSpec]":
    entries = context.list(kind=source_kind)
    roots, declarations = _package_declarations(entries, ("/AGENT.md",))
    values = await context.read_many(tuple(entry.key for entry in declarations))
    by_key = dict(zip((entry.key for entry in declarations), values, strict=True))
    markdown = AgentMarkdownSpecCodec()
    result: list[AgentSpec] = []
    for entry in declarations:
        data = by_key[entry.key]
        if entry.key.id.endswith("/AGENT.md"):
            logical_id = entry.key.id[: -len("/AGENT.md")]
            try:
                validate_logical_id(logical_id)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
            value = markdown.decode(
                data,
                logical_id=logical_id,
                defaults=defaults,
            )
        else:
            raw = _decode_agent_flat(data)
            version = raw.get("version")
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version != 1
                or raw.get("id") != entry.key.id
            ):
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
            payload = {key: item for key, item in raw.items() if key != "version"}
            payload.setdefault("system_prompt", "")
            value = markdown.from_payload(
                payload,
                logical_id=entry.key.id,
                defaults=defaults,
            )
        result.append(value)
    return result


async def _load_skills(
    context: CapabilityLoadContext,
) -> "Sequence[SkillDefinition]":
    entries = context.list(kind="skill")
    roots, declarations = _package_declarations(entries, ("/SKILL.md",))
    values = await context.read_many(tuple(entry.key for entry in declarations))
    by_key = dict(zip((entry.key for entry in declarations), values, strict=True))
    result: list[SkillDefinition] = []
    markdown = SkillMarkdownSpecCodec()
    adapter = SkillMarkdownSpecAdapter()
    codec = SkillSpecCodec()
    for entry in declarations:
        key = entry.key
        if key.id.endswith("/SKILL.md"):
            logical_id = key.id[: -len("/SKILL.md")]
            try:
                validate_logical_id(logical_id)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
            value = adapter.to_logical(logical_id, markdown.decode(by_key[key]))
            definition = SkillDefinition(
                value,
                SkillSourceRef(context.group_id, logical_id),
            )
        elif _inside_package(key.id, roots):
            continue
        else:
            value = codec.decode_author(by_key[key])
            if value.id != key.id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
            definition = SkillDefinition(value)
        result.append(definition)
    return result


async def _load_mcp(
    context: CapabilityLoadContext,
) -> "Sequence[MCPServerSpec]":
    entries = context.list(kind="mcp")
    roots, declarations = _package_declarations(
        entries,
        ("/mcp.json", "/mcp.yaml"),
    )
    values = await context.read_many(tuple(entry.key for entry in declarations))
    by_key = dict(zip((entry.key for entry in declarations), values, strict=True))
    codec = MCPServerSpecCodec()
    result: list[MCPServerSpec] = []
    package_main_keys = {entry.key for entry in roots}
    for entry in declarations:
        key = entry.key
        if key in package_main_keys:
            suffix = "/mcp.json" if key.id.endswith("/mcp.json") else "/mcp.yaml"
            package_id = key.id[: -len(suffix)]
            format = "json" if suffix.endswith(".json") else "yaml"
            value = codec.decode_author(
                by_key[key],
                format=format,
                package_id=package_id,
            )
            _validate_package_resource_args(
                context,
                package_id,
                value,
            )
        else:
            value = codec.decode_author(by_key[key], format="json")
            if value.id != key.id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
            _validate_resource_args(context, value)
        result.append(value)
    return result


def _validate_package_resource_args(
    context: CapabilityLoadContext,
    package_id: str,
    server: MCPServerSpec,
) -> None:
    root = server.resource_root
    if root is None or root.kind != "mcp" or root.id != package_id:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    available = {
        relative
        for entry in context.list(kind=root.kind)
        if (relative := mcp_resource_path(entry.key, root)) is not None
    }
    validate_resource_tree(available)
    _validate_resource_arguments(server.args, available)


def _validate_resource_args(
    context: CapabilityLoadContext,
    server: MCPServerSpec,
) -> None:
    root = server.resource_root
    if root is None:
        if any(argument.startswith("resource:") for argument in server.args):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return
    available = {
        relative
        for entry in context.list(kind=root.kind)
        if (relative := mcp_resource_path(entry.key, root)) is not None
    }
    validate_resource_tree(available)
    _validate_resource_arguments(server.args, available)


def _validate_resource_arguments(
    args: Sequence[str],
    available: set[str],
) -> None:
    for argument in args:
        if not argument.startswith("resource:"):
            continue
        relative = argument[len("resource:") :]
        validate_resource_path(relative)
        if relative not in available:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _decode_agent_flat(data: bytes) -> dict[str, object]:
    return AgentSpecCodec().decode_author_mapping(data)


def _package_declarations(
    entries: Sequence[CapabilityLoadEntry],
    suffixes: Sequence[str],
) -> "tuple[tuple[CapabilityLoadEntry, ...], tuple[CapabilityLoadEntry, ...]]":
    roots = tuple(
        entry
        for entry in entries
        if any(entry.key.id.endswith(suffix) for suffix in suffixes)
    )
    root_ids: list[str] = []
    for entry in roots:
        suffix = next(
            suffix for suffix in suffixes if entry.key.id.endswith(suffix)
        )
        root = entry.key.id[: -len(suffix)]
        if not root:
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        if root in root_ids:
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        root_ids.append(root)
    ordered_roots = tuple(sorted(root_ids))
    _validate_package_roots(ordered_roots)
    flat = tuple(
        entry
        for entry in entries
        if entry not in roots and not _inside_package(entry.key.id, ordered_roots)
    )
    if set(ordered_roots).intersection(entry.key.id for entry in flat):
        raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
    return roots, tuple(sorted((*flat, *roots), key=lambda entry: entry.key.id))


def _validate_package_roots(roots: Sequence[str]) -> None:
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if other.startswith(f"{root}/") or root.startswith(f"{other}/"):
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)


def _inside_package(identifier: str, roots: Sequence[str]) -> bool:
    return any(identifier.startswith(f"{root}/") for root in roots)


__all__ = [
    "AgentDeclarationLoader",
    "BuiltinDeclarationLoader",
]
