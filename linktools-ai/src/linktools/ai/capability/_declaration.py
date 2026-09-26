#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-backed declaration loaders for CapabilityGroup."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ..core import (
    DEFAULT_DISCOVERY_POLICY,
    ImmutableJsonMapping,
    JsonValue,
    validate_logical_id,
)
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentMarkdownSpecCodec,
    AgentSpec,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    RepositoryInstructionDocument,
)
from ._contribution import CapabilityContribution
from ._resource_path import (
    mcp_resource_path,
    validate_resource_path,
    validate_resource_tree,
)
from ._skill import SkillDefinition
from ._skill_source import SkillResource, SkillSourceRef

if TYPE_CHECKING:
    from ._loading import CapabilityLoadContext, CapabilityLoadEntry

_DECLARATION_SUFFIXES = {
    "agent": ("/AGENT.md",),
    "skill": ("/SKILL.md",),
    "mcp": ("/mcp.json", "/mcp.yaml"),
    "rule": (".md",),
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
                self._defaults = ImmutableJsonMapping(defaults)
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
        if self.source_kind == "mcp":
            return await _load_mcp(context)
        return await _load_rules(context)


async def _load_agents(
    context: CapabilityLoadContext,
    *,
    source_kind: str,
    defaults: "Mapping[str, object] | None",
) -> "Sequence[AgentSpec]":
    declarations = _package_declarations(context.list(kind=source_kind), ("/AGENT.md",))
    values = await context.read_many(tuple(entry.key for entry in declarations))
    markdown = AgentMarkdownSpecCodec()
    result: list[AgentSpec] = []
    for entry, data in zip(declarations, values, strict=True):
        logical_id = entry.key.id[: -len("/AGENT.md")]
        try:
            validate_logical_id(logical_id)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        result.append(markdown.decode(data, logical_id=logical_id, defaults=defaults))
    return result


async def _load_skills(
    context: CapabilityLoadContext,
) -> "Sequence[SkillDefinition]":
    entries = context.list(kind="skill")
    declarations = _package_declarations(entries, ("/SKILL.md",))
    values = await context.read_many(tuple(entry.key for entry in declarations))
    result: list[SkillDefinition] = []
    markdown = SkillMarkdownSpecCodec()
    adapter = SkillMarkdownSpecAdapter()
    for entry, data in zip(declarations, values, strict=True):
        logical_id = entry.key.id[: -len("/SKILL.md")]
        try:
            validate_logical_id(logical_id)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        value = adapter.to_logical(logical_id, markdown.decode(data))
        prefix = f"{logical_id}/"
        resources: list[tuple[str, CapabilityLoadEntry]] = []
        for candidate in entries:
            if not candidate.key.id.startswith(prefix):
                continue
            relative = candidate.key.id[len(prefix) :]
            if relative == "SKILL.md":
                continue
            validate_resource_path(relative)
            resources.append((relative, candidate))
        resources.sort(key=lambda item: item[0])
        resource_keys = tuple(candidate.key for _relative, candidate in resources)
        refs = context.bind_versions(resource_keys)
        paths = await context.asset_reader.local_paths(resource_keys)
        versions: list[SkillResource] = []
        for (relative, _candidate), ref, path in zip(resources, refs, paths, strict=True):
            mode = 0
            if path is not None:
                try:
                    mode = (await asyncio.to_thread(path.stat)).st_mode & 0o111
                except OSError as error:
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            versions.append(SkillResource(relative, ref, mode))
        result.append(
            SkillDefinition(
                value,
                SkillSourceRef(context.group_id, logical_id, tuple(versions)),
            )
        )
    return result


async def _load_mcp(
    context: CapabilityLoadContext,
) -> "Sequence[MCPServerSpec]":
    entries = context.list(kind="mcp")
    declarations = _package_declarations(entries, ("/mcp.json", "/mcp.yaml"))
    values = await context.read_many(tuple(entry.key for entry in declarations))
    codec = MCPServerSpecCodec()
    result: list[MCPServerSpec] = []
    for entry, data in zip(declarations, values, strict=True):
        suffix = "/mcp.json" if entry.key.id.endswith("/mcp.json") else "/mcp.yaml"
        result.append(
            codec.decode_author(
                data,
                format="json" if suffix.endswith(".json") else "yaml",
                package_id=entry.key.id[: -len(suffix)],
            )
        )
    return result


def _bind_mcp_declaration(
    value: MCPServerSpec,
    context: CapabilityLoadContext,
) -> CapabilityContribution[object]:
    root = value.resource
    if root is None:
        return CapabilityContribution.from_declaration(value)

    resources = tuple(
        (relative, candidate.key)
        for candidate in context.list(kind=root.kind)
        if (relative := mcp_resource_path(candidate.key, root)) is not None
    )
    available = {relative for relative, _key in resources}
    validate_resource_tree(available)
    _validate_resource_arguments(value.args, available)
    refs = context.bind_versions(tuple(key for _relative, key in resources))
    return CapabilityContribution.from_mcp_contract(
        MCPServerSpecCodec().to_binding_payload(
            value,
            refs,
            asset_source_id=context.group_id,
        ),
        value,
    )


async def _load_rules(
    context: CapabilityLoadContext,
) -> "Sequence[RepositoryInstructionDocument]":
    candidates = tuple(
        entry
        for entry in context.list(kind="rule")
        if entry.key.id.endswith(".md")
    )
    for entry in candidates:
        try:
            validate_logical_id(entry.key.id[:-3])
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    entries = tuple(
        entry
        for entry in candidates
        if not DEFAULT_DISCOVERY_POLICY.ignores(entry.key.id)
    )
    values = await context.read_many(tuple(entry.key for entry in entries))
    result: list[RepositoryInstructionDocument] = []
    for entry, value in zip(entries, values, strict=True):
        source_id = entry.key.id[:-3]
        try:
            content = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AIError(
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                safe_details={
                    "asset_source_id": context.group_id,
                    "asset_id": entry.key.id,
                },
            ) from error
        scope = "."
        if content.startswith("---\n"):
            metadata, separator, content = content[4:].partition("\n---\n")
            if not separator:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            declared_scope: str | None = None
            for field in metadata.splitlines():
                if not field.startswith("scope:"):
                    continue
                if declared_scope is not None or not field.startswith("scope: "):
                    raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
                declared_scope = field[len("scope: ") :]
            if declared_scope is not None:
                scope = declared_scope
        result.append(
            RepositoryInstructionDocument(
                f"rule:{source_id}",
                scope,
                content,
            )
        )
    return result


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


def _package_declarations(
    entries: Sequence[CapabilityLoadEntry],
    suffixes: Sequence[str],
) -> "tuple[CapabilityLoadEntry, ...]":
    declarations = tuple(
        entry for entry in entries if any(entry.key.id.endswith(suffix) for suffix in suffixes)
    )
    root_ids: list[str] = []
    for entry in declarations:
        suffix = next(suffix for suffix in suffixes if entry.key.id.endswith(suffix))
        root = entry.key.id[: -len(suffix)]
        if not root or root in root_ids:
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        root_ids.append(root)
    _validate_package_roots(tuple(sorted(root_ids)))
    return tuple(sorted(declarations, key=lambda entry: entry.key.id))


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
