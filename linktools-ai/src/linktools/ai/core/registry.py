#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generic path-based registry machinery for loading agent-like specs.

Concrete spec/registry pairs for skills, subagents, and MCP servers live in
their own feature packages (`..skill.registry`, `..subagent.registry`,
`..mcp.registry`) and build on the base classes defined here.
"""

import abc
import asyncio
from collections import ChainMap
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Self, TypeVar, cast

import logging
from typing import TYPE_CHECKING

from ..support.config import (
    load_yaml_file as _load_yaml_file,
    load_markdown_file as _load_markdown_file,
    load_markdown_text as _load_markdown_text,
    as_str_dict as _as_str_dict,
)

if TYPE_CHECKING:
    from ..registry_store.store import CapabilityStore


logger = logging.getLogger("linktools.ai.core.registry")


def _get(payload: "Mapping[str, object]", key: str) -> object:
    """Look up a field by kebab-case key, falling back to snake_case."""
    kebab = key.replace("_", "-")
    return payload[kebab] if kebab in payload else payload.get(key)


# ===========================================================================
# Spec dataclasses
# ===========================================================================

@dataclass(slots=True)
class BaseSpec:
    name: str
    path: Path
    base_dir: "Path | None"
    enabled: bool


@dataclass(frozen=True, slots=True)
class SpecSource:
    name: str
    path: Path
    base_dir: "Path | None"
    instructions: str = ""


@dataclass(slots=True)
class AgentSpec(BaseSpec):
    description: str = ""
    model: str = "standard"
    allowed_tools: "list[str] | None" = None
    allowed_skills: "list[str] | None" = None
    allowed_subagents: "list[str] | None" = None
    system_prompt: str = ""
    prompt_sections: "dict[str, str]" = field(default_factory=dict)
    metadata: "dict[str, object]" = field(default_factory=dict)
    instructions: str = ""

    @property
    def tools(self) -> "list[str]":
        return list(self.allowed_tools or [])

    @classmethod
    def from_dict(cls, payload: "Mapping[str, object]", source: SpecSource) -> Self:
        meta: "dict[str, object]" = {}
        sources = payload.maps if isinstance(payload, ChainMap) else [payload]
        for source_payload in reversed(sources):
            raw = source_payload.get("metadata")
            if isinstance(raw, Mapping):
                meta.update(raw)
        raw_tools = _get(payload, "allowed_tools")
        raw_skills = _get(payload, "allowed_skills")
        raw_subagents = _get(payload, "allowed_subagents")
        if raw_subagents is None:
            # Back-compat: derive from `metadata.subagents` list of dicts with `id`.
            raw_meta_subagents = meta.get("subagents")
            if isinstance(raw_meta_subagents, list):
                ids = [
                    str(item["id"])
                    for item in cast(list[object], raw_meta_subagents)
                    if isinstance(item, Mapping) and item.get("id")
                ]
                if ids:
                    raw_subagents = ids
        return cls(
            name=source.name,
            path=source.path,
            base_dir=source.base_dir,
            enabled=bool(payload.get("enabled", True)),
            description=str(payload.get("description", "")),
            allowed_tools=[str(t) for t in cast(list[object], raw_tools)] if raw_tools is not None else None,
            allowed_skills=[str(s) for s in cast(list[object], raw_skills)] if raw_skills is not None else None,
            allowed_subagents=[str(s) for s in cast(list[object], raw_subagents)] if raw_subagents is not None else None,
            model=str(payload.get("model") or "standard"),
            system_prompt=str(payload.get("system") or ""),
            prompt_sections=_as_str_dict(payload.get("sections")),
            metadata=meta,
            instructions=source.instructions,
        )


SpecType = TypeVar("SpecType", bound=BaseSpec)


# ===========================================================================
# Base Registry
# ===========================================================================

class BaseRegistry(Generic[SpecType], metaclass=abc.ABCMeta):

    def __init__(self, *paths: Path):
        self._paths: "list[Path]" = list(paths)
        self._specs: "dict[str, SpecType] | None" = None
        self._alock = asyncio.Lock()

    async def preload(self) -> None:
        """Load specs from filesystem and DB. Must be awaited before first use."""
        if self._specs is not None:
            return
        async with self._alock:
            if self._specs is None:
                self._specs = await self._load()

    def invalidate(self) -> None:
        self._specs = None

    def _get_specs(self) -> "dict[str, SpecType]":
        if self._specs is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been preloaded — call await registry.preload() first"
            )
        return self._specs

    @abc.abstractmethod
    async def _load(self) -> "dict[str, SpecType]":
        pass

    def get(self, spec_id: str) -> SpecType:
        return self._get_specs()[spec_id]

    def all(self) -> "list[SpecType]":
        return list(self._get_specs().values())

    def __contains__(self, spec_id: str) -> bool:
        return spec_id in self._get_specs()

    @staticmethod
    def _group_defaults(agent_dir: Path) -> "list[dict[str, object]]":
        groups: "list[dict[str, object]]" = []
        for path in [agent_dir, agent_dir.parent]:
            group_path = path / "agent-group.yaml"
            if group_path.is_file():
                groups.append(_load_yaml_file(group_path))
        return groups

    @classmethod
    def _resolve_groups(cls, payload: "dict[str, object]", base_dir: Path) -> "list[dict[str, object]]":
        """Directory `agent-group.yaml` defaults for this agent, unless its frontmatter
        opts out with `inherit-group: false`. The opt-out lets a conversational agent
        (e.g. analyst/reporter) live in the same capability dir as pipeline stages without
        absorbing the audit-orchestration group's system prompt / sections."""
        if _get(payload, "inherit_group") is False:
            return []
        return cls._group_defaults(base_dir)

    @staticmethod
    def _merge_sections(payload: "dict[str, object]", groups: "list[dict[str, object]]") -> "dict[str, str]":
        return dict(ChainMap(
            _as_str_dict(payload.get("sections")),
            *(_as_str_dict(g.get("sections")) for g in groups),
        ))

    def __len__(self) -> int:
        return len(self._get_specs())

    def __iter__(self):
        return iter(self._get_specs().values())

    @staticmethod
    def _runtime_name_from_capability(
        capability_id: str,
        content_name: object,
        *,
        label: str,
    ) -> str:
        resolved = str(content_name or capability_id)
        if resolved != capability_id:
            logger.warning(
                "%s '%s' loaded from DB with stale frontmatter name '%s'; "
                "using capability_id as the runtime name",
                label, capability_id, resolved,
            )
        return capability_id


# ===========================================================================
# Agent / Subagent Registry
# ===========================================================================

_AgentSpecT = TypeVar("_AgentSpecT", bound=AgentSpec)


class MarkdownAgentRegistry(BaseRegistry[_AgentSpecT]):
    """Scan a directory of `<name>/agent.md` files and build agent specs."""

    _kind: str = "agent"  # for log messages
    _spec_cls: "type[_AgentSpecT]"

    def __init__(
        self,
        *paths: Path,
        cap_store: "CapabilityStore | None" = None,
        capabilities_root: "Path | None" = None,
        cap_kind: "str | None" = None,
    ) -> None:
        super().__init__(*paths)
        self._cap_store = cap_store
        self._capabilities_root = capabilities_root  # source caps root for agent-group.yaml lookup
        self._cap_kind = cap_kind  # DB kind string e.g. "worker", "stage", "subagent"

    async def _load(self) -> "dict[str, _AgentSpecT]":
        result: "dict[str, _AgentSpecT]" = {}
        for path in self._paths:
            if not path.is_dir():
                continue
            for child in sorted(path.iterdir()):
                if not child.is_dir():
                    continue
                markdown = find_file(child, "agent.md")
                if markdown is None:
                    continue
                spec = self._load_spec(child.name, markdown, child)
                if not spec.enabled:
                    continue
                if spec.name in result:
                    logger.warning("%s '%s' from %s overrides existing registration", self._kind.title(), spec.name, child)
                result[spec.name] = spec
                logger.debug("%s loaded: name=%s path=%s", self._kind, spec.name, markdown)
        # Load DB-managed capabilities from memory store (overrides source)
        if self._cap_store is not None and self._capabilities_root is not None and self._cap_kind is not None:
            for capability_id, _, content in await self._cap_store.iter_primaries(self._cap_kind):
                base_dir = self._capabilities_root / self._cap_kind / capability_id
                spec = self._load_spec_from_content(capability_id, content, base_dir)
                if not spec.enabled:
                    continue
                if spec.name in result:
                    logger.warning("%s '%s' from DB overrides source registration", self._kind.title(), spec.name)
                result[spec.name] = spec
                logger.debug("%s loaded from DB: name=%s", self._kind, spec.name)
        logger.debug("%s: loaded %d entries [%s]", type(self).__name__, len(result), ", ".join(result))
        return result

    def _load_spec(self, name: str, path: Path, base_dir: Path) -> _AgentSpecT:
        payload, instructions = _load_markdown_file(path)
        groups = self._resolve_groups(payload, base_dir)
        override: "dict[str, object]" = {"sections": self._merge_sections(payload, groups)}
        merged = ChainMap(override, payload, *groups)
        return self._spec_cls.from_dict(
            merged,
            SpecSource(
                name=str(merged.get("name") or name),
                path=path,
                base_dir=base_dir,
                instructions=instructions,
            ),
        )

    def _load_spec_from_content(self, capability_id: str, content: str, base_dir: Path) -> _AgentSpecT:
        payload, instructions = _load_markdown_text(content, source=f"<db:{capability_id}>")
        spec_base_dir = base_dir if base_dir.is_dir() else None
        groups = self._resolve_groups(payload, base_dir) if spec_base_dir else []
        override: "dict[str, object]" = {"sections": self._merge_sections(payload, groups)}
        merged = ChainMap(override, payload, *groups)
        runtime_name = self._runtime_name_from_capability(
            capability_id,
            merged.get("name"),
            label=self._kind.title(),
        )
        path = (spec_base_dir / "agent.md") if spec_base_dir else Path(f"{self._cap_kind or self._kind}/{capability_id}/agent.md")
        return self._spec_cls.from_dict(
            merged,
            SpecSource(
                name=runtime_name,
                path=path,
                base_dir=spec_base_dir,
                instructions=instructions,
            ),
        )


class AgentRegistry(MarkdownAgentRegistry[AgentSpec]):
    _kind: str = "agent"
    _spec_cls: "type[AgentSpec]" = AgentSpec


def find_file(directory: Path, name: str) -> "Path | None":
    if not directory.is_dir():
        return None
    target = name.lower()
    for child in directory.iterdir():
        if child.is_file() and child.name.lower() == target:
            return child
    return None
