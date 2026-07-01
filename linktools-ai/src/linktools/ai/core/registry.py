#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Path-based registries for loading skill, subagent, MCP, and agent specs."""

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
    load_yaml_text as _load_yaml_text,
    load_markdown_file as _load_markdown_file,
    load_markdown_text as _load_markdown_text,
    as_str_dict as _as_str_dict,
)

if TYPE_CHECKING:
    from ..capabilities.store import CapabilityStore


logger = logging.getLogger("linktools.ai.core.registry")


def _get(payload: Mapping[str, object], key: str) -> object:
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
    base_dir: Path | None
    enabled: bool


@dataclass(frozen=True, slots=True)
class SpecSource:
    name: str
    path: Path
    base_dir: Path | None
    instructions: str = ""


@dataclass(slots=True)
class AgentSpec(BaseSpec):
    description: str = ""
    model: str = "standard"
    allowed_tools: list[str] | None = None
    allowed_skills: list[str] | None = None
    allowed_subagents: list[str] | None = None
    system_prompt: str = ""
    prompt_sections: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    instructions: str = ""

    @property
    def tools(self) -> list[str]:
        return list(self.allowed_tools or [])

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], source: SpecSource) -> Self:
        meta: dict[str, object] = {}
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


@dataclass(slots=True)
class SubagentSpec(AgentSpec):
    pass


@dataclass(slots=True)
class SkillSpec(BaseSpec):
    description: str = ""
    source: str = ""
    category: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    instructions: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], source: SpecSource) -> Self:
        return cls(
            name=source.name,
            path=source.path,
            base_dir=source.base_dir,
            enabled=bool(payload.get("enabled", True)),
            description=str(payload.get("description") or ""),
            source=str(payload.get("source") or ""),
            category=str(payload.get("category") or ""),
            metadata=dict(payload.get("metadata", {})),
            instructions=source.instructions,
        )


@dataclass(slots=True)
class MCPServerSpec(BaseSpec):
    description: str = ""
    server_name: str = ""
    kind: str = "read"
    provides: list[str] = field(default_factory=list)
    mcp_type: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    cache: dict[str, object] = field(default_factory=dict)
    circuit_breaker: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], source: SpecSource) -> Self:
        mcp = dict(payload.get("mcp", {}))
        return cls(
            name=source.name,
            path=source.path,
            base_dir=source.base_dir,
            enabled=bool(payload.get("enabled", mcp.get("enabled", True))),
            description=str(payload.get("description") or payload.get("display_name") or ""),
            server_name=str(mcp.get("server", source.name)),
            kind=str(payload.get("kind", "read")),
            provides=list(payload.get("provides", [])),
            mcp_type=str(mcp.get("type", "stdio")),
            command=str(mcp.get("command", "")),
            args=[str(a) for a in mcp.get("args", [])],
            env={str(k): str(v) for k, v in mcp.get("env", {}).items()},
            url=str(mcp.get("url", "")),
            headers={str(k): str(v) for k, v in mcp.get("headers", {}).items()},
            cache=dict(payload.get("cache", {})),
            circuit_breaker=dict(payload.get("circuit_breaker", {})),
        )


SpecType = TypeVar("SpecType", bound=BaseSpec)


# ===========================================================================
# Base Registry
# ===========================================================================

class BaseRegistry(Generic[SpecType], metaclass=abc.ABCMeta):

    def __init__(self, *paths: Path):
        self._paths: list[Path] = list(paths)
        self._specs: dict[str, SpecType] | None = None
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

    def _get_specs(self) -> dict[str, SpecType]:
        if self._specs is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been preloaded — call await registry.preload() first"
            )
        return self._specs

    @abc.abstractmethod
    async def _load(self) -> dict[str, SpecType]:
        pass

    def get(self, spec_id: str) -> SpecType:
        return self._get_specs()[spec_id]

    def all(self) -> list[SpecType]:
        return list(self._get_specs().values())

    def __contains__(self, spec_id: str) -> bool:
        return spec_id in self._get_specs()

    @staticmethod
    def _group_defaults(agent_dir: Path) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        for path in [agent_dir, agent_dir.parent]:
            group_path = path / "agent-group.yaml"
            if group_path.is_file():
                groups.append(_load_yaml_file(group_path))
        return groups

    @classmethod
    def _resolve_groups(cls, payload: dict[str, object], base_dir: Path) -> list[dict[str, object]]:
        """Directory `agent-group.yaml` defaults for this agent, unless its frontmatter
        opts out with `inherit-group: false`. The opt-out lets a conversational agent
        (e.g. analyst/reporter) live in the same capability dir as pipeline stages without
        absorbing the audit-orchestration group's system prompt / sections."""
        if _get(payload, "inherit_group") is False:
            return []
        return cls._group_defaults(base_dir)

    @staticmethod
    def _merge_sections(payload: dict[str, object], groups: list[dict[str, object]]) -> dict[str, str]:
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


class _MarkdownAgentRegistry(BaseRegistry[_AgentSpecT]):
    """Scan a directory of `<name>/agent.md` files and build agent specs."""

    _kind: str = "agent"  # for log messages
    _spec_cls: type[_AgentSpecT]

    def __init__(
        self,
        *paths: Path,
        cap_store: "CapabilityStore | None" = None,
        capabilities_root: Path | None = None,
        cap_kind: str | None = None,
    ) -> None:
        super().__init__(*paths)
        self._cap_store = cap_store
        self._capabilities_root = capabilities_root  # source caps root for agent-group.yaml lookup
        self._cap_kind = cap_kind  # DB kind string e.g. "worker", "stage", "subagent"

    async def _load(self) -> dict[str, _AgentSpecT]:
        result: dict[str, _AgentSpecT] = {}
        for path in self._paths:
            if not path.is_dir():
                continue
            for child in sorted(path.iterdir()):
                if not child.is_dir():
                    continue
                markdown = _find_file(child, "agent.md")
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
        override: dict[str, object] = {"sections": self._merge_sections(payload, groups)}
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
        override: dict[str, object] = {"sections": self._merge_sections(payload, groups)}
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


class AgentRegistry(_MarkdownAgentRegistry[AgentSpec]):
    _kind: str = "agent"
    _spec_cls: type[AgentSpec] = AgentSpec


class SubagentRegistry(_MarkdownAgentRegistry[SubagentSpec]):
    _kind: str = "subagent"
    _spec_cls: type[SubagentSpec] = SubagentSpec


# ===========================================================================
# Skill Registry
# ===========================================================================

class SkillRegistry(BaseRegistry[SkillSpec]):
    """Scan skill directories and load SkillSpec objects keyed by skill_id."""

    def __init__(
        self,
        *paths: Path,
        cap_store: "CapabilityStore | None" = None,
        capabilities_root: Path | None = None,
    ) -> None:
        super().__init__(*paths)
        self._cap_store = cap_store
        self._capabilities_root = capabilities_root

    async def _load(self) -> dict[str, SkillSpec]:
        result: dict[str, SkillSpec] = {}
        for path in self._paths:
            if not path.is_dir():
                continue
            for child in sorted(path.iterdir()):
                spec = None
                if child.is_file() and child.name.lower() == "skill.md":
                    spec = self._load_spec(child.stem, child, None)
                elif child.is_dir():
                    markdown = _find_file(child, "skill.md")
                    if markdown is not None:
                        spec = self._load_spec(child.name, markdown, child)
                if spec and spec.enabled:
                    if spec.name in result:
                        logger.warning("Skill '%s' from %s overrides existing registration", spec.name, child)
                    result[spec.name] = spec
                    logger.debug("skill loaded: name=%s category=%s path=%s", spec.name, spec.category, child)
        if self._cap_store is not None:
            for capability_id, _, content in await self._cap_store.iter_primaries("skill"):
                base_dir = self._capabilities_root / "skill" / capability_id if self._capabilities_root else None
                spec = self._load_spec_from_content(
                    capability_id,
                    content,
                    base_dir,
                )
                if spec and spec.enabled:
                    if spec.name in result:
                        logger.warning("Skill '%s' from DB overrides source registration", spec.name)
                    result[spec.name] = spec
                    logger.debug("skill loaded from DB: name=%s", spec.name)
        logger.debug("SkillRegistry: loaded %d skills [%s]", len(result), ", ".join(result))
        return result

    def _load_spec(self, name: str, path: Path, base_dir: Path | None) -> SkillSpec:
        payload, instructions = _load_markdown_file(path)
        return SkillSpec.from_dict(
            payload,
            SpecSource(
                name=str(payload.get("name") or name),
                path=path,
                base_dir=base_dir,
                instructions=instructions,
            )
        )

    def _load_spec_from_content(self, capability_id: str, content: str, base_dir: Path | None = None) -> SkillSpec:
        payload, instructions = _load_markdown_text(content, source=f"<db:{capability_id}>")
        if base_dir is not None and not base_dir.is_dir():
            base_dir = None
        runtime_name = self._runtime_name_from_capability(capability_id, payload.get("name"), label="Skill")
        return SkillSpec.from_dict(
            payload,
            SpecSource(
                name=runtime_name,
                path=(base_dir / "SKILL.md") if base_dir else Path(f"skill/{capability_id}/SKILL.md"),
                base_dir=base_dir,
                instructions=instructions,
            ),
        )


# ===========================================================================
# MCP Registry
# ===========================================================================

class MCPRegistry(BaseRegistry[MCPServerSpec]):
    """Scan MCP directories and load MCPServerSpec objects keyed by server_id."""

    def __init__(self, *paths: Path, cap_store: "CapabilityStore | None" = None) -> None:
        super().__init__(*paths)
        self._cap_store = cap_store

    async def _load(self) -> dict[str, MCPServerSpec]:
        result: dict[str, MCPServerSpec] = {}
        for path in self._paths:
            if not path.is_dir():
                continue
            for child in sorted(path.iterdir()):
                spec = None
                if child.is_file() and child.name.lower() == "mcp.yaml":
                    spec = self._load_spec(child.stem, child, None)
                elif child.is_dir():
                    yaml_file = _find_file(child, "mcp.yaml")
                    if yaml_file is not None:
                        spec = self._load_spec(child.name, yaml_file, child)
                if spec and spec.enabled:
                    if spec.name in result:
                        logger.warning("MCP server '%s' from %s overrides existing registration", spec.name, child)
                    result[spec.name] = spec
                    logger.debug("mcp loaded: name=%s path=%s", spec.name, child)
        if self._cap_store is not None:
            for capability_id, _, content in await self._cap_store.iter_primaries("mcp"):
                payload = _load_yaml_text(content, source=f"<db:{capability_id}>")
                base_dir = next(
                    (p / capability_id for p in self._paths if (p / capability_id).is_dir()), None
                )
                runtime_name = self._runtime_name_from_capability(capability_id, payload.get("name"), label="MCP server")
                spec = MCPServerSpec.from_dict(
                    payload,
                    SpecSource(
                        name=runtime_name,
                        path=(base_dir / "mcp.yaml") if base_dir else Path(f"adapter/{capability_id}/mcp.yaml"),
                        base_dir=base_dir,
                    ),
                )
                if spec.enabled:
                    if spec.name in result:
                        logger.warning("MCP server '%s' from DB overrides source registration", spec.name)
                    result[spec.name] = spec
                    logger.debug("mcp loaded from DB: name=%s", spec.name)
        logger.debug("MCPRegistry: loaded %d servers [%s]", len(result), ", ".join(result))
        return result

    def _load_spec(self, name: str, path: Path, base_dir: Path | None) -> MCPServerSpec:
        payload = _load_yaml_file(path)
        return MCPServerSpec.from_dict(
            payload,
            SpecSource(
                name=str(payload.get("name") or name),
                path=path,
                base_dir=base_dir,
            )
        )

    def resolve_by_capability(self, capability: str) -> MCPServerSpec | None:
        """Find the first registered server whose `provides` list contains `capability`."""
        for spec in self:
            if capability in spec.provides:
                return spec
        return None


def _find_file(directory: Path, name: str) -> Path | None:
    if not directory.is_dir():
        return None
    target = name.lower()
    for child in directory.iterdir():
        if child.is_file() and child.name.lower() == target:
            return child
    return None
