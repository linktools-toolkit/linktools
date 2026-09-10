#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability groups freeze runtime candidate definitions before execution."""

import functools
import hashlib
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar, cast

from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..asset import AssetKey, AssetStore
from ..core import ImmutableJsonMapping, JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpecCodec,
    ThinkingValue,
)
from ..task import TaskNodeHandler
from ._context import AgentContext
from ._names import SKILL_TOOL_NAMES, SUBAGENT_TOOL_NAMES
from ._skill import SkillDefinition
from ._skill_source import AssetSkillResourceSource, SkillResourceSource, SkillSourceRef

AppT = TypeVar("AppT")
PLAN_SAFE_METADATA_KEY = "linktools.ai.plan_safe"


ContributionKind = Literal["tool", "agent", "skill", "mcp", "capability", "task"]
ContributionSemanticValue: TypeAlias = (
    Tool
    | AgentSpec
    | SkillDefinition
    | MCPServerSpec
    | AbstractCapability
    | TaskNodeHandler[object]
)
_RESERVED_TOOL_NAMES = frozenset(
    {
        *SKILL_TOOL_NAMES,
        *SUBAGENT_TOOL_NAMES,
        "write_plan",
        "delete_memory",
        "read_memory",
        "search_memory",
        "write_memory",
    }
)
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_TASK_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RESERVED_TASK_TYPE_PREFIX = "linktools.ai."


@dataclass(frozen=True, slots=True)
class CapabilityContribution(Generic[AppT]):
    kind: ContributionKind
    id: str
    fingerprint: str
    value: "Tool[AgentContext[AppT]] | AgentSpec | SkillDefinition | MCPServerSpec | AbstractCapability[AgentContext[AppT]] | TaskNodeHandler[AppT]"

    def __post_init__(self) -> None:
        if self.kind not in {"tool", "agent", "skill", "mcp", "capability", "task"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if not isinstance(self.id, str) or not self.id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        _validate_fingerprint(self.fingerprint)
        if self.kind == "tool" and not isinstance(self.value, Tool):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "agent" and not isinstance(self.value, AgentSpec):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "skill" and not isinstance(self.value, SkillDefinition):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "mcp" and not isinstance(self.value, MCPServerSpec):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "capability" and not isinstance(self.value, AbstractCapability):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "task" and not isinstance(self.value, TaskNodeHandler):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "tool" and cast(Tool, self.value).name != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "agent" and cast(AgentSpec, self.value).id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "skill" and cast(SkillDefinition, self.value).id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "mcp" and cast(MCPServerSpec, self.value).id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "capability":
            capability = cast(AbstractCapability, self.value)
            capability_id = capability.id
            if not isinstance(capability.defer_loading, bool):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability_id is not None and capability_id != self.id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability_id is None and capability.defer_loading:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            _validate_external_capability_id(self.id)
        if self.kind == "task":
            handler = cast("TaskNodeHandler[object]", self.value)
            task_type, task_version = _task_identity(handler)
            if self.id != f"{task_type}@{task_version}":
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.fingerprint != capability_fingerprint(
            self.kind,
            self.id,
            self.semantic_contract,
        ):
            raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)

    @classmethod
    def from_opaque(
        cls,
        kind: Literal["tool", "capability"],
        identity: str,
        value: "Tool[AgentContext[AppT]] | AbstractCapability[AgentContext[AppT]]",
        *,
        revision: int = 1,
        semantic_id: "str | None" = None,
        semantic_config: "Mapping[str, JsonValue] | None" = None,
    ) -> "CapabilityContribution[AppT]":
        """Create an opaque Python Tool or Capability from its public semantic inputs."""
        if kind not in {"tool", "capability"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        _validate_revision(revision)
        contract = contribution_semantic_contract(
            kind,
            identity,
            value,
            semantic_revision=revision,
            semantic_id=semantic_id,
            semantic_config=semantic_config,
        )
        return _SemanticContribution(
            kind,
            identity,
            capability_fingerprint(kind, identity, contract),
            value,
            contract,
        )

    @property
    def semantic_contract(self) -> "dict[str, JsonValue]":
        return contribution_semantic_contract(
            self.kind,
            self.id,
            self.value,
        )


@dataclass(frozen=True, slots=True)
class _SemanticContribution(CapabilityContribution[AppT]):
    _contract: Mapping[str, JsonValue] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            contract = ImmutableJsonMapping(self._contract)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        object.__setattr__(self, "_contract", contract)
        CapabilityContribution.__post_init__(self)

    @property
    def semantic_contract(self) -> "dict[str, JsonValue]":
        return dict(self._contract)


@dataclass(frozen=True, slots=True)
class CapabilityLoadEntry:
    """Declaration-relevant metadata captured at the start of a group freeze."""

    key: AssetKey
    etag: str
    size: int
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.etag) != 64 or any(
            character not in "0123456789abcdef" for character in self.etag
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.size < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            metadata = ImmutableJsonMapping(self.metadata)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "metadata", metadata)


class CapabilityLoadContext:
    def __init__(
        self,
        group_id: str,
        store: AssetStore,
        entries: Sequence[CapabilityLoadEntry],
    ) -> None:
        self._group_id = group_id
        self._store = store
        self._entries = tuple(entries)
        self._by_key = {entry.key: entry for entry in self._entries}
        self._cache: dict[AssetKey, bytes] = {}
        if len(self._by_key) != len(self._entries):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def group_id(self) -> str:
        return self._group_id

    def list(
        self,
        *,
        kind: "str | None" = None,
        prefix: "str | None" = None,
    ) -> "tuple[CapabilityLoadEntry, ...]":
        """List captured declaration metadata without opening the backing store."""
        return tuple(
            entry
            for entry in self._entries
            if (kind is None or entry.key.kind == kind)
            and (prefix is None or entry.key.id.startswith(prefix))
        )

    async def read(self, key: AssetKey) -> bytes:
        entry = self._by_key.get(key)
        if entry is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        current = await self._store.stat(key)
        if (
            current is None
            or current.etag != entry.etag
            or current.size != entry.size
            or current.metadata != entry.metadata
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        value = await self._store.get(key)
        if value is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        data = bytes(value)
        if hashlib.sha256(data).hexdigest() != entry.etag:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._cache[key] = data
        return data


class CapabilityLoader(Protocol[AppT]):
    @property
    def id(self) -> str: ...

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[AppT]]": ...


class CapabilityGroup(Generic[AppT]):
    """Register and freeze one named set of runtime candidate definitions."""

    def __init__(self, group_id: str) -> None:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("capability group id must be a non-empty string")
        self._id = group_id
        self._store: AssetStore | None = None
        self._skill_source: SkillResourceSource | None = None
        self._loaders: list[CapabilityLoader[AppT]] = []
        self._contributions: list[CapabilityContribution[AppT]] = []

    @classmethod
    def from_store(
        cls,
        group_id: str,
        store: AssetStore,
        *,
        skill_source: "SkillResourceSource | None" = None,
    ) -> "CapabilityGroup[AppT]":
        if not isinstance(store, AssetStore):
            raise TypeError("store must be AssetStore")
        if skill_source is not None and not isinstance(skill_source, SkillResourceSource):
            raise TypeError("skill_source must implement SkillResourceSource")
        source = (
            AssetSkillResourceSource(group_id, store)
            if skill_source is None
            else skill_source
        )
        if source.id != group_id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        group = cls(group_id)
        group._store = store
        group._skill_source = source
        group._loaders.append(cast("CapabilityLoader[AppT]", _BuiltinDeclarationLoader()))
        return group

    @property
    def id(self) -> str:
        return self._id

    @property
    def skill_source(self) -> "SkillResourceSource | None":
        return self._skill_source

    def tool(
        self,
        function: Callable[..., object],
        *,
        name: "str | None" = None,
        revision: int = 1,
        effect: Literal["none", "replay_safe", "non_replay_safe"] = "non_replay_safe",
        plan_safe: bool = False,
    ) -> "Tool[AgentContext[AppT]]":
        """Register one ordinary model-visible Python tool."""
        _validate_revision(revision)
        _validate_tool_effect(effect, plan_safe)
        tool_name = name or function.__name__
        _validate_business_tool_name(tool_name)
        adapted = _adapt_tool(function, name=tool_name)
        self._contributions.append(
            CapabilityContribution.from_opaque(
                "tool",
                tool_name,
                adapted,
                revision=revision,
                semantic_config={"effect": effect, "plan_safe": plan_safe},
            )
        )
        return adapted

    def task(self, handler: "TaskNodeHandler[AppT]") -> "TaskNodeHandler[AppT]":
        """Register one application-owned TaskNode handler version."""
        task_type, task_version = _task_identity(handler)
        identity = f"{task_type}@{task_version}"
        contract: dict[str, JsonValue] = {
            "version": 1,
            "task_type": task_type,
            "task_version": task_version,
        }
        if any(
            value.kind == "task" and value.id == identity
            for value in self._contributions
        ):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._contributions.append(
            _SemanticContribution(
                "task",
                identity,
                capability_fingerprint("task", identity, contract),
                handler,
                contract,
            )
        )
        return handler

    def capability(
        self,
        capability: "AbstractCapability[AgentContext[AppT]]",
        *,
        semantic_id: "str | None" = None,
        revision: int = 1,
        semantic_config: "Mapping[str, JsonValue] | None" = None,
    ) -> "AbstractCapability[AgentContext[AppT]]":
        """Register one always-selected Pydantic runtime behavior capability."""
        _validate_revision(revision)
        capability_id = _capability_registration_id(capability, semantic_id)
        _validate_external_capability_id(capability_id)
        self._contributions.append(
            CapabilityContribution.from_opaque(
                "capability",
                capability_id,
                capability,
                revision=revision,
                semantic_id=capability_id,
                semantic_config=semantic_config,
            )
        )
        return capability

    def agent(
        self,
        name: str,
        *,
        model: str = "default",
        system_prompt: str = "",
        instructions: "str | Sequence[str]" = (),
        allow_tools: Sequence[str] = ("*",),
        allow_skills: Sequence[str] = ("*",),
        allow_subagents: Sequence[str] = ("*",),
        allow_capabilities: Sequence[str] = ("*",),
        usage_limits: "AgentUsageLimits | None" = None,
        planning: bool = False,
        thinking: ThinkingValue = False,
        tool_retries: int = 3,
        output_retries: int = 3,
        description: "str | None" = None,
    ) -> AgentSpec:
        """Register one declarative Agent before Runtime.open()."""
        values = (instructions,) if isinstance(instructions, str) else tuple(instructions)
        spec = AgentSpec(
            id=name,
            model=model,
            system_prompt=system_prompt,
            instructions=values,
            allow_tools=tuple(allow_tools),
            allow_skills=tuple(allow_skills),
            allow_subagents=tuple(allow_subagents),
            allow_capabilities=tuple(allow_capabilities),
            usage_limits=usage_limits,
            planning=planning,
            thinking=thinking,
            tool_retries=tool_retries,
            output_retries=output_retries,
            description=description,
        )
        self._contributions.append(_declaration_contribution("agent", spec))
        return spec

    def loader(self, loader: CapabilityLoader[AppT]) -> CapabilityLoader[AppT]:
        """Register one deterministic loader for the group's frozen Store snapshot."""
        loader_id = loader.id
        if not isinstance(loader_id, str) or not loader_id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if any(existing.id == loader_id for existing in self._loaders):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._loaders.append(loader)
        return loader

    async def freeze(self) -> "tuple[CapabilityContribution[AppT], ...]":
        """Freeze direct registrations and a metadata-stable Store snapshot."""
        contributions = list(tuple(self._contributions))
        loaders = tuple(self._loaders)
        store = self._store
        if store is not None:
            if not store.ready:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            metadata = await store.metadata_snapshot()
            entries = tuple(
                CapabilityLoadEntry(
                    info.key,
                    info.etag,
                    info.size,
                    info.metadata,
                )
                for info in metadata
            )
            context = CapabilityLoadContext(self._id, store, entries)
            for loader in loaders:
                loaded = await loader.load(context)
                if any(not isinstance(item, CapabilityContribution) for item in loaded):
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                contributions.extend(loaded)
        _validate_unique(contributions)
        generic = [item for item in contributions if item.kind == "capability"]
        declarations = sorted(
            (item for item in contributions if item.kind != "capability"),
            key=lambda item: (item.kind, item.id, item.fingerprint),
        )
        return tuple((*declarations, *generic))


class _BuiltinDeclarationLoader:
    @property
    def id(self) -> str:
        return "linktools-declarations-v1"

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        entries = context.list()
        directory_roots = tuple(
            sorted(
                entry.key.id[: -len("/SKILL.md")]
                for entry in entries
                if entry.key.kind == "skill" and entry.key.id.endswith("/SKILL.md")
            )
        )
        if any(not root for root in directory_roots):
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)
        _validate_skill_roots(directory_roots)
        directory_root_set = frozenset(directory_roots)
        flat_skill_ids = {
            entry.key.id
            for entry in entries
            if entry.key.kind == "skill"
            and not entry.key.id.endswith("/SKILL.md")
            and not _inside_skill_root(entry.key.id, directory_roots)
        }
        if directory_root_set.intersection(flat_skill_ids):
            raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)

        result: list[CapabilityContribution[object]] = []
        skill_codec = SkillSpecCodec()
        markdown_codec = SkillMarkdownSpecCodec()
        adapter = SkillMarkdownSpecAdapter()
        for entry in entries:
            key = entry.key
            if key.kind == "agent":
                value = AgentSpecCodec().decode(await context.read(key))
                if value.id != key.id:
                    raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
                result.append(_declaration_contribution("agent", value))
                continue
            if key.kind == "mcp":
                value = MCPServerSpecCodec().decode(await context.read(key))
                if value.id != key.id:
                    raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
                result.append(_declaration_contribution("mcp", value))
                continue
            if key.kind != "skill":
                continue
            if key.id.endswith("/SKILL.md"):
                logical_id = key.id[: -len("/SKILL.md")]
                value = adapter.to_logical(
                    logical_id,
                    markdown_codec.decode(await context.read(key)),
                )
                result.append(
                    _declaration_contribution(
                        "skill",
                        SkillDefinition(
                            value,
                            SkillSourceRef(context.group_id, logical_id),
                        ),
                    )
                )
                continue
            if _inside_skill_root(key.id, directory_roots):
                continue
            value = skill_codec.decode(await context.read(key))
            if value.id != key.id:
                raise AIError(ErrorCode.ASSET_CONTENT_MISMATCH)
            result.append(_declaration_contribution("skill", SkillDefinition(value)))
        return result


def _validate_skill_roots(roots: Sequence[str]) -> None:
    if len(set(roots)) != len(roots):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if other.startswith(f"{root}/") or root.startswith(f"{other}/"):
                raise AIError(ErrorCode.ASSET_LAYOUT_CONFLICT)


def _inside_skill_root(identifier: str, roots: Sequence[str]) -> bool:
    return any(identifier.startswith(f"{root}/") for root in roots)


def _declaration_contribution(
    kind: Literal["agent", "skill", "mcp"],
    value: AgentSpec | SkillDefinition | MCPServerSpec,
) -> CapabilityContribution[object]:
    if not (
        kind == "agent" and isinstance(value, AgentSpec)
        or kind == "skill" and isinstance(value, SkillDefinition)
        or kind == "mcp" and isinstance(value, MCPServerSpec)
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    identity = value.id
    semantic = contribution_semantic_contract(kind, identity, value)
    return CapabilityContribution(
        kind,
        identity,
        capability_fingerprint(kind, identity, semantic),
        value,
    )


def _adapt_tool(function: Callable[..., object], *, name: str) -> Tool:
    if not callable(function):
        raise TypeError("tool function must be callable")
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "tool requires AgentContext")

    @functools.wraps(function)
    async def invoke(
        ctx: PydanticRunContext[AgentContext[object]],
        *args: object,
        **kwargs: object,
    ) -> object:
        result = function(ctx.deps, *args, **kwargs)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    first = parameters[0].replace(
        annotation=PydanticRunContext[AgentContext[object]],
    )
    invoke.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=(first, *parameters[1:]),
    )
    return Tool(invoke, takes_ctx=True, name=name)


def contribution_semantic_contract(
    kind: ContributionKind,
    identity: str,
    value: ContributionSemanticValue,
    *,
    semantic_revision: "int | None" = None,
    semantic_id: "str | None" = None,
    semantic_config: "Mapping[str, JsonValue] | None" = None,
) -> "dict[str, JsonValue]":
    if semantic_config is not None and kind not in {"tool", "capability"}:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if kind == "tool" and isinstance(value, Tool):
        definition = value.tool_def
        contract: dict[str, JsonValue] = {
            "version": 1,
            "description": definition.description,
            "parameters": cast(JsonValue, definition.parameters_json_schema),
            "return_schema": cast(JsonValue, definition.return_schema),
            "strict": definition.strict,
            "metadata": cast(JsonValue, definition.metadata),
        }
        if semantic_revision is not None:
            contract["semantic_revision"] = semantic_revision
        if semantic_config is not None:
            try:
                contract.update(dict(ImmutableJsonMapping(semantic_config)))
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        return contract
    if kind == "agent" and isinstance(value, AgentSpec):
        return AgentSpecCodec().to_payload(value)
    if kind == "skill" and isinstance(value, SkillDefinition):
        return value.semantic_contract
    if kind == "mcp" and isinstance(value, MCPServerSpec):
        return MCPServerSpecCodec().to_payload(value)
    if kind == "capability" and isinstance(value, AbstractCapability):
        contract: dict[str, JsonValue] = {
            "version": 1,
            "revision": semantic_revision or 1,
            "defer_loading": value.defer_loading,
            "config": {},
        }
        if semantic_config is not None:
            try:
                contract["config"] = dict(ImmutableJsonMapping(semantic_config))
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        return contract
    if kind == "task" and isinstance(value, TaskNodeHandler):
        task_type, task_version = _task_identity(value)
        if identity != f"{task_type}@{task_version}":
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return {
            "version": 1,
            "task_type": task_type,
            "task_version": task_version,
        }
    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def capability_fingerprint(
    kind: ContributionKind,
    identity: str,
    semantic_contract: Mapping[str, JsonValue],
) -> str:
    return canonical_sha256(
        {
            "contract": "capability-fingerprint-v1",
            "kind": kind,
            "id": identity,
            "semantic": dict(semantic_contract),
        }
    )


def _task_identity(handler: object) -> tuple[str, int]:
    if not isinstance(handler, TaskNodeHandler):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    task_type = handler.type
    task_version = handler.version
    if (
        not isinstance(task_type, str)
        or _TASK_TYPE.fullmatch(task_type) is None
        or task_type.startswith(_RESERVED_TASK_TYPE_PREFIX)
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if (
        not isinstance(task_version, int)
        or isinstance(task_version, bool)
        or task_version < 1
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return task_type, task_version


def _validate_business_tool_name(value: str) -> None:
    if (
        not isinstance(value, str)
        or _TOOL_NAME.fullmatch(value) is None
        or value.startswith("mcp__")
        or value in _RESERVED_TOOL_NAMES
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _capability_registration_id(
    value: AbstractCapability[object],
    semantic_id: str | None = None,
) -> str:
    capability_id = value.id
    if capability_id is None:
        if value.defer_loading or semantic_id is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        capability_id = semantic_id
    elif semantic_id is not None and semantic_id != capability_id:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(capability_id, str) or not capability_id.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return capability_id



def _validate_external_capability_id(value: str) -> None:
    if value.startswith("linktools."):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_tool_effect(effect: str, plan_safe: bool) -> None:
    if effect not in {"none", "replay_safe", "non_replay_safe"}:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(plan_safe, bool):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_revision(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_fingerprint(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


def _validate_unique(values: Sequence[CapabilityContribution[object]]) -> None:
    seen: set[tuple[str, str]] = set()
    for value in values:
        identity = (value.kind, value.id)
        if identity in seen:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        seen.add(identity)


__all__ = [
    "CapabilityContribution",
    "CapabilityGroup",
    "CapabilityLoadContext",
    "CapabilityLoadEntry",
    "CapabilityLoader",
    "PLAN_SAFE_METADATA_KEY",
    "capability_fingerprint",
    "contribution_semantic_contract",
]
