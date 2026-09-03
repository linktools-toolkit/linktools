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

from ..asset import AssetInfo, AssetKey, AssetStore
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
    SkillSpec,
    SkillSpecCodec,
    ThinkingValue,
)
from ..task import TaskNodeHandler
from ._context import RunContext
from ._names import SKILL_TOOL_NAMES, SUBAGENT_TOOL_NAMES
from ._skill import SkillDefinition
from ._skill_source import AssetSkillResourceSource, SkillResourceSource, SkillSourceRef

AppT = TypeVar("AppT")


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
_RESERVED_CAPABILITY_IDS = frozenset(
    {
        "workspace-filesystem",
        "workspace-shell",
        "workspace-sandbox",
        "linktools-skill",
        "linktools-memory",
        "linktools-planning",
        "linktools-subagent",
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
    value: "Tool[RunContext[AppT]] | AgentSpec | SkillDefinition | MCPServerSpec | AbstractCapability[RunContext[AppT]] | TaskNodeHandler[AppT]"

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
            capability_id = capability.id  # type: ignore[attr-defined]
            if not isinstance(capability_id, str) or capability_id != self.id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability.defer_loading:
                raise AIError(
                    ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                    safe_details={"capability_id": capability_id, "reason": "deferred_loading_not_supported"},
                )
            _validate_external_capability_id(capability_id)
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
        value: "Tool[RunContext[AppT]] | AbstractCapability[RunContext[AppT]]",
        *,
        revision: int = 1,
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


class CapabilityLoadContext:
    def __init__(
        self,
        group_id: str,
        store: AssetStore,
        entries: Sequence[AssetInfo],
    ) -> None:
        self._group_id = group_id
        self._store = store
        self._entries = tuple(entries)
        self._by_key = {entry.key: entry for entry in self._entries}
        if len(self._by_key) != len(self._entries):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def entries(self) -> tuple[AssetInfo, ...]:
        return self._entries

    async def read(self, key: AssetKey) -> bytes:
        entry = self._by_key.get(key)
        if entry is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        value = await self._store.get(key)
        if value is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        data = bytes(value)
        if hashlib.sha256(data).hexdigest() != entry.etag:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
    ) -> "Tool[RunContext[AppT]]":
        """Register one ordinary model-visible Python tool."""
        _validate_revision(revision)
        tool_name = name or function.__name__
        _validate_business_tool_name(tool_name)
        adapted = _adapt_tool(function, name=tool_name)
        self._contributions.append(
            CapabilityContribution.from_opaque(
                "tool",
                tool_name,
                adapted,
                revision=revision,
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
        capability: "AbstractCapability[RunContext[AppT]]",
        *,
        revision: int = 1,
        semantic_config: "Mapping[str, JsonValue] | None" = None,
    ) -> "AbstractCapability[RunContext[AppT]]":
        """Register one always-selected Pydantic runtime behavior capability."""
        _validate_revision(revision)
        try:
            capability_id = capability.id  # type: ignore[attr-defined]
        except AttributeError as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        _validate_external_capability_id(capability_id)
        self._contributions.append(
            CapabilityContribution.from_opaque(
                "capability",
                capability_id,
                capability,
                revision=revision,
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
        usage_limits: "AgentUsageLimits | None" = None,
        planning: bool = False,
        thinking: ThinkingValue = False,
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
            usage_limits=usage_limits,
            planning=planning,
            thinking=thinking,
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
            before = await store.current_revision()
            entries = tuple(await _list_all(store))
            context = CapabilityLoadContext(self._id, store, entries)
            for loader in loaders:
                loaded = await loader.load(context)
                if any(not isinstance(item, CapabilityContribution) for item in loaded):
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                contributions.extend(loaded)
            after = await store.current_revision()
            if before != after:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        _validate_unique(contributions)
        return tuple(
            sorted(contributions, key=lambda item: (item.kind, item.id, item.fingerprint))
        )


async def _list_all(store: AssetStore) -> "tuple[AssetInfo, ...]":
    values: list[AssetInfo] = []
    cursor: str | None = None
    while True:
        page = await store.list_info(cursor=cursor, limit=200)
        values.extend(page.items)
        if page.next_cursor is None:
            return tuple(values)
        cursor = page.next_cursor


class _BuiltinDeclarationLoader:
    @property
    def id(self) -> str:
        return "linktools-declarations-v2"

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[object]]":
        entries = tuple(sorted(context.entries, key=lambda item: (item.key.kind, item.key.id)))
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
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "tool requires RunContext")

    @functools.wraps(function)
    async def invoke(
        ctx: PydanticRunContext[RunContext[object]],
        *args: object,
        **kwargs: object,
    ) -> object:
        result = function(ctx.deps, *args, **kwargs)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    first = parameters[0].replace(
        annotation=PydanticRunContext[RunContext[object]],
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
    semantic_config: "Mapping[str, JsonValue] | None" = None,
) -> "dict[str, JsonValue]":
    if semantic_config is not None and kind != "capability":
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
        return contract
    if kind == "agent" and isinstance(value, AgentSpec):
        return AgentSpecCodec().to_payload(value)
    if kind == "skill" and isinstance(value, SkillDefinition):
        return value.semantic_contract
    if kind == "mcp" and isinstance(value, MCPServerSpec):
        return MCPServerSpecCodec().to_payload(value)
    if kind == "capability" and isinstance(value, AbstractCapability):
        contract = {"version": 1}
        if semantic_revision is not None:
            contract["semantic_revision"] = semantic_revision
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


def _validate_external_capability_id(value: str) -> None:
    if value.startswith("mcp__") or value in _RESERVED_CAPABILITY_IDS:
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
    "CapabilityLoader",
    "capability_fingerprint",
    "contribution_semantic_contract",
]
