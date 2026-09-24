#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability groups freeze runtime candidate definitions before execution."""

import functools
import hashlib
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar, cast, get_type_hints

from linktools.core import environ
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..asset import (
    AssetInfo,
    AssetKey,
    AssetStore,
    AssetStoreReader,
    AssetVersionRef,
)
from ..core import ImmutableJsonMapping, JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
    MCPServerSpec,
    MCPServerSpecCodec,
    ThinkingValue,
    canonicalize_json_schema,
    canonicalize_pydantic_model_schema,
    capability_identity_payload,
    parse_mcp_tool_selector,
)
from ..task import TaskEffectResolution, TaskExpanderRef, TaskNodeContext, TaskNodeHandler
from ..storage import ObjectRef, ObjectStore, StorageRevision
from ..workspace import Workspace
from ._context import AgentContext
from ._skill import SkillDefinition
from ._task import TaskExpander
from ._tool_semantic import (
    tool_semantic_metadata,
    validate_tool_semantic_metadata,
)
from ._workspace import _workspace_tool_definitions

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.capability.group")


ContributionKind = Literal[
    "tool",
    "agent",
    "skill",
    "mcp",
    "capability",
    "task",
    "task_expander",
]
ContributionSemanticValue: TypeAlias = (
    Tool
    | AgentSpec
    | SkillDefinition
    | MCPServerSpec
    | AbstractCapability
    | TaskNodeHandler[object]
    | TaskExpander
)
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_TASK_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RESERVED_TASK_TYPE_PREFIX = "linktools.ai."
_RESERVED_EXPANDER_ID_PREFIX = "linktools.ai."


@dataclass(frozen=True, slots=True)
class _RegisteredTaskHandler(Generic[AppT]):
    handler: TaskNodeHandler[AppT]
    effect: Literal["none", "replay_safe", "non_replay_safe"]
    output: object | None
    reconcile: (
        Callable[[TaskNodeContext[AppT]], Awaitable[TaskEffectResolution]] | None
    ) = field(default=None, repr=False, compare=False)

    @property
    def type(self) -> str:
        return self.handler.type

    @property
    def version(self) -> int:
        return self.handler.version

    def normalize(
        self,
        input: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        return self.handler.normalize(input)

    async def run(self, context: TaskNodeContext[AppT]) -> JsonValue:
        return await self.handler.run(context)

    async def cancel(self, context: TaskNodeContext[AppT]) -> None:
        await self.handler.cancel(context)


@dataclass(frozen=True, slots=True)
class CapabilityContribution(Generic[AppT]):
    kind: ContributionKind
    id: str
    fingerprint: str
    value: (
        "Tool[AgentContext[AppT]] | AgentSpec | SkillDefinition | MCPServerSpec | "
        "AbstractCapability[AgentContext[AppT]] | TaskNodeHandler[AppT] | TaskExpander"
    )

    def __post_init__(self) -> None:
        if self.kind not in {
            "tool",
            "agent",
            "skill",
            "mcp",
            "capability",
            "task",
            "task_expander",
        }:
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
        if self.kind == "task_expander" and not isinstance(self.value, TaskExpander):
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
        if self.kind == "task_expander":
            expander = cast(TaskExpander, self.value)
            expander_id, expander_version = _expander_identity(expander)
            if self.id != f"{expander_id}@{expander_version}":
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
        revision: "int | None" = None,
        semantic_id: "str | None" = None,
        semantic_config: "Mapping[str, JsonValue] | None" = None,
    ) -> "CapabilityContribution[AppT]":
        """Create an opaque Python Tool or Capability from its public semantic inputs."""
        if kind not in {"tool", "capability"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if revision is not None:
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

    @classmethod
    def from_declaration(
        cls,
        value: AgentSpec | SkillDefinition | MCPServerSpec,
    ) -> "CapabilityContribution[object]":
        """Create a declaration contribution from its public semantic value."""
        if isinstance(value, AgentSpec):
            kind: Literal["agent", "skill", "mcp"] = "agent"
        elif isinstance(value, SkillDefinition):
            kind = "skill"
        elif isinstance(value, MCPServerSpec):
            kind = "mcp"
        else:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        contract = contribution_semantic_contract(kind, value.id, value)
        return _SemanticContribution(
            kind,
            value.id,
            capability_fingerprint(kind, value.id, contract),
            value,
            contract,
        )

    @classmethod
    def from_mcp_contract(
        cls,
        contract: Mapping[str, JsonValue],
    ) -> "CapabilityContribution[object]":
        """Restore an MCP contribution from its already frozen semantic contract."""
        if not isinstance(contract, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value, _resource_versions = MCPServerSpecCodec().from_frozen_payload(
            cast("Mapping[str, object]", contract)
        )
        return _SemanticContribution(
            "mcp",
            value.id,
            capability_fingerprint("mcp", value.id, contract),
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
class _CapabilityAssetReader:
    _store: AssetStore = field(repr=False, compare=False)
    _revision: StorageRevision
    _versions: Mapping[AssetKey, AssetVersionRef] = field(repr=False, compare=False)

    async def _verify(self) -> None:
        if await self._store.current_revision() != self._revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)

    async def current_revision(self) -> StorageRevision:
        await self._verify()
        return self._revision

    async def get(self, key: AssetKey) -> "bytes | None":
        await self._verify()
        value = await self._store.get(key)
        await self._verify()
        return value

    async def get_many(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[bytes | None, ...]":
        await self._verify()
        values = await self._store.get_many(keys)
        await self._verify()
        return values

    async def local_paths(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[Path | None, ...]":
        await self._verify()
        values = await self._store.local_paths(keys)
        await self._verify()
        return values

    async def metadata_snapshot(self) -> "tuple[AssetInfo, ...]":
        await self._verify()
        values = await self._store.metadata_snapshot()
        await self._verify()
        return values

    async def resolve_versions(
        self,
        keys: Sequence[AssetKey],
    ) -> "tuple[AssetVersionRef, ...]":
        result: list[AssetVersionRef] = []
        for key in keys:
            ref = self._versions.get(key)
            if ref is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            result.append(ref)
        return tuple(result)

    async def read_versions(
        self,
        refs: Sequence[AssetVersionRef],
    ) -> "tuple[bytes, ...]":
        return await self._store.read_versions(refs)

    async def snapshot(
        self,
        keys: Sequence[AssetKey],
        *,
        object_store: ObjectStore,
        expected_revision: "StorageRevision | None" = None,
    ) -> ObjectRef:
        if (
            expected_revision is not None
            and expected_revision != self._revision
        ):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        return await self._store.snapshot(
            keys,
            object_store=object_store,
            expected_revision=self._revision,
        )


@dataclass(frozen=True, slots=True)
class CapabilityGroupSnapshot(Generic[AppT]):
    """One parsed declaration set bound to its source revision."""

    group_id: str
    contributions: tuple[CapabilityContribution[AppT], ...]
    source_revision: "StorageRevision | None"
    workspace: "Workspace | None"
    _asset_reader: "AssetStoreReader | None" = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("snapshot group_id must be non-empty")
        contributions = tuple(self.contributions)
        if any(
            not isinstance(value, CapabilityContribution)
            for value in contributions
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if (self._asset_reader is None) != (self.source_revision is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            self.source_revision is not None
            and not isinstance(self.source_revision, StorageRevision)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object.__setattr__(self, "contributions", contributions)

    @property
    def asset_reader(self) -> "AssetStoreReader | None":
        """Return read-only access to this snapshot's source resources."""
        return self._asset_reader

    async def verify_source_revision(self) -> None:
        """Fail if the declaration source changed after this snapshot was made."""
        if self._asset_reader is None:
            return
        if not isinstance(self.source_revision, StorageRevision):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if await self._asset_reader.current_revision() != self.source_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)


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
        self._read_keys: set[AssetKey] = set()
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
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        cached = self._cache.get(key)
        if cached is not None:
            self._read_keys.add(key)
            return cached
        value = await self._store.get(key)
        if value is None:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        data = bytes(value)
        if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.etag:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        self._cache[key] = data
        self._read_keys.add(key)
        return data

    async def read_many(self, keys: Sequence[AssetKey]) -> "tuple[bytes, ...]":
        """Read captured assets once, preserving the requested order."""
        requested = tuple(dict.fromkeys(keys))
        if any(key not in self._by_key for key in requested):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        pending = tuple(
            self._by_key[key]
            for key in requested
            if key not in self._cache
        )
        if not pending:
            return tuple(self._cache[key] for key in keys)
        values = await self._store.get_many(tuple(entry.key for entry in pending))
        for entry, value in zip(pending, values, strict=True):
            if (
                value is None
            ):
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
            data = bytes(value)
            if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.etag:
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
            self._cache[entry.key] = data
            self._read_keys.add(entry.key)
        return tuple(self._cache[key] for key in keys)

    async def verify(self) -> None:
        """Recheck metadata for the captured assets read by loaders."""
        if not self._read_keys:
            return
        current = {
            info.key: info
            for info in await self._store.metadata_snapshot()
            if info.key in self._read_keys
        }
        for key in self._read_keys:
            entry = self._by_key[key]
            info = current.get(key)
            if (
                info is None
                or info.etag != entry.etag
                or info.size != entry.size
                or info.metadata != entry.metadata
            ):
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)


class CapabilityLoader(Protocol[AppT]):
    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> "Sequence[CapabilityContribution[AppT] | AgentSpec | SkillDefinition | MCPServerSpec]": ...


class CapabilityGroup(Generic[AppT]):
    """Register and freeze one named set of runtime candidate definitions."""

    def __init__(
        self,
        group_id: str,
        *,
        assets: "AssetStore | None" = None,
        workspace: "Workspace | None" = None,
    ) -> None:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("capability group id must be a non-empty string")
        if assets is not None and not isinstance(assets, AssetStore):
            raise TypeError("assets must be AssetStore")
        if workspace is not None and not isinstance(workspace, Workspace):
            raise TypeError("workspace must be Workspace")
        self._id = group_id
        self._store = assets
        self._workspace = workspace
        self._loaders: dict[str, CapabilityLoader[AppT]] = {}
        self._contributions: list[CapabilityContribution[AppT]] = []
        if workspace is not None:
            self._contributions.extend(
                CapabilityContribution.from_opaque("tool", tool.name, tool)
                for tool in _workspace_tool_definitions(workspace)
            )
        if assets is not None:
            from ._declaration import BuiltinDeclarationLoader

            for kind in ("agent", "skill", "mcp"):
                self._loaders[kind] = cast(
                    "CapabilityLoader[AppT]",
                    BuiltinDeclarationLoader(kind),
                )

    @property
    def id(self) -> str:
        return self._id

    @property
    def workspace(self) -> "Workspace | None":
        return self._workspace

    @property
    def asset_store(self) -> "AssetStore | None":
        """Return the explicit AssetStore used by this capability group."""
        return self._store

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
        adapted.metadata = tool_semantic_metadata(
            effect=effect,
            plan_safe=plan_safe,
            tool_class="business",
            base=adapted.metadata,
        )
        self._contributions.append(
            CapabilityContribution.from_opaque(
                "tool",
                tool_name,
                adapted,
                revision=revision,
            )
        )
        return adapted

    def task(
        self,
        handler: "TaskNodeHandler[AppT]",
        *,
        effect: Literal["none", "replay_safe", "non_replay_safe"] = "non_replay_safe",
        output: object | None = None,
        reconcile: (
            "Callable[[TaskNodeContext[AppT]], Awaitable[TaskEffectResolution]] | None"
        ) = None,
    ) -> "TaskNodeHandler[AppT]":
        """Register one application-owned TaskNode handler version."""
        if effect not in {"none", "replay_safe", "non_replay_safe"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if reconcile is not None and not callable(reconcile):
            raise TypeError("reconcile must be callable")
        registered = _RegisteredTaskHandler(
            handler,
            effect,
            output,
            reconcile,
        )
        task_type, task_version = _task_identity(registered)
        identity = f"{task_type}@{task_version}"
        contract: dict[str, JsonValue] = {
            "version": 1,
            "task_type": task_type,
            "task_version": task_version,
            "effect": _task_effect(registered),
            "output": _task_output_contract(registered),
            "reconcile": registered.reconcile is not None,
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
                registered,
                contract,
            )
        )
        return handler

    def task_expander(self, expander: TaskExpander) -> TaskExpanderRef:
        """Register one pure application-owned TaskGraph expander version."""
        expander_id, expander_version = _expander_identity(expander)
        identity = f"{expander_id}@{expander_version}"
        contract: dict[str, JsonValue] = {
            "version": 1,
            "expander_id": expander_id,
            "expander_version": expander_version,
        }
        if any(
            value.kind == "task_expander" and value.id == identity
            for value in self._contributions
        ):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._contributions.append(
            _SemanticContribution(
                "task_expander",
                identity,
                capability_fingerprint("task_expander", identity, contract),
                expander,
                contract,
            )
        )
        return TaskExpanderRef(expander_id, expander_version)

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
        tool_retries: int = AgentSpec.DEFAULT_TOOL_RETRIES,
        output_retries: int = AgentSpec.DEFAULT_OUTPUT_RETRIES,
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
        self._contributions.append(CapabilityContribution.from_declaration(spec))
        return spec

    def loader(
        self,
        kind: str,
        loader: CapabilityLoader[AppT],
    ) -> CapabilityLoader[AppT]:
        """Register the sole loader for one input Asset kind, replacing its slot."""
        if not isinstance(kind, str) or not kind.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if not callable(getattr(loader, "load", None)):
            raise TypeError("loader must implement load")
        source_kind = getattr(loader, "source_kind", None)
        if source_kind is not None and source_kind != kind:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._loaders[kind] = loader
        return loader

    async def freeze(self) -> "CapabilityGroupSnapshot[AppT]":
        """Freeze direct registrations and declarations at one source revision."""
        contributions = list(tuple(self._contributions))
        loaders = tuple(self._loaders.items())
        store = self._store
        source_revision: StorageRevision | None = None
        if store is not None:
            if not store.ready:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            source_revision = await store.current_revision()
            if not isinstance(source_revision, StorageRevision):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
            for kind, loader in loaders:
                if getattr(loader, "source_kind", kind) != kind:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                loaded = await loader.load(context)
                normalized: list[CapabilityContribution[AppT]] = []
                for value in loaded:
                    if isinstance(
                        value,
                        (AgentSpec, SkillDefinition, MCPServerSpec),
                    ):
                        item = cast(
                            "CapabilityContribution[AppT]",
                            CapabilityContribution.from_declaration(value),
                        )
                    elif isinstance(value, CapabilityContribution):
                        item = cast("CapabilityContribution[AppT]", value)
                    else:
                        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                    if item.kind != "skill":
                        normalized.append(item)
                        continue
                    skill = cast(SkillDefinition, item.value)
                    if skill.source_ref is not None and (
                        skill.source_ref.source_id != self._id
                        or skill.source_ref.frozen
                    ):
                        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                    normalized.append(item)
                contributions.extend(normalized)
            await context.verify()
            if await store.current_revision() != source_revision:
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        elif loaders:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        frozen = tuple(_freeze_contribution(item) for item in contributions)
        _validate_unique(frozen)
        generic = [item for item in frozen if item.kind == "capability"]
        declarations = sorted(
            (item for item in frozen if item.kind != "capability"),
            key=lambda item: (item.kind, item.id, item.fingerprint),
        )
        frozen_contributions = tuple((*declarations, *generic))
        snapshot = CapabilityGroupSnapshot(
            self._id,
            frozen_contributions,
            source_revision,
            self._workspace,
            None
            if store is None
            else _CapabilityAssetReader(
                store,
                cast(StorageRevision, source_revision),
                {
                    info.key: AssetVersionRef(
                        info.key,
                        info.root_digest,
                        info.revision,
                        info.etag,
                        info.size,
                    )
                    for info in metadata
                },
            ),
        )
        _logger.info(
            "capability group frozen: group=%s contributions=%d source_revision=%s",
            self._id,
            len(frozen_contributions),
            None if source_revision is None else source_revision.value,
        )
        return snapshot


def _freeze_contribution(
    value: CapabilityContribution[AppT],
) -> CapabilityContribution[AppT]:
    if isinstance(value, _SemanticContribution):
        return value
    return _SemanticContribution(
        value.kind,
        value.id,
        value.fingerprint,
        value.value,
        value.semantic_contract,
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
    invoke.__annotations__ = {
        **get_type_hints(function, include_extras=True),
        parameters[0].name: first.annotation,
    }
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
    if semantic_config is not None and kind != "capability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if kind == "tool" and isinstance(value, Tool):
        definition = value.tool_def
        validate_tool_semantic_metadata(
            definition.metadata,
            require_effect=True,
            require_tool_class=True,
        )
        contract: dict[str, JsonValue] = {
            "version": 1,
            "description": definition.description,
            "parameters": cast(JsonValue, definition.parameters_json_schema),
            "return_schema": cast(JsonValue, definition.return_schema),
            "strict": definition.strict,
            "metadata": cast(JsonValue, definition.metadata),
        }
        if value.max_retries is not None:
            contract["max_retries"] = value.max_retries
        if definition.sequential:
            contract["sequential"] = True
        if definition.kind != "function":
            contract["kind"] = definition.kind
        if definition.timeout is not None:
            contract["timeout"] = float(definition.timeout)
        if definition.defer_loading:
            contract["defer_loading"] = True
        if definition.include_return_schema is not None:
            contract["include_return_schema"] = definition.include_return_schema
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
            "effect": _task_effect(value),
            "output": _task_output_contract(value),
        }
    if kind == "task_expander" and isinstance(value, TaskExpander):
        expander_id, expander_version = _expander_identity(value)
        if identity != f"{expander_id}@{expander_version}":
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return {
            "version": 1,
            "expander_id": expander_id,
            "expander_version": expander_version,
        }
    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def capability_fingerprint(
    kind: ContributionKind,
    identity: str,
    semantic_contract: Mapping[str, JsonValue],
) -> str:
    return canonical_sha256(
        capability_identity_payload(kind, identity, semantic_contract)
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


def _task_effect(handler: object) -> str:
    effect = getattr(handler, "effect", "none")
    if effect not in {"none", "replay_safe", "non_replay_safe"}:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return effect


def _task_output_contract(handler: object) -> JsonValue:
    output = getattr(handler, "output", None)
    if output is None:
        return {"kind": "json"}
    model_schema = getattr(output, "model_json_schema", None)
    if not callable(model_schema):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    schema = (
        canonicalize_pydantic_model_schema(output)
        if isinstance(output, type) and issubclass(output, BaseModel)
        else canonicalize_json_schema(model_schema())
    )
    return {
        "kind": "schema",
        "schema": schema,
    }


def _expander_identity(expander: object) -> tuple[str, int]:
    if not isinstance(expander, TaskExpander):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    try:
        reference = TaskExpanderRef(expander.id, expander.version)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if reference.id.startswith(_RESERVED_EXPANDER_ID_PREFIX):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return reference.id, reference.version


def _validate_business_tool_name(value: str) -> None:
    if (
        not isinstance(value, str)
        or _TOOL_NAME.fullmatch(value) is None
        or value.startswith(("linktools.", "mcp__"))
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if parse_mcp_tool_selector(value) is not None:
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
    "capability_fingerprint",
    "contribution_semantic_contract",
]
