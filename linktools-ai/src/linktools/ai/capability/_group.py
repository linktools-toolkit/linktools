#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability groups capture runtime candidate definitions before execution."""

import functools
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, cast, get_type_hints

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..asset import AssetStore, AssetStoreReader
from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentSpec,
    AgentUsageLimits,
    MCPServerSpec,
    RepositoryInstructionDocument,
    RepositoryInstructions,
    ThinkingValue,
    parse_mcp_tool_selector,
)
from ..task import TaskEffectResolution, TaskExpanderRef, TaskNodeContext, TaskNodeHandler
from ..storage import StorageRevision
from ..workspace import Sandbox, Workspace
from ._context import AgentContext
from ._contribution import CapabilityContribution, _freeze_contribution
from ._declaration import BuiltinDeclarationLoader
from ._loading import CapabilityLoadContext, CapabilityLoadEntry, CapabilityLoader
from ._skill import SkillDefinition
from ._task import TaskExpander
from ._tool_metadata import (
    tool_metadata,
    validate_tool_metadata,
)
from ._workspace import _workspace_tool_definitions

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.capability.group")


_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class _RegisteredTaskHandler(Generic[AppT]):
    handler: TaskNodeHandler[AppT]
    effect_policy: Literal["none", "replay_safe", "non_replay_safe"]
    output_type: object | None
    reconcile: (
        Callable[[TaskNodeContext[AppT]], Awaitable[TaskEffectResolution]] | None
    ) = field(default=None, repr=False, compare=False)

    @property
    def id(self) -> str:
        return self.handler.id

    @property
    def revision(self) -> int:
        return self.handler.revision

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
class CapabilityGroupCapture(Generic[AppT]):
    """Declarations and a versioned Asset reader captured from one revision."""

    group_id: str
    contributions: tuple[CapabilityContribution[AppT], ...]
    source_revision: "StorageRevision | None"
    workspace: "Workspace | None"
    instructions: RepositoryInstructions
    _asset_reader: "AssetStoreReader | None" = field(repr=False, compare=False)
    sandbox: "Sandbox | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("capture group_id must be non-empty")
        contributions = tuple(self.contributions)
        if any(
            not isinstance(value, CapabilityContribution)
            for value in contributions
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if not isinstance(self.instructions, RepositoryInstructions):
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
        """Return read-only access to this capture's source resources."""
        return self._asset_reader

    async def verify_source_revision(self) -> None:
        """Fail if the declaration source changed after this capture was made."""
        if self._asset_reader is None:
            return
        if not isinstance(self.source_revision, StorageRevision):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if await self._asset_reader.current_revision() != self.source_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)


class CapabilityGroup(Generic[AppT]):
    """Register and capture one named set of runtime candidate definitions."""

    def __init__(
        self,
        group_id: str,
        *,
        assets: "AssetStore | None" = None,
        workspace: "Workspace | None" = None,
        sandbox: "Sandbox | None" = None,
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
        self._sandbox = sandbox
        self._loaders: dict[str, CapabilityLoader[AppT]] = {}
        self._contributions: list[CapabilityContribution[AppT]] = []
        if workspace is not None:
            self._contributions.extend(
                CapabilityContribution.from_opaque("tool", tool.name, tool)
                for tool in _workspace_tool_definitions(workspace)
            )
        if assets is not None:
            for kind in ("agent", "skill", "mcp", "rule"):
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
    def sandbox(self) -> "Sandbox | None":
        return self._sandbox

    @property
    def assets(self) -> "AssetStore | None":
        """Return the explicit AssetStore used by this capability group."""
        return self._store

    def tool(
        self,
        function: Callable[..., object],
        *,
        name: "str | None" = None,
        revision: int = 1,
        effect_policy: Literal["none", "replay_safe", "non_replay_safe"] = "non_replay_safe",
        plan_safe: bool = False,
    ) -> "Tool[AgentContext[AppT]]":
        """Register one ordinary model-visible Python tool."""
        _validate_revision(revision)
        _validate_tool_effect_policy(effect_policy, plan_safe)
        tool_name = name or function.__name__
        _validate_business_tool_name(tool_name)
        adapted = _adapt_tool(function, name=tool_name)
        adapted.metadata = tool_metadata(
            effect_policy=effect_policy,
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
        effect_policy: Literal["none", "replay_safe", "non_replay_safe"] = "non_replay_safe",
        output_type: object | None = None,
        reconcile: (
            "Callable[[TaskNodeContext[AppT]], Awaitable[TaskEffectResolution]] | None"
        ) = None,
    ) -> "TaskNodeHandler[AppT]":
        """Register one application-owned TaskNode handler revision."""
        if effect_policy not in {"none", "replay_safe", "non_replay_safe"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if reconcile is not None and not callable(reconcile):
            raise TypeError("reconcile must be callable")
        registered = _RegisteredTaskHandler(
            handler,
            effect_policy,
            output_type,
            reconcile,
        )
        contribution = cast(
            "CapabilityContribution[AppT]",
            CapabilityContribution.from_task(registered),
        )
        if any(
            value.kind == "task"
            and value.id == contribution.id
            and value.revision == contribution.revision
            for value in self._contributions
        ):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._contributions.append(contribution)
        return handler

    def task_expander(self, expander: TaskExpander) -> TaskExpanderRef:
        """Register one pure application-owned TaskGraph expander revision."""
        contribution = cast(
            "CapabilityContribution[AppT]",
            CapabilityContribution.from_task_expander(expander),
        )
        if any(
            value.kind == "task_expander"
            and value.id == contribution.id
            and value.revision == contribution.revision
            for value in self._contributions
        ):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._contributions.append(contribution)
        return TaskExpanderRef(contribution.id, contribution.revision)


    def runtime_capability(
        self,
        capability: "AbstractCapability[AgentContext[AppT]]",
        *,
        id: "str | None" = None,
        revision: int = 1,
        config: "Mapping[str, JsonValue] | None" = None,
    ) -> "AbstractCapability[AgentContext[AppT]]":
        """Register one always-selected Pydantic runtime behavior capability."""
        _validate_revision(revision)
        capability_id = _capability_registration_id(capability, id)
        _validate_external_capability_id(capability_id)
        self._contributions.append(
            CapabilityContribution.from_opaque(
                "runtime_capability",
                capability_id,
                capability,
                revision=revision,
                config=config,
            )
        )
        return capability

    def agent(
        self,
        name: str,
        *,
        revision: int = 1,
        model_route: str = "default",
        system_prompt: str = "",
        instructions: "str | Sequence[str]" = (),
        allow_tools: Sequence[str] = ("*",),
        allow_skills: Sequence[str] = ("*",),
        allow_subagents: Sequence[str] = ("*",),
        allow_runtime_capabilities: Sequence[str] = ("*",),
        usage_limits: "AgentUsageLimits | None" = None,
        planning: bool = False,
        thinking: ThinkingValue = False,
        tool_retries: int = AgentSpec.DEFAULT_TOOL_RETRIES,
        output_retries: int = AgentSpec.DEFAULT_OUTPUT_RETRIES,
        description: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> AgentSpec:
        """Register one declarative Agent before Runtime.open()."""
        _validate_revision(revision)
        values = (instructions,) if isinstance(instructions, str) else tuple(instructions)
        spec = AgentSpec(
            id=name,
            model_route=model_route,
            system_prompt=system_prompt,
            instructions=values,
            allow_tools=tuple(allow_tools),
            allow_skills=tuple(allow_skills),
            allow_subagents=tuple(allow_subagents),
            allow_runtime_capabilities=tuple(allow_runtime_capabilities),
            usage_limits=usage_limits,
            planning=planning,
            thinking=thinking,
            tool_retries=tool_retries,
            output_retries=output_retries,
            description=description,
            metadata={} if metadata is None else metadata,
            revision=revision,
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

    async def capture(self) -> "CapabilityGroupCapture[AppT]":
        """Capture registrations and declarations at one Asset revision."""
        contributions = list(tuple(self._contributions))
        instruction_documents: list[RepositoryInstructionDocument] = []
        loaders = tuple(self._loaders.items())
        store = self._store
        source_revision: StorageRevision | None = None
        asset_reader: AssetStoreReader | None = None
        if store is not None:
            context = await CapabilityLoadContext.capture(self._id, store)
            for _kind, loader in loaders:
                loaded = await loader.load(context)
                for value in loaded:
                    _validate_skill_source(value, context)
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
                    elif isinstance(value, RepositoryInstructionDocument):
                        instruction_documents.append(value)
                        continue
                    else:
                        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                    contributions.append(item)
            await context.verify()
            source_revision = context.source_revision
            asset_reader = context.asset_reader
        elif loaders:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        captured_items = tuple(
            _freeze_contribution(item) for item in contributions
        )
        _validate_unique(captured_items)
        generic = [item for item in captured_items if item.kind == "runtime_capability"]
        declarations = sorted(
            (item for item in captured_items if item.kind != "runtime_capability"),
            key=lambda item: (item.kind, item.id, item.revision),
        )
        capture_contributions = tuple((*declarations, *generic))
        capture = CapabilityGroupCapture(
            self._id,
            capture_contributions,
            source_revision,
            self._workspace,
            RepositoryInstructions(tuple(instruction_documents)),
            asset_reader,
            sandbox=self._sandbox,
        )
        _logger.info(
            "capability group captured: group=%s contributions=%d instructions=%d "
            "source_revision=%s",
            self._id,
            len(capture_contributions),
            len(instruction_documents),
            None if source_revision is None else source_revision.value,
        )
        return capture


def _validate_skill_source(
    value: object,
    context: CapabilityLoadContext,
) -> None:
    if isinstance(value, CapabilityContribution):
        if value.kind != "skill" or not isinstance(value.value, SkillDefinition):
            return
        definition = value.value
    elif isinstance(value, SkillDefinition):
        definition = value
    else:
        return
    source_ref = definition.source_ref
    if source_ref is None:
        return
    if source_ref.asset_source_id != context.group_id:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    versions = tuple(item.asset for item in source_ref.resource_versions)
    if not versions:
        return
    try:
        captured = context.bind_versions(tuple(ref.key for ref in versions))
    except AIError as error:
        if error.code is ErrorCode.SNAPSHOT_CONFLICT:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        raise
    if versions != captured:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _adapt_tool(function: Callable[..., object], *, name: str) -> Tool:
    if not callable(function):
        raise TypeError("tool function must be callable")
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    if not parameters:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "tool requires AgentContext")

    invoke: Callable[..., object]
    if inspect.iscoroutinefunction(function):
        async def invoke_async(
            runtime_context: PydanticRunContext[AgentContext[object]],
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
            return await cast(Callable[..., Awaitable[object]], function)(
                runtime_context.deps,
                *args,
                **kwargs,
            )

        invoke = functools.wraps(function)(invoke_async)
    else:
        def invoke_sync(
            runtime_context: PydanticRunContext[AgentContext[object]],
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
            return function(runtime_context.deps, *args, **kwargs)

        invoke = functools.wraps(function)(invoke_sync)

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
    explicit_id: str | None = None,
) -> str:
    capability_id = value.id
    if capability_id is None:
        if value.defer_loading or explicit_id is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        capability_id = explicit_id
    elif explicit_id is not None and explicit_id != capability_id:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(capability_id, str) or not capability_id.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return capability_id



def _validate_external_capability_id(value: str) -> None:
    if value.startswith("linktools."):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_tool_effect_policy(effect_policy: str, plan_safe: bool) -> None:
    if effect_policy not in {"none", "replay_safe", "non_replay_safe"}:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(plan_safe, bool):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_revision(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_unique(values: Sequence[CapabilityContribution[object]]) -> None:
    seen: set[tuple[str, str, int]] = set()
    for value in values:
        identity = (value.kind, value.id, value.revision)
        if identity in seen:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        seen.add(identity)


__all__ = ["CapabilityGroup", "CapabilityGroupCapture"]
