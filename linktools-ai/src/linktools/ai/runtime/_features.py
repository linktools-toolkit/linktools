#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime feature declarations and their immutable captured composition."""

import functools
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Literal, TypeVar, get_type_hints

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.tools import RunContext as PydanticRunContext

from ..asset import AssetStoreReader
from ..core import AuthorizationPolicy, ImmutableJsonMapping, JsonValue, PromptLimits
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import (
    AgentSpec,
    AgentUsageLimits,
    MCPServerSpec,
    RepositoryInstructionDocument,
    RepositoryInstructions,
    ThinkingValue,
    parse_mcp_tool_selector,
)
from ..storage import StorageRevision
from ..workspace import Sandbox, Workspace
from ..capability import (
    AgentContext,
    AssetCapability,
    CapabilityContribution,
    CapabilityLoadContext,
    ToolEffectPolicy,
    ToolResponsesCapability,
    WorkspaceCapability,
    capture_contributions,
    is_tool_effect_policy,
    tool_metadata,
    workspace_tool_declarations,
)

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.runtime.features")


_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class RuntimeFeaturesCapture(Generic[AppT]):
    """Declarations and a versioned Asset reader captured from one revision."""

    group_id: str
    contributions: tuple[CapabilityContribution[AppT], ...]
    source_revision: "StorageRevision | None"
    workspace: "Workspace | None"
    instructions: RepositoryInstructions
    _asset_reader: "AssetStoreReader | None" = field(repr=False, compare=False)
    models: Mapping[str, ModelBinding | Callable[[], ModelBinding]] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False,
    )
    sandbox: "Sandbox | None" = field(default=None, repr=False, compare=False)
    authorization: AuthorizationPolicy | None = field(default=None, repr=False, compare=False)
    limits: PromptLimits = field(default_factory=PromptLimits)
    defaults: Mapping[str, JsonValue] = field(default_factory=dict)
    filter: Callable[[AgentSpec, CapabilityContribution[object]], bool] | None = field(
        default=None, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("capture group_id must be non-empty")
        if not isinstance(self.limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        _validate_filter(self.filter)
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        object.__setattr__(self, "defaults", _freeze_defaults(self.defaults))
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


class RuntimeFeatures(Generic[AppT]):
    """Register and capture one named set of runtime candidate definitions."""

    def __init__(
        self,
        group_id: str = "default",
        *,
        sandbox: "Sandbox | None" = None,
        authorization: AuthorizationPolicy | None = None,
        limits: PromptLimits | None = None,
        defaults: Mapping[str, JsonValue] | None = None,
        filter: Callable[[AgentSpec, CapabilityContribution[object]], bool] | None = None,
    ) -> None:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("capability group id must be a non-empty string")
        self._defaults = _freeze_defaults({} if defaults is None else defaults)
        _validate_filter(filter)
        self._authorization = authorization
        self._limits = PromptLimits() if limits is None else limits
        if not isinstance(self._limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        self._filter = filter
        self._id = group_id
        self._asset_capability: AssetCapability[AppT] | None = None
        self._tool_responses_registered = False
        self._workspace: Workspace | None = None
        self._sandbox = sandbox
        self._contributions: list[CapabilityContribution[AppT]] = []
        self._models: dict[str, ModelBinding | Callable[[], ModelBinding]] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def defaults(self) -> Mapping[str, JsonValue]:
        """Return frozen, namespaced settings for declaration-time consumers."""
        return self._defaults

    @property
    def authorization(self) -> AuthorizationPolicy | None:
        """Return the current borrowed policy without loading declarations."""
        return self._authorization

    @property
    def workspace(self) -> "Workspace | None":
        return self._workspace

    @property
    def sandbox(self) -> "Sandbox | None":
        return self._sandbox

    def model(self, name: str, binding: ModelBinding | Callable[[], ModelBinding]) -> None:
        """Declare a route; synchronous loaders resolve one binding for each agent run."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model route must be non-empty")
        if inspect.iscoroutinefunction(binding):
            raise TypeError("model loaders must be synchronous")
        self._models[name] = binding

    def tool(
        self,
        function: Callable[..., object],
        *,
        name: "str | None" = None,
        revision: int = 1,
        effect_policy: ToolEffectPolicy = "non_replay_safe",
        plan_safe: bool = False,
    ) -> "Tool[AgentContext[AppT]]":
        """Register one ordinary model-visible Python tool."""
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

    def capability(
        self,
        capability: "AbstractCapability[AgentContext[AppT]]",
        *,
        id: "str | None" = None,
        revision: int = 1,
        config: "Mapping[str, JsonValue] | None" = None,
        scope: Literal["agent", "runtime"] = "runtime",
    ) -> "AbstractCapability[AgentContext[AppT]]":
        """Register Runtime-wide behavior, or explicitly opt into Agent selection.

        Runtime-scoped capabilities bypass Agent selection and are current host
        policy, outside durable Agent bindings. They must not defer loading.
        """
        if not isinstance(capability, AbstractCapability):
            raise TypeError("capability must be an AbstractCapability")
        capability_id = _capability_registration_id(capability, id)
        contribution = CapabilityContribution.from_opaque(
            "capability", capability_id, capability, revision=revision,
            config=config, scope=scope,
        )
        workspace_capabilities: list[WorkspaceCapability] = []
        asset_capabilities: list[AssetCapability[AppT]] = []
        response_capabilities: list[ToolResponsesCapability[AppT]] = []

        def collect(value: AbstractCapability[object]) -> None:
            if isinstance(value, WrapperCapability):
                collect(value.wrapped)
            elif isinstance(value, CombinedCapability):
                for child in value.capabilities:
                    collect(child)
            elif isinstance(value, ToolResponsesCapability):
                response_capabilities.append(value)
            elif isinstance(value, AssetCapability):
                asset_capabilities.append(value)
            elif isinstance(value, WorkspaceCapability):
                workspace_capabilities.append(value)
            else:
                def collect_leaf(leaf: AbstractCapability[object]) -> None:
                    if isinstance(leaf, ToolResponsesCapability):
                        response_capabilities.append(leaf)
                    elif isinstance(leaf, WorkspaceCapability):
                        workspace_capabilities.append(leaf)
                    elif isinstance(leaf, AssetCapability):
                        asset_capabilities.append(leaf)

                value.apply(collect_leaf)

        collect(capability)
        if response_capabilities:
            if scope != "runtime":
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID, "ToolResponsesCapability requires Runtime scope")
            if self._tool_responses_registered or len(response_capabilities) != 1:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        if asset_capabilities:
            if scope != "runtime":
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                              "AssetCapability requires Runtime scope")
            if self._asset_capability is not None or len(asset_capabilities) != 1:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        if workspace_capabilities:
            if scope != "runtime":
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                              "WorkspaceCapability requires Runtime scope")
            if self._workspace is not None or len(workspace_capabilities) != 1:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            workspace_capability = workspace_capabilities[0]
            declarations = {item.name: item for item in workspace_tool_declarations()}
            tools = tuple(
                CapabilityContribution.from_opaque(
                    "tool", tool.name, tool, revision=declarations[tool.name].revision,
                )
                for tool in workspace_capability.tool_definitions()
            )
            self._workspace = workspace_capability.workspace
            self._contributions.extend(tools)
        if asset_capabilities:
            self._asset_capability = asset_capabilities[0]
        self._tool_responses_registered |= bool(response_capabilities)
        self._contributions.append(contribution)
        return capability

    def agent(
        self,
        name: str,
        *,
        revision: int = 1,
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
        tool_retries: int | None = None,
        output_retries: int | None = None,
        description: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> AgentSpec:
        """Register one declarative Agent before Runtime.open()."""
        CapabilityContribution.validate_revision(revision)
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
            metadata={} if metadata is None else metadata,
            revision=revision,
        )
        self._contributions.append(CapabilityContribution.from_declaration(spec))
        return spec

    def mcp(self, server: MCPServerSpec) -> MCPServerSpec:
        """Register an MCP declaration, binding any resources during capture."""
        if not isinstance(server, MCPServerSpec):
            raise TypeError("server must be MCPServerSpec")
        self._contributions.append(CapabilityContribution.from_declaration(server))
        return server

    async def capture(self) -> "RuntimeFeaturesCapture[AppT]":
        """Capture registrations and declarations at one Asset revision."""
        authorization, limits = self._authorization, self._limits
        models = MappingProxyType(dict(self._models))
        contributions = list(tuple(self._contributions))
        instruction_documents: list[RepositoryInstructionDocument] = []
        context: CapabilityLoadContext | None = None
        source_revision: StorageRevision | None = None
        asset_reader: AssetStoreReader | None = None
        if self._asset_capability is not None:
            context, loaded = await self._asset_capability.capture(self._id)
            for value in loaded:
                if isinstance(value, RepositoryInstructionDocument):
                    instruction_documents.append(value)
                else:
                    contributions.append(value)
            source_revision = context.source_revision
            asset_reader = context.asset_reader
        captured_items = capture_contributions(contributions, context=context)
        capture = RuntimeFeaturesCapture(
            self._id,
            captured_items,
            source_revision,
            self._workspace,
            RepositoryInstructions(tuple(instruction_documents)),
            asset_reader,
            models=models,
            sandbox=self._sandbox,
            authorization=authorization,
            limits=limits,
            defaults=self._defaults,
            filter=self._filter,
        )
        _logger.info(
            "capability group captured: group=%s contributions=%d instructions=%d "
            "source_revision=%s",
            self._id,
            len(captured_items),
            len(instruction_documents),
            None if source_revision is None else source_revision.value,
        )
        return capture


def _validate_filter(value: object) -> None:
    if value is not None and not callable(value):
        raise TypeError("filter must be callable or None")


def _freeze_defaults(value: Mapping[str, JsonValue]) -> ImmutableJsonMapping:
    frozen = ImmutableJsonMapping(value)
    if any(not isinstance(section, dict) for section in frozen.values()):
        raise TypeError("defaults namespaces must contain JSON objects")
    return frozen


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
            return await function(
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



def _validate_tool_effect_policy(effect_policy: str, plan_safe: bool) -> None:
    if not is_tool_effect_policy(effect_policy):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(plan_safe, bool):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)



__all__ = ["RuntimeFeatures", "RuntimeFeaturesCapture"]
