#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability contribution contracts and deterministic serialization."""

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability

from ..core import ImmutableJsonMapping, JsonValue
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentSpec,
    AgentSpecCodec,
    MCPServerSpec,
    MCPServerSpecCodec,
    canonicalize_pydantic_model_schema,
    capability_ref_payload,
)
from ..task import (
    TaskEffectResolution,
    TaskExpanderRef,
    TaskNodeContext,
    TaskNodeHandler,
)
from ._context import AgentContext
from ._skill import SkillDefinition
from ._task import TaskExpander
from ._tool_metadata import validate_tool_metadata

AppT = TypeVar("AppT")

ContributionKind = Literal[
    "tool",
    "agent",
    "skill",
    "mcp",
    "runtime_capability",
    "task",
    "task_expander",
]
ContributionValue: TypeAlias = (
    Tool
    | AgentSpec
    | SkillDefinition
    | MCPServerSpec
    | AbstractCapability
    | TaskNodeHandler[object]
    | TaskExpander
)
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RESERVED_TASK_ID_PREFIX = "linktools.ai."
_RESERVED_EXPANDER_ID_PREFIX = "linktools.ai."


@dataclass(frozen=True, slots=True)
class _TaskHandlerAdapter(Generic[AppT]):
    handler: TaskNodeHandler[AppT]
    effect_policy: Literal["none", "replay_safe", "non_replay_safe"]
    output_type: "type[BaseModel] | None"
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
class CapabilityContribution(Generic[AppT]):
    kind: ContributionKind
    id: str
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
            "runtime_capability",
            "task",
            "task_expander",
        } or not isinstance(self.id, str) or not self.id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "tool" and not isinstance(self.value, Tool):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "agent" and not isinstance(self.value, AgentSpec):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "skill" and not isinstance(self.value, SkillDefinition):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "mcp" and not isinstance(self.value, MCPServerSpec):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "runtime_capability" and not isinstance(self.value, AbstractCapability):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "task" and not isinstance(self.value, TaskNodeHandler):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "task_expander" and not isinstance(self.value, TaskExpander):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "tool" and self.value.name != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "agent" and self.value.id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "skill" and self.value.id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "mcp" and self.value.id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "runtime_capability":
            capability = self.value
            if not isinstance(capability.defer_loading, bool):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability.id is not None and capability.id != self.id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability.id is None and capability.defer_loading:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            _validate_external_capability_id(self.id)
        if self.kind == "task":
            identity, _revision = _task_identity(self.value)
            if self.id != identity:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "task_expander":
            identity, _revision = _expander_identity(self.value)
            if self.id != identity:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    @property
    def revision(self) -> int:
        value = capability_ref_payload(self.kind, self.id, self.contract)["revision"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return value

    @classmethod
    def from_opaque(
        cls,
        kind: Literal["tool", "runtime_capability"],
        identity: str,
        value: "Tool[AgentContext[AppT]] | AbstractCapability[AgentContext[AppT]]",
        *,
        revision: int = 1,
        config: "Mapping[str, JsonValue] | None" = None,
    ) -> "CapabilityContribution[AppT]":
        if kind not in {"tool", "runtime_capability"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        _validate_revision(revision)
        return _ContractContribution(
            kind,
            identity,
            value,
            _contribution_contract(
                kind,
                identity,
                value,
                revision=revision,
                config=config,
            ),
        )

    @classmethod
    def from_declaration(
        cls,
        value: AgentSpec | SkillDefinition | MCPServerSpec,
    ) -> "CapabilityContribution[object]":
        if isinstance(value, AgentSpec):
            kind: Literal["agent", "skill", "mcp"] = "agent"
        elif isinstance(value, SkillDefinition):
            kind = "skill"
        elif isinstance(value, MCPServerSpec):
            kind = "mcp"
        else:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return _ContractContribution(
            kind,
            value.id,
            value,
            _contribution_contract(kind, value.id, value),
        )

    @classmethod
    def from_mcp_contract(
        cls,
        contract: Mapping[str, JsonValue],
    ) -> "CapabilityContribution[object]":
        if not isinstance(contract, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value, _resource_versions = MCPServerSpecCodec().from_execution_payload(
            contract
        )
        return _ContractContribution("mcp", value.id, value, contract)

    @classmethod
    def from_task(
        cls,
        value: "TaskNodeHandler[AppT]",
        *,
        effect_policy: Literal["none", "replay_safe", "non_replay_safe"] = "non_replay_safe",
        output_type: "type[BaseModel] | None" = None,
        reconcile: (
            "Callable[[TaskNodeContext[AppT]], Awaitable[TaskEffectResolution]] | None"
        ) = None,
    ) -> "CapabilityContribution[AppT]":
        if not isinstance(value, TaskNodeHandler):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if effect_policy not in {"none", "replay_safe", "non_replay_safe"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if reconcile is not None and not callable(reconcile):
            raise TypeError("reconcile must be callable")
        adapted = _TaskHandlerAdapter(
            value,
            effect_policy,
            output_type,
            reconcile,
        )
        identity, _revision = _task_identity(adapted)
        return _ContractContribution(
            "task",
            identity,
            adapted,
            _contribution_contract("task", identity, adapted),
        )

    @classmethod
    def from_task_expander(
        cls,
        value: TaskExpander,
    ) -> "CapabilityContribution[object]":
        identity, _revision = _expander_identity(value)
        return _ContractContribution(
            "task_expander",
            identity,
            value,
            _contribution_contract("task_expander", identity, value),
        )

    @property
    def contract(self) -> "dict[str, JsonValue]":
        return _contribution_contract(self.kind, self.id, self.value)


@dataclass(frozen=True, slots=True)
class _ContractContribution(CapabilityContribution[AppT]):
    _contract: Mapping[str, JsonValue] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            contract = ImmutableJsonMapping(self._contract)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        object.__setattr__(self, "_contract", contract)
        CapabilityContribution.__post_init__(self)

    @property
    def contract(self) -> "dict[str, JsonValue]":
        return dict(self._contract)


def _freeze_contribution(
    value: CapabilityContribution[AppT],
) -> CapabilityContribution[AppT]:
    if isinstance(value, _ContractContribution):
        return value
    return _ContractContribution(
        value.kind,
        value.id,
        value.value,
        value.contract,
    )


def _contribution_contract(
    kind: ContributionKind,
    identity: str,
    value: ContributionValue,
    *,
    revision: "int | None" = None,
    config: "Mapping[str, JsonValue] | None" = None,
) -> "dict[str, JsonValue]":
    if config is not None and kind != "runtime_capability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if kind == "tool" and isinstance(value, Tool):
        definition = value.tool_def
        validate_tool_metadata(
            definition.metadata,
            require_effect_policy=True,
            require_tool_class=True,
        )
        contract: dict[str, JsonValue] = {
            "version": 1,
            "description": definition.description,
            "parameters": definition.parameters_json_schema,
            "return_schema": definition.return_schema,
            "strict": definition.strict,
            "metadata": definition.metadata,
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
        contract["revision"] = revision or 1
        return contract
    if kind == "agent" and isinstance(value, AgentSpec):
        return AgentSpecCodec().to_payload(value)
    if kind == "skill" and isinstance(value, SkillDefinition):
        return value.contract
    if kind == "mcp" and isinstance(value, MCPServerSpec):
        return MCPServerSpecCodec().to_payload(value)
    if kind == "runtime_capability" and isinstance(value, AbstractCapability):
        contract = {
            "version": 1,
            "revision": revision or 1,
            "defer_loading": value.defer_loading,
            "config": {},
        }
        if config is not None:
            try:
                contract["config"] = dict(ImmutableJsonMapping(config))
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        return contract
    if kind == "task" and isinstance(value, TaskNodeHandler):
        task_id, task_revision = _task_identity(value)
        if identity != task_id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return {
            "version": 1,
            "id": task_id,
            "revision": task_revision,
            "effect_policy": _task_effect_policy(value),
            "output_contract": _task_output_contract(value),
            "reconcile": getattr(value, "reconcile", None) is not None,
        }
    if kind == "task_expander" and isinstance(value, TaskExpander):
        expander_id, expander_revision = _expander_identity(value)
        if identity != expander_id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return {
            "version": 1,
            "id": expander_id,
            "revision": expander_revision,
        }
    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _task_identity(handler: object) -> tuple[str, int]:
    if not isinstance(handler, TaskNodeHandler):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    task_id = handler.id
    task_revision = handler.revision
    if (
        not isinstance(task_id, str)
        or _TASK_ID.fullmatch(task_id) is None
        or task_id.startswith(_RESERVED_TASK_ID_PREFIX)
        or isinstance(task_revision, bool)
        or not isinstance(task_revision, int)
        or task_revision < 1
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return task_id, task_revision


def _task_effect_policy(handler: object) -> str:
    effect_policy = getattr(handler, "effect_policy", None)
    if effect_policy not in {"none", "replay_safe", "non_replay_safe"}:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return effect_policy


def _task_output_contract(handler: object) -> JsonValue:
    output_type = getattr(handler, "output_type", None)
    if output_type is None:
        return {"kind": "json"}
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return {
        "kind": "schema",
        "schema": canonicalize_pydantic_model_schema(output_type),
    }


def _expander_identity(expander: object) -> tuple[str, int]:
    if not isinstance(expander, TaskExpander):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    try:
        reference = TaskExpanderRef(expander.id, expander.revision)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if reference.id.startswith(_RESERVED_EXPANDER_ID_PREFIX):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return reference.id, reference.revision


def _validate_external_capability_id(value: str) -> None:
    if value.startswith("linktools."):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_revision(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


__all__ = ["CapabilityContribution"]
