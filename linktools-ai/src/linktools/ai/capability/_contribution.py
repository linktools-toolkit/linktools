#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability contribution contracts and deterministic serialization."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar, cast

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
    canonicalize_json_schema,
    canonicalize_pydantic_model_schema,
    capability_ref_payload,
)
from ..task import TaskExpanderRef, TaskNodeHandler
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
    "capability",
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
            "capability",
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
        kind: Literal["tool", "capability"],
        identity: str,
        value: "Tool[AgentContext[AppT]] | AbstractCapability[AgentContext[AppT]]",
        *,
        revision: int = 1,
        config: "Mapping[str, JsonValue] | None" = None,
    ) -> "CapabilityContribution[AppT]":
        if kind not in {"tool", "capability"}:
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
            cast("Mapping[str, object]", contract)
        )
        return _ContractContribution("mcp", value.id, value, contract)

    @classmethod
    def from_task(
        cls,
        value: "TaskNodeHandler[AppT]",
    ) -> "CapabilityContribution[AppT]":
        identity, _revision = _task_identity(value)
        return _ContractContribution(
            "task",
            identity,
            value,
            _contribution_contract("task", identity, value),
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
    if config is not None and kind != "capability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if kind == "tool" and isinstance(value, Tool):
        definition = value.tool_def
        validate_tool_metadata(
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
        contract["revision"] = revision or 1
        return contract
    if kind == "agent" and isinstance(value, AgentSpec):
        return AgentSpecCodec().to_payload(value)
    if kind == "skill" and isinstance(value, SkillDefinition):
        return value.contract
    if kind == "mcp" and isinstance(value, MCPServerSpec):
        return MCPServerSpecCodec().to_payload(value)
    if kind == "capability" and isinstance(value, AbstractCapability):
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
            "effect": _task_effect(value),
            "output": _task_output_contract(value),
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
    return {"kind": "schema", "schema": schema}


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
