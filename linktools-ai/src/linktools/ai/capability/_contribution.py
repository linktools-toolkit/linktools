#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability contribution contracts and deterministic serialization."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Generic, Literal, TypeAlias, TypeVar, cast

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
from ._context import AgentContext
from ._skill import SkillDefinition
from ._tool_metadata import validate_tool_metadata

AppT = TypeVar("AppT")

ContributionKind = Literal[
    "tool",
    "agent",
    "skill",
    "mcp",
    "capability",
]
ContributionValue: TypeAlias = (
    Tool
    | AgentSpec
    | SkillDefinition
    | MCPServerSpec
    | AbstractCapability
)


@dataclass(frozen=True, slots=True)
class CapabilityContribution(Generic[AppT]):
    kind: ContributionKind
    id: str
    value: (
        "Tool[AgentContext[AppT]] | AgentSpec | SkillDefinition | MCPServerSpec | "
        "AbstractCapability[AgentContext[AppT]]"
    )

    scope: Literal["agent", "runtime"] = field(default="agent", kw_only=True)

    def __post_init__(self) -> None:
        if self.scope not in ("agent", "runtime") or (
            self.scope == "runtime" and self.kind != "capability"
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind not in {
            "tool",
            "agent",
            "skill",
            "mcp",
            "capability",
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
        if self.kind == "tool" and self.value.name != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "agent" and self.value.id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "skill" and self.value.id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "mcp" and self.value.id != self.id:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if self.kind == "capability":
            capability = self.value
            if not isinstance(capability.defer_loading, bool):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability.id is not None and capability.id != self.id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if capability.id is None and capability.defer_loading:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            _validate_external_capability_id(self.id)
            if self.scope == "runtime":
                _validate_runtime_capability(capability)
                capability.apply(_validate_runtime_capability)

    @property
    def revision(self) -> int:
        value = capability_ref_payload(self.kind, self.id, self.contract)["revision"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return value

    @classmethod
    def validate_revision(cls, value: int) -> None:
        """Validate the explicit revision of a named contribution."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    @classmethod
    def from_opaque(
        cls,
        kind: Literal["tool", "capability"],
        identity: str,
        value: "Tool[AgentContext[AppT]] | AbstractCapability[AgentContext[AppT]]",
        *,
        revision: int = 1,
        config: "Mapping[str, JsonValue] | None" = None,
        scope: Literal["agent", "runtime"] = "agent",
    ) -> "CapabilityContribution[AppT]":
        if kind not in {"tool", "capability"}:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        cls.validate_revision(revision)
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
            scope=scope,
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
        value: MCPServerSpec,
    ) -> "CapabilityContribution[object]":
        if not isinstance(contract, Mapping) or not isinstance(value, MCPServerSpec):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        MCPServerSpecCodec().decode_binding_payload(contract, declaration=value)
        return _ContractContribution("mcp", value.id, value, contract)

    @property
    def contract(self) -> "dict[str, JsonValue]":
        return _contribution_contract(self.kind, self.id, self.value)

    def materialize_tool(self) -> "Tool[AgentContext[AppT]]":
        """Create an execution-local tool from this contribution's contract."""
        return _freeze_contribution(self).materialize_tool()


@dataclass(frozen=True, slots=True)
class _ContractContribution(CapabilityContribution[AppT]):
    _contract: Mapping[str, JsonValue] = field(repr=False, compare=False)
    _tool_materializer: (
        "Callable[[str, Mapping[str, JsonValue]], Tool[AgentContext[AppT]]] | None"
    ) = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            contract = ImmutableJsonMapping(self._contract)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        object.__setattr__(self, "_contract", contract)
        CapabilityContribution.__post_init__(self)
        if self.kind == "tool":
            object.__setattr__(
                self, "_tool_materializer", _capture_tool_materializer(self.value),
            )

    @property
    def contract(self) -> "dict[str, JsonValue]":
        return dict(self._contract)

    def materialize_tool(self) -> "Tool[AgentContext[AppT]]":
        if self._tool_materializer is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return self._tool_materializer(self.id, self.contract)


def _capture_tool_materializer(
    tool: "Tool[AgentContext[AppT]]",
) -> "Callable[[str, Mapping[str, JsonValue]], Tool[AgentContext[AppT]]]":
    # Retain implementation references, not the authoring Tool or its semantics.
    schema = replace(
        tool.function_schema,
        name="",
        description=None,
        json_schema={},
        return_schema={},
        positional_fields=list(tool.function_schema.positional_fields),
    )
    function = tool.function
    prepare = tool.prepare
    args_validator = tool.args_validator
    docstring_format = tool.docstring_format
    require_parameter_descriptions = tool.require_parameter_descriptions

    def materialize(
        name: str,
        contract: Mapping[str, JsonValue],
    ) -> "Tool[AgentContext[AppT]]":
        description = cast("str | None", contract["description"])
        function_schema = replace(
            schema,
            name=name,
            description=description,
            json_schema=cast("dict[str, JsonValue]", contract["parameters"]),
            return_schema=cast("dict[str, JsonValue]", contract["return_schema"]),
            positional_fields=list(schema.positional_fields),
        )
        return Tool(
            function,
            name=name,
            description=description,
            max_retries=cast("int | None", contract.get("max_retries")),
            prepare=prepare,
            args_validator=args_validator,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            strict=cast("bool | None", contract["strict"]),
            sequential=cast(bool, contract.get("sequential", False)),
            requires_approval=contract.get("kind", "function") == "unapproved",
            metadata=cast("dict[str, JsonValue]", contract["metadata"]),
            timeout=cast("float | None", contract.get("timeout")),
            defer_loading=cast(bool, contract.get("defer_loading", False)),
            include_return_schema=cast("bool | None", contract.get("include_return_schema")),
            function_schema=function_schema,
        )

    return materialize


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
        scope=value.scope,
    )


def _validate_runtime_capability(capability: AbstractCapability[object]) -> None:
    if not isinstance(capability.defer_loading, bool) or capability.defer_loading:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if capability.id is not None:
        if not isinstance(capability.id, str) or not capability.id.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        _validate_external_capability_id(capability.id)


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
        return AgentSpecCodec().to_contract_payload(value)
    if kind == "skill" and isinstance(value, SkillDefinition):
        return value.contract
    if kind == "mcp" and isinstance(value, MCPServerSpec):
        return MCPServerSpecCodec().to_contract_payload(value)
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
    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _validate_external_capability_id(value: str) -> None:
    if value.startswith("linktools."):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)



__all__ = ["CapabilityContribution"]
