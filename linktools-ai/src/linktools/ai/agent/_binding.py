#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable exact Agent execution binding contract."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ..capability import capability_fingerprint
from ..core import ImmutableJsonMapping, JsonValue
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, AgentSpecCodec
from ._output import OutputBinding, OutputMode

if TYPE_CHECKING:
    from ._definition import AgentDefinition

_PIN_KINDS = frozenset({"tool", "skill", "mcp", "capability"})
_PIN_FIELDS = frozenset({"kind", "id", "contract_version", "contract"})
_REQUIRED_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
        "binding_digest",
    }
)


@dataclass(frozen=True, slots=True)
class SemanticPin:
    kind: Literal["tool", "skill", "mcp", "capability"]
    id: str
    contract_version: int
    contract: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if (
            self.kind not in _PIN_KINDS
            or not isinstance(self.id, str)
            or not self.id.strip()
            or self.contract_version != 1
            or isinstance(self.contract_version, bool)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            contract = ImmutableJsonMapping(self.contract)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "contract", contract)

    @property
    def fingerprint(self) -> str:
        return capability_fingerprint(self.kind, self.id, self.contract)

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "kind": self.kind,
            "id": self.id,
            "contract_version": self.contract_version,
            "contract": dict(self.contract),
        }

    @classmethod
    def from_payload(cls, value: object) -> "SemanticPin":
        if not isinstance(value, Mapping) or set(value) != _PIN_FIELDS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = value["kind"]
        identity = value["id"]
        version = value["contract_version"]
        contract = value["contract"]
        if (
            kind not in _PIN_KINDS
            or not isinstance(identity, str)
            or not isinstance(version, int)
            or isinstance(version, bool)
            or not isinstance(contract, Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls(
            cast(Literal["tool", "skill", "mcp", "capability"], kind),
            identity,
            version,
            _normalize_mapping(contract),
        )


@dataclass(frozen=True, slots=True)
class SubagentRef:
    kind: Literal["agent"]
    id: str

    def __post_init__(self) -> None:
        if self.kind != "agent" or not isinstance(self.id, str) or not self.id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def to_payload(self) -> "dict[str, JsonValue]":
        return {"kind": "agent", "id": self.id}

    @classmethod
    def from_payload(cls, value: object) -> "SubagentRef":
        if not isinstance(value, Mapping) or value.get("kind") != "agent":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = value.get("id")
        if not isinstance(identity, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls("agent", identity)


@dataclass(frozen=True, slots=True)
class AgentBindingSnapshot:
    """Persist every semantic input needed to restore one exact v1 binding."""

    version: int
    agent_spec: AgentSpec
    model: Mapping[str, JsonValue]
    selected: "tuple[SemanticPin, ...]"
    subagents: "tuple[SubagentRef, ...]"
    output_mode: OutputMode
    output_schema: Mapping[str, JsonValue]
    binding_digest: str

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool) or not _is_digest(self.binding_digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(self.agent_spec, AgentSpec) or self.output_mode not in {"text", "structured"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            model = ImmutableJsonMapping(self.model)
            output_schema = ImmutableJsonMapping(self.output_schema)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "output_schema", output_schema)
        selected = tuple(sorted(self.selected, key=lambda item: (item.kind, item.id)))
        if selected != self.selected or len({(item.kind, item.id) for item in selected}) != len(selected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        subagents = tuple(sorted(self.subagents, key=lambda item: item.id))
        if subagents != self.subagents or len({item.id for item in subagents}) != len(subagents):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def subagent_ids(self) -> "tuple[str, ...]":
        return tuple(item.id for item in self.subagents)

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "version": 1,
            "agent_spec": AgentSpecCodec().to_payload(self.agent_spec),
            "model": dict(self.model),
            "selected": [item.to_payload() for item in self.selected],
            "subagents": [item.to_payload() for item in self.subagents],
            "output_mode": self.output_mode,
            "output_schema": dict(self.output_schema),
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "AgentBindingSnapshot":
        if not isinstance(value, Mapping) or not _REQUIRED_FIELDS.issubset(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        selected = value["selected"]
        subagents = value["subagents"]
        mode = value["output_mode"]
        if not isinstance(selected, list) or not isinstance(subagents, list) or mode not in {"text", "structured"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return cls(
                version=1,
                agent_spec=AgentSpecCodec().from_payload(_require_mapping(value["agent_spec"])),
                model=_normalize_mapping(value["model"]),
                selected=tuple(SemanticPin.from_payload(item) for item in selected),
                subagents=tuple(SubagentRef.from_payload(item) for item in subagents),
                output_mode=cast(OutputMode, mode),
                output_schema=_normalize_mapping(value["output_schema"]),
                binding_digest=_require_digest(value["binding_digest"]),
            )
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


@dataclass(frozen=True, slots=True)
class AgentBinding:
    digest: str
    definition: "AgentDefinition"
    output_binding: OutputBinding
    snapshot: AgentBindingSnapshot

    def __post_init__(self) -> None:
        from ._definition import AgentDefinition

        if (
            not isinstance(self.definition, AgentDefinition)
            or not isinstance(self.output_binding, OutputBinding)
            or not isinstance(self.snapshot, AgentBindingSnapshot)
            or self.digest != self.snapshot.binding_digest
            or self.definition.spec != self.snapshot.agent_spec
            or dict(self.definition.model.semantic_payload) != dict(self.snapshot.model)
            or self.definition.selected_subagents != self.snapshot.subagent_ids
            or self.output_binding.mode != self.snapshot.output_mode
            or self.output_binding.schema_definition != dict(self.snapshot.output_schema)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def output_type(self) -> "type[object]":
        return self.output_binding.runtime_output_type

    @property
    def output_fingerprint(self) -> str:
        return self.output_binding.fingerprint


def _normalize_mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return dict(ImmutableJsonMapping(cast("Mapping[str, JsonValue]", value)))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _require_mapping(value: object) -> "dict[str, object]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _require_digest(value: object) -> str:
    if not _is_digest(value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return cast(str, value)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = ["AgentBinding", "AgentBindingSnapshot", "SemanticPin", "SubagentRef"]
