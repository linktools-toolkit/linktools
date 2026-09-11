#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable exact Agent execution binding contract."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ..core import ImmutableJsonMapping, JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, AgentSpecCodec, SubagentRef
from ._output import OutputBinding, OutputMode

if TYPE_CHECKING:
    from ._definition import AgentDefinition

_PIN_KINDS = frozenset({"tool", "skill", "mcp", "capability"})
_PIN_FIELDS = frozenset({"kind", "id", "contract"})
_BINDING_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "base_model",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
)


@dataclass(frozen=True, slots=True)
class SemanticPin:
    kind: Literal["tool", "skill", "mcp", "capability"]
    id: str
    contract: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if (
            self.kind not in _PIN_KINDS
            or not isinstance(self.id, str)
            or not self.id.strip()
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            contract = ImmutableJsonMapping(self.contract)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        version = contract.get("version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        object.__setattr__(self, "contract", contract)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "contract": "capability-fingerprint-v1",
                "kind": self.kind,
                "id": self.id,
                "semantic": dict(self.contract),
            }
        )

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "kind": self.kind,
            "id": self.id,
            "contract": dict(self.contract),
        }

    @classmethod
    def from_payload(cls, value: object) -> "SemanticPin":
        if not isinstance(value, Mapping) or set(value) != _PIN_FIELDS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = value["kind"]
        identity = value["id"]
        contract = value["contract"]
        if (
            kind not in _PIN_KINDS
            or not isinstance(identity, str)
            or not isinstance(contract, Mapping)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls(
            cast(Literal["tool", "skill", "mcp", "capability"], kind),
            identity,
            _normalize_mapping(contract),
        )


@dataclass(frozen=True, slots=True)
class AgentBindingSnapshot:
    """Persist the semantic inputs required to restore one Agent binding."""

    version: int
    agent_spec: AgentSpec
    base_model: Mapping[str, JsonValue]
    selected: "tuple[SemanticPin, ...]"
    subagents: "tuple[SubagentRef, ...]"
    output_mode: OutputMode
    output_schema: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self.version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if not isinstance(self.agent_spec, AgentSpec) or self.output_mode not in {"text", "structured"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            base_model = ImmutableJsonMapping(self.base_model)
            output_schema = ImmutableJsonMapping(self.output_schema)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "output_schema", output_schema)
        selected = tuple(
            (
                *sorted(
                    (item for item in self.selected if item.kind != "capability"),
                    key=lambda item: (item.kind, item.id),
                ),
                *(item for item in self.selected if item.kind == "capability"),
            )
        )
        if selected != self.selected or len({(item.kind, item.id) for item in selected}) != len(selected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        subagents = tuple(sorted(self.subagents, key=lambda item: item.id))
        if subagents != self.subagents or len({item.id for item in subagents}) != len(subagents):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def subagent_ids(self) -> "tuple[str, ...]":
        return tuple(item.id for item in self.subagents)

    @property
    def binding_digest(self) -> str:
        return canonical_sha256(
            {
                "contract": "agent-binding-v1",
                "snapshot": self.to_payload(),
            }
        )

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "version": self.version,
            "agent_spec": AgentSpecCodec().to_wire_payload(self.agent_spec),
            "base_model": dict(self.base_model),
            "selected": [item.to_payload() for item in self.selected],
            "subagents": [item.to_payload() for item in self.subagents],
            "output_mode": self.output_mode,
            "output_schema": dict(self.output_schema),
        }

    @classmethod
    def from_payload(cls, value: object) -> "AgentBindingSnapshot":
        if not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        selected = value["selected"]
        subagents = value["subagents"]
        mode = value["output_mode"]
        if (
            not isinstance(selected, list)
            or not isinstance(subagents, list)
            or mode not in {"text", "structured"}
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return cls(
                version=version,
                agent_spec=AgentSpecCodec().from_payload(_require_mapping(value["agent_spec"])),
                base_model=_normalize_mapping(value["base_model"]),
                selected=tuple(SemanticPin.from_payload(item) for item in selected),
                subagents=tuple(SubagentRef.from_payload(item) for item in subagents),
                output_mode=cast(OutputMode, mode),
                output_schema=_normalize_mapping(value["output_schema"]),
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
            or AgentSpecCodec().to_payload(self.definition.spec)
            != AgentSpecCodec().to_payload(self.snapshot.agent_spec)
            or dict(self.definition.model.semantic_payload)
            != dict(self.snapshot.base_model)
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


__all__ = ["AgentBinding", "AgentBindingSnapshot", "SemanticPin", "SubagentRef"]
