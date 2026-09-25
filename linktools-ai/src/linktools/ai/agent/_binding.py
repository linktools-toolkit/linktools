#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable exact Agent execution binding contract."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from ..core import ImmutableJsonMapping, JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import (
    AgentSpec,
    AgentSpecCodec,
    SubagentRef,
    binding_digest_payload,
    capability_ref_payload,
)
from ._output import OutputBinding, OutputMode

if TYPE_CHECKING:
    from ._compiled import CompiledAgent

_PIN_KINDS = frozenset({"tool", "skill", "mcp", "capability"})
_PIN_FIELDS = frozenset({"kind", "id", "contract"})
_BINDING_VERSION = 1
_BINDING_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "model_contract",
        "selected",
        "subagents",
        "output_mode",
        "output_schema",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityPin:
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
        capability_ref_payload(self.kind, self.id, contract)

    @property
    def revision(self) -> int:
        value = capability_ref_payload(self.kind, self.id, self.contract)["revision"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "kind": self.kind,
            "id": self.id,
            "contract": dict(self.contract),
        }

    @classmethod
    def from_payload(cls, value: object) -> "CapabilityPin":
        if not isinstance(value, Mapping) or not _PIN_FIELDS.issubset(value):
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
    """Persist identity inputs and locators required to restore one Agent binding."""

    agent_spec: AgentSpec
    model_contract: Mapping[str, JsonValue]
    selected: "tuple[CapabilityPin, ...]"
    subagents: "tuple[SubagentRef, ...]"
    output_mode: OutputMode
    output_schema: Mapping[str, JsonValue]
    subagent_bindings: "tuple[AgentBindingSnapshot, ...]" = ()
    _wire_extensions: Mapping[str, JsonValue] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _agent_spec_extensions: Mapping[str, JsonValue] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _binding_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_spec, AgentSpec) or self.output_mode not in {"text", "structured"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            model_contract = ImmutableJsonMapping(self.model_contract)
            output_schema = ImmutableJsonMapping(self.output_schema)
            wire_extensions = ImmutableJsonMapping(self._wire_extensions)
            agent_spec_extensions = ImmutableJsonMapping(
                self._agent_spec_extensions
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "model_contract", model_contract)
        object.__setattr__(self, "output_schema", output_schema)
        object.__setattr__(self, "_wire_extensions", wire_extensions)
        object.__setattr__(
            self,
            "_agent_spec_extensions",
            agent_spec_extensions,
        )
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
        child_bindings = tuple(
            sorted(
                self.subagent_bindings,
                key=lambda item: item.agent_spec.id,
            )
        )
        if (
            child_bindings != self.subagent_bindings
            or any(
                not isinstance(item, AgentBindingSnapshot)
                or item.subagents
                or item.subagent_bindings
                for item in child_bindings
            )
            or len({item.agent_spec.id for item in child_bindings})
            != len(child_bindings)
            or (
                child_bindings
                and tuple(item.agent_spec.id for item in child_bindings)
                != self.subagent_ids
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object.__setattr__(
            self,
            "_binding_digest",
            canonical_sha256(binding_digest_payload(self.to_payload())),
        )

    @property
    def subagent_ids(self) -> "tuple[str, ...]":
        return tuple(item.id for item in self.subagents)

    @property
    def subagent_binding_map(
        self,
    ) -> "Mapping[str, AgentBindingSnapshot]":
        return {
            item.agent_spec.id: item
            for item in self.subagent_bindings
        }

    @property
    def binding_digest(self) -> str:
        return self._binding_digest

    def to_payload(self) -> "dict[str, JsonValue]":
        agent_spec = AgentSpecCodec().to_wire_payload(self.agent_spec)
        for key, value in self._agent_spec_extensions.items():
            agent_spec.setdefault(key, value)
        payload: dict[str, JsonValue] = {
            "version": _BINDING_VERSION,
            "agent_spec": agent_spec,
            "model_contract": dict(self.model_contract),
            "selected": [item.to_payload() for item in self.selected],
            "subagents": [item.to_payload() for item in self.subagents],
            "output_mode": self.output_mode,
            "output_schema": dict(self.output_schema),
        }
        if self.subagent_bindings:
            payload["subagent_bindings"] = [
                item.to_payload()
                for item in self.subagent_bindings
            ]
        for key, value in self._wire_extensions.items():
            payload.setdefault(key, value)
        return payload

    @classmethod
    def from_payload(cls, value: object) -> "AgentBindingSnapshot":
        if not isinstance(value, Mapping) or not _BINDING_FIELDS.issubset(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != _BINDING_VERSION:
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
            agent_spec_payload = _require_mapping(value["agent_spec"])
            agent_spec_codec = AgentSpecCodec()
            agent_spec = agent_spec_codec.from_payload(agent_spec_payload)
            canonical_agent_spec = agent_spec_codec.to_wire_payload(agent_spec)
            agent_spec_extensions = {
                key: item
                for key, item in agent_spec_payload.items()
                if key not in canonical_agent_spec
            }
            subagent_bindings = (
                ()
                if "subagent_bindings" not in value
                else _decode_subagent_bindings(value["subagent_bindings"])
            )
            wire_extensions = {
                key: item
                for key, item in value.items()
                if key not in _BINDING_FIELDS
                and key != "subagent_bindings"
            }
            return cls(
                agent_spec=agent_spec,
                model_contract=_normalize_mapping(value["model_contract"]),
                selected=tuple(CapabilityPin.from_payload(item) for item in selected),
                subagents=tuple(SubagentRef.from_payload(item) for item in subagents),
                output_mode=cast(OutputMode, mode),
                output_schema=_normalize_mapping(value["output_schema"]),
                subagent_bindings=subagent_bindings,
                _wire_extensions=wire_extensions,
                _agent_spec_extensions=agent_spec_extensions,
            )
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


@dataclass(frozen=True, slots=True)
class AgentBinding:
    compiled_agent: "CompiledAgent"
    output_binding: OutputBinding
    snapshot: AgentBindingSnapshot

    def __post_init__(self) -> None:
        from ._compiled import CompiledAgent

        if (
            not isinstance(self.compiled_agent, CompiledAgent)
            or not isinstance(self.output_binding, OutputBinding)
            or not isinstance(self.snapshot, AgentBindingSnapshot)
            or AgentSpecCodec().to_payload(self.compiled_agent.spec)
            != AgentSpecCodec().to_payload(self.snapshot.agent_spec)
            or dict(self.compiled_agent.model.contract)
            != dict(self.snapshot.model_contract)
            or _compiled_agent_selected_pins(self.compiled_agent) != self.snapshot.selected
            or self.compiled_agent.selected_subagents != self.snapshot.subagent_ids
            or self.output_binding.mode != self.snapshot.output_mode
            or self.output_binding.schema_definition != dict(self.snapshot.output_schema)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def binding_digest(self) -> str:
        return self.snapshot.binding_digest

    @property
    def output_type(self) -> "type[object]":
        return self.output_binding.runtime_output_type


def _compiled_agent_selected_pins(
    compiled_agent: "CompiledAgent",
) -> "tuple[CapabilityPin, ...]":
    candidates = (
        *sorted(
            (
                *compiled_agent.selected_tools,
                *compiled_agent.selected_skills,
                *compiled_agent.selected_mcp,
            ),
            key=lambda item: (item.kind, item.id),
        ),
        *compiled_agent.selected_capabilities,
    )
    return tuple(
        CapabilityPin(
            cast(Literal["tool", "skill", "mcp", "capability"], candidate.kind),
            candidate.id,
            candidate.contract,
        )
        for candidate in candidates
    )


def _decode_subagent_bindings(
    value: object,
) -> "tuple[AgentBindingSnapshot, ...]":
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return tuple(
        AgentBindingSnapshot.from_payload(item)
        for item in value
    )


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


__all__ = ["AgentBinding", "AgentBindingSnapshot", "CapabilityPin", "SubagentRef"]
