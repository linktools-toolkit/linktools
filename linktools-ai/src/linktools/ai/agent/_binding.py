#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime and durable contracts for one exact Agent execution binding."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ..core import ImmutableJsonMapping, JsonValue
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, AgentSpecCodec
from ._output import OutputBinding

if TYPE_CHECKING:
    from ._definition import AgentDefinition

_REQUIRED_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "agent_digest",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
        "global_runtime_capability_descriptors",
    }
)


@dataclass(frozen=True, slots=True)
class AgentBindingSnapshot:
    """Persist the inputs required to reconstruct one exact Agent binding."""

    version: int
    agent_spec: AgentSpec
    agent_digest: str
    output_schema_id: str
    output_schema_revision: int
    output_schema_fingerprint: str
    local_runtime_capability_descriptors: "tuple[Mapping[str, JsonValue], ...]"
    binding_digest: str
    global_runtime_capability_descriptors: "tuple[Mapping[str, JsonValue], ...]"
    output_schema_definition: "Mapping[str, JsonValue] | None" = None

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(self.agent_spec, AgentSpec):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(self.output_schema_id, str) or not self.output_schema_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            not _is_digest(self.agent_digest)
            or not isinstance(self.output_schema_revision, int)
            or isinstance(self.output_schema_revision, bool)
            or self.output_schema_revision < 1
            or not _is_digest(self.output_schema_fingerprint)
            or not _is_digest(self.binding_digest)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            local_descriptors = tuple(
                ImmutableJsonMapping(value)
                for value in self.local_runtime_capability_descriptors
            )
            global_descriptors = tuple(
                ImmutableJsonMapping(value)
                for value in self.global_runtime_capability_descriptors
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if self.output_schema_definition is not None:
            try:
                output_schema_definition = ImmutableJsonMapping(
                    self.output_schema_definition
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            object.__setattr__(
                self,
                "output_schema_definition",
                output_schema_definition,
            )
        object.__setattr__(
            self,
            "local_runtime_capability_descriptors",
            local_descriptors,
        )
        object.__setattr__(
            self,
            "global_runtime_capability_descriptors",
            global_descriptors,
        )

    def to_payload(self) -> "dict[str, JsonValue]":
        payload: dict[str, JsonValue] = {
            "version": 1,
            "agent_spec": AgentSpecCodec().to_payload(self.agent_spec),
            "agent_digest": self.agent_digest,
            "output_schema_id": self.output_schema_id,
            "output_schema_revision": self.output_schema_revision,
            "output_schema_fingerprint": self.output_schema_fingerprint,
            "local_runtime_capability_descriptors": [
                dict(value) for value in self.local_runtime_capability_descriptors
            ],
            "binding_digest": self.binding_digest,
            "global_runtime_capability_descriptors": [
                dict(value) for value in self.global_runtime_capability_descriptors
            ],
        }
        if self.output_schema_definition is not None:
            payload["output_schema_definition"] = dict(self.output_schema_definition)
        return payload

    @classmethod
    def from_payload(cls, value: object) -> "AgentBindingSnapshot":
        if not isinstance(value, Mapping) or not _REQUIRED_FIELDS.issubset(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value["version"]
        revision = value["output_schema_revision"]
        local_descriptors = value["local_runtime_capability_descriptors"]
        global_descriptors = value["global_runtime_capability_descriptors"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(local_descriptors, list) or not isinstance(global_descriptors, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if "output_schema_definition" in value:
            output_schema_definition = _normalize_mapping(value["output_schema_definition"])
        else:
            output_schema_definition = None
        try:
            agent_spec_payload = _require_mapping(value["agent_spec"])
            agent_spec = AgentSpecCodec().from_payload(agent_spec_payload)
            normalized_local = tuple(
                _normalize_mapping(item) for item in local_descriptors
            )
            normalized_global = tuple(
                _normalize_mapping(item) for item in global_descriptors
            )
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls(
            version=1,
            agent_spec=agent_spec,
            agent_digest=_require_digest(value["agent_digest"]),
            output_schema_id=_require_string(value["output_schema_id"]),
            output_schema_revision=revision,
            output_schema_fingerprint=_require_digest(
                value["output_schema_fingerprint"]
            ),
            local_runtime_capability_descriptors=normalized_local,
            binding_digest=_require_digest(value["binding_digest"]),
            global_runtime_capability_descriptors=normalized_global,
            output_schema_definition=output_schema_definition,
        )


@dataclass(frozen=True, slots=True)
class AgentBinding:
    """Bind an AgentDefinition to the exact output contract used by execution."""

    digest: str
    definition: "AgentDefinition"
    output_binding: OutputBinding
    snapshot: AgentBindingSnapshot

    def __post_init__(self) -> None:
        from ._definition import AgentDefinition

        if not isinstance(self.definition, AgentDefinition):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(self.output_binding, OutputBinding):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(self.snapshot, AgentBindingSnapshot):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            self.digest != self.snapshot.binding_digest
            or self.definition.digest != self.snapshot.agent_digest
            or self.definition.spec != self.snapshot.agent_spec
            or self.definition.local_runtime_capability_descriptors
            != self.snapshot.local_runtime_capability_descriptors
            or self.output_binding.schema_id != self.snapshot.output_schema_id
            or self.output_binding.schema_revision
            != self.snapshot.output_schema_revision
            or self.output_binding.schema_fingerprint
            != self.snapshot.output_schema_fingerprint
            or self.definition.global_runtime_capability_descriptors
            != self.snapshot.global_runtime_capability_descriptors
            or (
                self.snapshot.output_schema_definition is not None
                and dict(self.snapshot.output_schema_definition)
                != self.output_binding.schema_definition
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def output_type(self) -> "type[object]":
        return self.output_binding.runtime_output_type

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_binding.schema_fingerprint


def _normalize_mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return dict(ImmutableJsonMapping(cast("Mapping[str, JsonValue]", value)))
    except (AttributeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _require_mapping(value: object) -> "dict[str, object]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _require_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or not _is_digest(value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = ["AgentBinding", "AgentBindingSnapshot"]
