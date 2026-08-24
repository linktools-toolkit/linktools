#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime and durable contracts for one exact Agent execution binding."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from ..core import ImmutableJsonMapping, JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, AgentSpecCodec
from ._output import OutputBinding

if TYPE_CHECKING:
    from ._definition import AgentDefinition

_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "agent_digest",
        "output_type_module",
        "output_type_qualname",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
    }
)


@dataclass(frozen=True, slots=True)
class AgentBindingSnapshot:
    """Persist the inputs required to reconstruct one exact Agent binding."""

    version: int
    agent_spec: AgentSpec
    agent_digest: str
    output_type_module: str
    output_type_qualname: str
    output_schema_id: str
    output_schema_revision: int
    output_schema_fingerprint: str
    local_runtime_capability_descriptors: "tuple[Mapping[str, JsonValue], ...]"
    binding_digest: str

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(self.agent_spec, AgentSpec):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for value in (
            self.output_type_module,
            self.output_type_qualname,
            self.output_schema_id,
        ):
            if not isinstance(value, str) or not value.strip():
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
            descriptors = tuple(
                ImmutableJsonMapping(value)
                for value in self.local_runtime_capability_descriptors
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        object.__setattr__(self, "local_runtime_capability_descriptors", descriptors)

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "version": 1,
            "agent_spec": _agent_spec_payload(self.agent_spec),
            "agent_digest": self.agent_digest,
            "output_type_module": self.output_type_module,
            "output_type_qualname": self.output_type_qualname,
            "output_schema_id": self.output_schema_id,
            "output_schema_revision": self.output_schema_revision,
            "output_schema_fingerprint": self.output_schema_fingerprint,
            "local_runtime_capability_descriptors": [
                dict(value) for value in self.local_runtime_capability_descriptors
            ],
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_payload(cls, value: object) -> "AgentBindingSnapshot":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        version = value["version"]
        revision = value["output_schema_revision"]
        descriptors = value["local_runtime_capability_descriptors"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(descriptors, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            agent_spec = AgentSpecCodec().decode(
                canonical_json_bytes(_require_mapping(value["agent_spec"]))
            )
            normalized = tuple(_normalize_mapping(item) for item in descriptors)
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls(
            version=1,
            agent_spec=agent_spec,
            agent_digest=_require_digest(value["agent_digest"]),
            output_type_module=_require_string(value["output_type_module"]),
            output_type_qualname=_require_string(value["output_type_qualname"]),
            output_schema_id=_require_string(value["output_schema_id"]),
            output_schema_revision=revision,
            output_schema_fingerprint=_require_digest(
                value["output_schema_fingerprint"]
            ),
            local_runtime_capability_descriptors=normalized,
            binding_digest=_require_digest(value["binding_digest"]),
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
            or self.output_type.__module__ != self.snapshot.output_type_module
            or self.output_type.__qualname__ != self.snapshot.output_type_qualname
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @property
    def output_type(self) -> type[BaseModel]:
        return self.output_binding.value_type

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_binding.schema_fingerprint


def _agent_spec_payload(spec: AgentSpec) -> "dict[str, JsonValue]":
    value = json.loads(AgentSpecCodec().encode(spec).decode("utf-8"))
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return cast("dict[str, JsonValue]", value)


def _normalize_mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return dict(ImmutableJsonMapping(cast("Mapping[str, JsonValue]", value)))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _require_mapping(value: object) -> "dict[str, JsonValue]":
    return _normalize_mapping(value)


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
