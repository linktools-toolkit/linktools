#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable data contract for one effective Agent definition."""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, AgentSpecCodec

_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "output_type_module",
        "output_type_qualname",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
    }
)


class _ImmutableJsonMapping(Mapping[str, JsonValue]):
    """Keep one JSON object immutable while exposing Mapping semantics."""

    __slots__ = ("_payload",)

    def __init__(self, value: Mapping[str, JsonValue]) -> None:
        self._payload = canonical_json_bytes(_normalize_mapping(value))

    def __getitem__(self, key: str) -> JsonValue:
        return self._decode()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decode())

    def __len__(self) -> int:
        return len(self._decode())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self._decode() == dict(other)

    def _decode(self) -> "dict[str, JsonValue]":
        value = json.loads(self._payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cast("dict[str, JsonValue]", value)


@dataclass(frozen=True, slots=True)
class AgentBindingSnapshot:
    """Persist the local parts required to reconstruct an effective definition."""

    version: int
    agent_spec: AgentSpec
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
            not isinstance(self.output_schema_revision, int)
            or isinstance(self.output_schema_revision, bool)
            or self.output_schema_revision < 1
            or not _is_digest(self.output_schema_fingerprint)
            or not _is_digest(self.binding_digest)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        descriptors = tuple(
            _ImmutableJsonMapping(value)
            for value in self.local_runtime_capability_descriptors
        )
        object.__setattr__(self, "local_runtime_capability_descriptors", descriptors)

    def to_payload(self) -> "dict[str, JsonValue]":
        return {
            "version": 1,
            "agent_spec": _agent_spec_payload(self.agent_spec),
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
        if version != 1 or isinstance(version, bool):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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


def _agent_spec_payload(spec: AgentSpec) -> "dict[str, JsonValue]":
    value = json.loads(AgentSpecCodec().encode(spec).decode("utf-8"))
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return cast("dict[str, JsonValue]", value)


def _normalize_mapping(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return cast("dict[str, JsonValue]", normalized)


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


__all__ = ["AgentBindingSnapshot"]
