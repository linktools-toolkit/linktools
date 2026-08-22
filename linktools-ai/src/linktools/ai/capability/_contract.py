#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability binding and runtime materialization contracts."""

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability

from ..asset import AssetRef, AssetRepository
from ..core import (
    JsonValue,
    Principal,
    ResourceRef,
    canonical_json_bytes,
    canonical_sha256,
    canonical_string_tuple,
)
from ..errors import AIError, ErrorCode


@dataclass(frozen=True, slots=True)
class CapabilityRefResolution:
    ref: AssetRef
    resolved_revision: int
    fingerprint: str

    def __post_init__(self) -> None:
        if self.resolved_revision < 1 or not _is_fingerprint(self.fingerprint):
            raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


@dataclass(frozen=True, slots=True)
class CapabilityMaterializationContext:
    principal: Principal
    execution: ResourceRef
    execution_root: Path
    allow_tools: "tuple[str, ...]" = ("*",)

    def __post_init__(self) -> None:
        if self.principal.tenant_id != self.execution.tenant_id:
            raise ValueError("capability context tenant mismatch")
        if not isinstance(self.execution_root, Path):
            raise TypeError("execution_root must be a Path")
        object.__setattr__(
            self,
            "execution_root",
            self.execution_root.expanduser().resolve(strict=False),
        )
        object.__setattr__(
            self,
            "allow_tools",
            canonical_string_tuple(self.allow_tools, field="allow_tools"),
        )


class CapabilityBinding(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def resolutions(self) -> "tuple[CapabilityRefResolution, ...]": ...

    @property
    def fingerprint(self) -> str: ...

    async def materialize(
        self,
        context: CapabilityMaterializationContext,
    ) -> "tuple[PydanticAgentCapability[None], ...]": ...


class CapabilityProvider(Protocol):
    """Bind one Asset value type into one frozen Runtime capability."""

    @property
    def provider(self) -> str: ...

    @property
    def value_type(self) -> type[object]: ...

    @property
    def revision(self) -> int: ...

    async def bind(
        self,
        refs: "tuple[AssetRef, ...]",
        *,
        assets: AssetRepository,
    ) -> CapabilityBinding: ...


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    """Stable Python capability binding supplied by Runtime composition."""

    id: str
    capability: "PydanticAgentCapability[None]"
    revision: int = 1
    _descriptor_payload: "bytes | None" = field(default=None, repr=False, compare=True)

    @property
    def provider(self) -> str:
        return "runtime"

    @property
    def resolutions(self) -> "tuple[CapabilityRefResolution, ...]":
        return ()

    @property
    def durable(self) -> bool:
        return self._descriptor_payload is not None

    @property
    def descriptor(self) -> "dict[str, JsonValue] | None":
        payload = self._descriptor_payload
        if payload is None:
            return None
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return cast("dict[str, JsonValue]", value)

    @property
    def fingerprint(self) -> str:
        descriptor = self.descriptor
        if descriptor is not None:
            value = descriptor.get("fingerprint")
            if not isinstance(value, str) or not _is_fingerprint(value):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            return value
        capability_type = type(self.capability)
        return canonical_sha256(
            {
                "id": self.id,
                "revision": self.revision,
                "module": capability_type.__module__,
                "qualname": capability_type.__qualname__,
            }
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or self.capability is None
            or not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("runtime capability is incomplete")
        if self._descriptor_payload is not None:
            descriptor = self.descriptor
            if descriptor is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            _validate_runtime_capability_descriptor(descriptor)

    @classmethod
    def from_spec(
        cls,
        id: str,
        capability_type: "type[AbstractCapability[None]]",
        *,
        config: "Mapping[str, JsonValue]",
        revision: int = 1,
    ) -> "RuntimeCapability":
        if (
            not isinstance(capability_type, type)
            or not issubclass(capability_type, AbstractCapability)
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        _validate_importable_capability_type(capability_type)
        serialization_name = capability_type.get_serialization_name()
        if not isinstance(serialization_name, str) or not serialization_name.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        normalized = _normalize_json_mapping(config)
        try:
            capability = capability_type.from_spec(**normalized)
        except Exception as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if type(capability) is not capability_type:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        fingerprint = _runtime_descriptor_fingerprint(
            id=id,
            revision=revision,
            module=capability_type.__module__,
            qualname=capability_type.__qualname__,
            serialization_name=serialization_name,
            config=normalized,
        )
        descriptor: dict[str, JsonValue] = {
            "id": id,
            "revision": revision,
            "capability_type": {
                "module": capability_type.__module__,
                "qualname": capability_type.__qualname__,
            },
            "serialization_name": serialization_name,
            "config": normalized,
            "fingerprint": fingerprint,
        }
        _validate_runtime_capability_descriptor(descriptor)
        return cls(
            id=id,
            capability=capability,
            revision=revision,
            _descriptor_payload=canonical_json_bytes(descriptor),
        )

    @classmethod
    def restore(
        cls,
        descriptor: "Mapping[str, JsonValue]",
    ) -> "RuntimeCapability":
        try:
            value = _normalize_json_mapping(descriptor)
            _validate_runtime_capability_descriptor(value)
        except AIError as error:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        type_value = cast("dict[str, JsonValue]", value["capability_type"])
        module_name = cast(str, type_value["module"])
        qualname = cast(str, type_value["qualname"])
        try:
            target = _resolve_capability_type(module_name, qualname)
        except AIError as error:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        serialization_name = target.get_serialization_name()
        if serialization_name != value["serialization_name"]:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        config = value["config"]
        if not isinstance(config, dict):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        expected = _runtime_descriptor_fingerprint(
            id=cast(str, value["id"]),
            revision=cast(int, value["revision"]),
            module=module_name,
            qualname=qualname,
            serialization_name=cast(str, value["serialization_name"]),
            config=cast("Mapping[str, JsonValue]", config),
        )
        if expected != value["fingerprint"]:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        try:
            capability = target.from_spec(**config)
        except Exception as error:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        if type(capability) is not target:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        restored = cls(
            id=cast(str, value["id"]),
            capability=capability,
            revision=cast(int, value["revision"]),
            _descriptor_payload=canonical_json_bytes(value),
        )
        if restored.fingerprint != expected:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return restored

    async def materialize(
        self,
        context: CapabilityMaterializationContext,
    ) -> "tuple[PydanticAgentCapability[None], ...]":
        del context
        return (self.capability,)


def validate_fingerprint(value: str) -> None:
    if not _is_fingerprint(value):
        raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


def _validate_importable_capability_type(
    capability_type: "type[AbstractCapability[None]]",
) -> None:
    try:
        resolved = _resolve_capability_type(
            capability_type.__module__,
            capability_type.__qualname__,
        )
    except AIError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if resolved is not capability_type:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _resolve_capability_type(
    module_name: str,
    qualname: str,
) -> "type[AbstractCapability[None]]":
    if module_name == "__main__" or "<locals>" in qualname:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    try:
        target: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (AttributeError, ImportError, ModuleNotFoundError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if not isinstance(target, type) or not issubclass(target, AbstractCapability):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return target


def _runtime_descriptor_fingerprint(
    *,
    id: str,
    revision: int,
    module: str,
    qualname: str,
    serialization_name: str,
    config: "Mapping[str, JsonValue]",
) -> str:
    return canonical_sha256(
        {
            "id": id,
            "revision": revision,
            "module": module,
            "qualname": qualname,
            "serialization_name": serialization_name,
            "config": dict(config),
        }
    )


def _normalize_json_mapping(
    value: Mapping[str, JsonValue],
) -> "dict[str, JsonValue]":
    try:
        decoded = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if not isinstance(decoded, dict):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return cast("dict[str, JsonValue]", decoded)


def _validate_runtime_capability_descriptor(
    value: Mapping[str, JsonValue],
) -> None:
    fields = {
        "id",
        "revision",
        "capability_type",
        "serialization_name",
        "config",
        "fingerprint",
    }
    if set(value) != fields:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    identity = value.get("id")
    revision = value.get("revision")
    capability_type = value.get("capability_type")
    serialization_name = value.get("serialization_name")
    config = value.get("config")
    fingerprint = value.get("fingerprint")
    if not isinstance(identity, str) or not identity.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if (
        not isinstance(capability_type, dict)
        or set(capability_type) != {"module", "qualname"}
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    module = capability_type.get("module")
    qualname = capability_type.get("qualname")
    if not isinstance(module, str) or not module.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(qualname, str) or not qualname.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(serialization_name, str) or not serialization_name.strip():
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(config, dict):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    if not isinstance(fingerprint, str) or not _is_fingerprint(fingerprint):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    expected = _runtime_descriptor_fingerprint(
        id=identity,
        revision=revision,
        module=module,
        qualname=qualname,
        serialization_name=serialization_name,
        config=cast("Mapping[str, JsonValue]", config),
    )
    if fingerprint != expected:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _is_fingerprint(value: "str | None") -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "CapabilityBinding",
    "CapabilityMaterializationContext",
    "CapabilityProvider",
    "CapabilityRefResolution",
    "RuntimeCapability",
    "validate_fingerprint",
]
