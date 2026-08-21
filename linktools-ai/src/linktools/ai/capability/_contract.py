#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability binding, runtime capability, and provider contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability

from ..asset import AssetRef, AssetRepository
from ..core import Principal, ResourceRef, canonical_sha256, canonical_string_tuple
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
        object.__setattr__(self, "execution_root", self.execution_root.expanduser().resolve(strict=False))
        object.__setattr__(self, "allow_tools", canonical_string_tuple(self.allow_tools, field="allow_tools"))


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
    """Convert every Asset of one exact value type into one frozen capability binding."""

    @property
    def provider(self) -> str: ...

    @property
    def value_type(self) -> type[object]: ...

    async def bind(
        self,
        refs: "tuple[AssetRef, ...]",
        *,
        assets: AssetRepository,
    ) -> CapabilityBinding: ...


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    """Stable Python capability binding supplied by the Runtime composition root."""

    id: str
    capability: "PydanticAgentCapability[None]"
    revision: int = 1
    inherit_to_subagents: bool = True

    @property
    def provider(self) -> str:
        return "runtime"

    @property
    def resolutions(self) -> "tuple[CapabilityRefResolution, ...]":
        return ()

    @property
    def fingerprint(self) -> str:
        capability_type = type(self.capability)
        return canonical_sha256(
            {
                "provider": self.provider,
                "id": self.id,
                "revision": self.revision,
                "type": f"{capability_type.__module__}.{capability_type.__qualname__}",
                "inherit_to_subagents": self.inherit_to_subagents,
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
            or not isinstance(self.inherit_to_subagents, bool)
        ):
            raise ValueError("runtime capability is incomplete")

    async def materialize(
        self,
        context: CapabilityMaterializationContext,
    ) -> "tuple[PydanticAgentCapability[None], ...]":
        del context
        return (self.capability,)


def validate_fingerprint(value: str) -> None:
    if not _is_fingerprint(value):
        raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


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
