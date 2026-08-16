#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability binding, runtime capability, and provider contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability

from ..asset import AssetRepository
from ..core import Principal, ResourceRef, canonical_sha256, canonical_string_tuple
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef


@dataclass(frozen=True, slots=True)
class CapabilityRefResolution:
    id: str
    requested_revision: "int | None"
    resolved_revision: "int | None"
    required: bool
    status: Literal["resolved", "unresolved"]
    fingerprint: "str | None"

    def __post_init__(self) -> None:
        if not self.id.strip() or not isinstance(self.required, bool):
            raise ValueError("capability resolution identity is invalid")
        if self.status == "resolved" and (self.resolved_revision is None or not _is_fingerprint(self.fingerprint)):
            raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)
        if self.status == "unresolved" and (self.required or self.resolved_revision is not None or self.fingerprint is not None):
            raise ValueError("unresolved capability resolution is invalid")
        if self.status not in {"resolved", "unresolved"}:
            raise ValueError("capability resolution status is invalid")


@dataclass(frozen=True, slots=True)
class CapabilityMaterializationContext:
    principal: Principal
    execution: ResourceRef
    execution_root: Path
    allow_tools: "tuple[str, ...]" = ("*",)
    allow_skills: "tuple[str, ...]" = ("*",)

    def __post_init__(self) -> None:
        if self.principal.tenant_id != self.execution.tenant_id:
            raise ValueError("capability context tenant mismatch")
        if not isinstance(self.execution_root, Path):
            raise TypeError("execution_root must be a Path")
        object.__setattr__(self, "execution_root", self.execution_root.expanduser().resolve(strict=False))
        object.__setattr__(self, "allow_tools", canonical_string_tuple(self.allow_tools, field="allow_tools"))
        object.__setattr__(self, "allow_skills", canonical_string_tuple(self.allow_skills, field="allow_skills"))


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
    @property
    def provider(self) -> str: ...

    async def bind(
        self,
        refs: "tuple[AgentCapabilityRef, ...]",
        *,
        assets: AssetRepository,
    ) -> CapabilityBinding: ...


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    id: str
    capability: "PydanticAgentCapability[None]"
    revision: "int | None" = None
    inherit_to_subagents: bool = True

    @property
    def provider(self) -> str:
        return "runtime"

    @property
    def resolutions(self) -> "tuple[CapabilityRefResolution, ...]":
        return ()

    @property
    def fingerprint(self) -> str:
        payload: dict[str, object] = {"provider": self.provider, "id": self.id}
        if self.revision is not None:
            payload["revision"] = self.revision
        return canonical_sha256(payload)

    def __post_init__(self) -> None:
        if (
            not self.id.strip()
            or self.capability is None
            or (
                self.revision is not None
                and (
                    not isinstance(self.revision, int)
                    or isinstance(self.revision, bool)
                    or self.revision < 1
                )
            )
            or not isinstance(self.inherit_to_subagents, bool)
        ):
            raise ValueError("runtime capability is incomplete")

    async def materialize(self, context: CapabilityMaterializationContext) -> "tuple[PydanticAgentCapability[None], ...]":
        del context
        return (self.capability,)


@dataclass(frozen=True, slots=True)
class UnresolvedCapabilityBinding:
    provider: str
    resolutions: "tuple[CapabilityRefResolution, ...]"
    fingerprint: str

    @property
    def id(self) -> str:
        return f"unresolved:{self.provider}"

    async def materialize(self, context: CapabilityMaterializationContext) -> "tuple[PydanticAgentCapability[None], ...]":
        del context
        return ()


def unresolved_binding(provider: str, refs: Sequence[AgentCapabilityRef]) -> UnresolvedCapabilityBinding:
    resolutions = tuple(CapabilityRefResolution(ref.id, ref.revision, None, False, "unresolved", None) for ref in refs)
    return UnresolvedCapabilityBinding(
        provider,
        resolutions,
        canonical_sha256({"provider": provider, "status": "UNRESOLVED_PROVIDER", "refs": [_ref_payload(ref) for ref in refs]}),
    )


def group_capability_refs(refs: Sequence[AgentCapabilityRef]) -> "tuple[tuple[str, tuple[AgentCapabilityRef, ...]], ...]":
    grouped: dict[str, list[AgentCapabilityRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.provider, []).append(ref)
    return tuple((provider, tuple(values)) for provider, values in grouped.items())


def validate_fingerprint(value: str) -> None:
    if not _is_fingerprint(value):
        raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


def _is_fingerprint(value: "str | None") -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _ref_payload(ref: AgentCapabilityRef) -> dict[str, object]:
    return {"provider": ref.provider, "id": ref.id, "revision": ref.revision, "required": ref.required, "config": dict(ref.config)}


__all__ = [
    "CapabilityBinding",
    "RuntimeCapability",
    "CapabilityMaterializationContext",
    "CapabilityProvider",
    "CapabilityRefResolution",
    "UnresolvedCapabilityBinding",
    "group_capability_refs",
    "unresolved_binding",
    "validate_fingerprint",
]
