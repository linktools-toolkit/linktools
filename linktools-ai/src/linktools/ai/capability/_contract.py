#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability binding and runtime materialization contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability

from ..core import Principal, ResourceRef, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef

_FINGERPRINT_LENGTH = 64


@dataclass(frozen=True, slots=True)
class CapabilityRefResolution:
    id: str
    requested_revision: "int | None"
    resolved_revision: "int | None"
    required: bool
    status: "Literal['resolved', 'unresolved']"
    fingerprint: "str | None"

    def __post_init__(self) -> None:
        if (
            not self.id.strip()
            or self.requested_revision is not None
            and (not isinstance(self.requested_revision, int) or isinstance(self.requested_revision, bool) or self.requested_revision < 1)
            or self.resolved_revision is not None
            and (not isinstance(self.resolved_revision, int) or isinstance(self.resolved_revision, bool) or self.resolved_revision < 1)
        ):
            raise ValueError("capability resolution identity is invalid")
        if self.status == "resolved" and not _is_fingerprint(self.fingerprint):
            raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)
        if self.status == "unresolved" and (self.required or self.fingerprint is not None or self.resolved_revision is not None):
            raise ValueError("unresolved capability resolution is invalid")
        if self.status not in {"resolved", "unresolved"}:
            raise ValueError("capability resolution status is invalid")
        if self.status == "resolved" and self.resolved_revision is None:
            raise ValueError("resolved capability revision is required")


@dataclass(frozen=True, slots=True)
class CapabilityRuntimeContext:
    principal: Principal
    execution: ResourceRef

    def __post_init__(self) -> None:
        if self.principal.tenant_id != self.execution.tenant_id:
            raise ValueError("capability context tenant mismatch")


class CapabilityBinding(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def resolutions(self) -> "tuple[CapabilityRefResolution, ...]": ...

    @property
    def fingerprint(self) -> str: ...

    @property
    def inherit_to_subagents(self) -> bool: ...

    async def materialize(
        self,
        context: CapabilityRuntimeContext,
    ) -> "tuple[PydanticAgentCapability[None], ...]": ...


@dataclass(frozen=True, slots=True)
class CapabilityInjection:
    id: str
    fingerprint: str
    capability: "PydanticAgentCapability[None]"
    inherit_to_subagents: bool = True

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("capability injection id is required")
        if not _is_fingerprint(self.fingerprint):
            raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


@dataclass(frozen=True, slots=True)
class UnresolvedCapabilityBinding:
    provider: str
    resolutions: "tuple[CapabilityRefResolution, ...]"
    fingerprint: str
    inherit_to_subagents: bool = False

    def __post_init__(self) -> None:
        if not self.provider.strip() or not _is_fingerprint(self.fingerprint):
            raise ValueError("unresolved capability binding is invalid")
        if not self.resolutions or any(item.status != "unresolved" for item in self.resolutions):
            raise ValueError("unresolved capability binding requires unresolved resolutions")

    async def materialize(
        self,
        context: CapabilityRuntimeContext,
    ) -> "tuple[PydanticAgentCapability[None], ...]":
        del context
        return ()


def unresolved_binding(provider: str, refs: Sequence[AgentCapabilityRef]) -> UnresolvedCapabilityBinding:
    resolutions = tuple(CapabilityRefResolution(ref.id, ref.revision, None, False, "unresolved", None) for ref in refs)
    return UnresolvedCapabilityBinding(
        provider,
        resolutions,
        canonical_sha256(
            {
                "provider": provider,
                "status": "UNRESOLVED_PROVIDER",
                "refs": [
                    {
                        "id": ref.id,
                        "requested_revision": ref.revision,
                        "required": ref.required,
                        "config": dict(ref.config),
                    }
                    for ref in refs
                ],
            }
        ),
    )


def group_capability_refs(
    refs: Sequence[AgentCapabilityRef],
) -> "tuple[tuple[str, tuple[AgentCapabilityRef, ...]], ...]":
    """Group refs by first-seen provider while preserving declaration order."""
    grouped: dict[str, list[AgentCapabilityRef]] = {}
    order: list[str] = []
    for ref in refs:
        if ref.provider not in grouped:
            grouped[ref.provider] = []
            order.append(ref.provider)
        grouped[ref.provider].append(ref)
    return tuple((provider, tuple(grouped[provider])) for provider in order)


def _is_fingerprint(value: "str | None") -> bool:
    return isinstance(value, str) and len(value) == _FINGERPRINT_LENGTH and all(character in "0123456789abcdef" for character in value)


def validate_fingerprint(value: str) -> None:
    if not _is_fingerprint(value):
        raise AIError(ErrorCode.CAPABILITY_FINGERPRINT_INVALID)


__all__ = [
    "CapabilityBinding", "CapabilityInjection", "CapabilityRefResolution", "CapabilityRuntimeContext",
    "UnresolvedCapabilityBinding", "group_capability_refs", "unresolved_binding", "validate_fingerprint",
]
