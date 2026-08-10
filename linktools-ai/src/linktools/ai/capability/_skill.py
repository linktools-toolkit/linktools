#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill capability contracts, resolution, and Harness integration."""

from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef, SkillSpec
from ._contract import (
    CapabilityRefResolution,
    CapabilityRuntimeContext,
    validate_fingerprint,
)


class SkillProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve_ref(self, skill_id: str, revision: "int | None" = None) -> SkillSpec: ...


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    id: str
    revision: int
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1:
            raise ValueError("skill descriptor is incomplete")


class SkillCatalogView(Protocol):
    async def list_skills(self) -> "tuple[SkillDescriptor, ...]": ...
    async def load_skill(self, skill_id: str) -> "SkillSpec | None": ...


@dataclass(frozen=True, slots=True)
class SkillCatalogSnapshot(SkillCatalogView):
    descriptors: "tuple[SkillDescriptor, ...]"
    specifications: "tuple[SkillSpec, ...]"

    def __post_init__(self) -> None:
        descriptors = tuple(sorted(self.descriptors, key=lambda item: item.id))
        specifications = tuple(sorted(self.specifications, key=lambda item: item.id))
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "specifications", specifications)
        descriptor_ids = tuple(item.id for item in descriptors)
        specification_ids = tuple(item.id for item in specifications)
        if len(set(descriptor_ids)) != len(descriptor_ids) or len(set(specification_ids)) != len(specification_ids):
            raise ValueError("skill catalog contains duplicate ids")
        if set(descriptor_ids) != set(specification_ids):
            raise ValueError("skill catalog snapshot is incomplete")
        if any(descriptor.revision != specification.revision for descriptor, specification in zip(descriptors, specifications)):
            raise ValueError("skill catalog snapshot revision mismatch")

    async def list_skills(self) -> "tuple[SkillDescriptor, ...]":
        return self.descriptors

    async def load_skill(self, skill_id: str) -> "SkillSpec | None":
        return next((item for item in self.specifications if item.id == skill_id), None)


@dataclass(frozen=True, slots=True)
class SkillCapabilityBinding:
    resolutions: "tuple[CapabilityRefResolution, ...]"
    catalog: SkillCatalogView
    fingerprint: str
    inherit_to_subagents: bool = True

    @property
    def provider(self) -> str:
        return "skill"

    async def materialize(self, context: CapabilityRuntimeContext) -> "tuple[AbstractCapability[None], ...]":
        del context
        if not self.resolutions or not any(item.status == "resolved" for item in self.resolutions):
            return ()
        return (SkillCapability(self.catalog),)


class SkillCapabilityResolver:
    provider = "skill"

    def __init__(self, provider: SkillProvider) -> None:
        self._source = provider
        self._fingerprint = provider.manifest()
        validate_fingerprint(self._fingerprint)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def resolve(self, refs: "tuple[AgentCapabilityRef, ...]") -> SkillCapabilityBinding:
        resolutions: list[CapabilityRefResolution] = []
        specifications: list[SkillSpec] = []
        for ref in refs:
            try:
                specification = self._source.resolve_ref(ref.id, ref.revision)
            except (KeyError, LookupError):
                specification = None
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND:
                    specification = None
                else:
                    raise
            if specification is None:
                if ref.required:
                    raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
                resolutions.append(CapabilityRefResolution(ref.id, ref.revision, None, False, "unresolved", None))
                continue
            if specification.id != ref.id or (ref.revision is not None and specification.revision != ref.revision):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            fingerprint = canonical_sha256(
                {"id": specification.id, "revision": specification.revision, "content": specification.content}
            )
            resolutions.append(CapabilityRefResolution(ref.id, ref.revision, specification.revision, ref.required, "resolved", fingerprint))
            specifications.append(specification)
        descriptors = tuple(SkillDescriptor(item.id, item.revision, f"Authorized skill {item.id}") for item in specifications)
        binding_resolutions = tuple(resolutions)
        return SkillCapabilityBinding(
            binding_resolutions,
            SkillCatalogSnapshot(descriptors, tuple(specifications)),
            canonical_sha256(
                {
                    "provider": self.provider,
                    "resolver_fingerprint": self.fingerprint,
                    "inherit_to_subagents": True,
                    "configs": [dict(ref.config) for ref in refs],
                    "resolutions": [_resolution_payload(item) for item in binding_resolutions],
                }
            ),
        )


async def snapshot_skill_catalog(catalog: SkillCatalogView) -> SkillCatalogSnapshot:
    descriptors = await catalog.list_skills()
    specifications: list[SkillSpec] = []
    for descriptor in descriptors:
        specification = await catalog.load_skill(descriptor.id)
        if specification is None or specification.revision != descriptor.revision:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        specifications.append(specification)
    return SkillCatalogSnapshot(tuple(descriptors), tuple(specifications))


@dataclass
class SkillCapability(AbstractCapability[None]):
    catalog: SkillCatalogView

    def get_toolset(self) -> "FunctionToolset[None]":
        toolset = FunctionToolset[None](id="linktools-skill")

        @toolset.tool
        async def list_skills(ctx: RunContext[None]) -> "list[dict[str, str]]":
            """List skills authorized for this agent run."""
            del ctx
            descriptors = await self.catalog.list_skills()
            return [{"id": item.id, "description": item.description} for item in descriptors]

        @toolset.tool
        async def load_skill(ctx: RunContext[None], skill_id: str) -> "dict[str, str]":
            """Load the selected authorized skill by id."""
            del ctx
            specification = await self.catalog.load_skill(skill_id)
            if specification is None:
                raise ValueError("skill unavailable")
            return {"id": specification.id, "content": specification.content}

        return toolset

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


def _resolution_payload(resolution: CapabilityRefResolution) -> "dict[str, object]":
    return {
        "id": resolution.id,
        "requested_revision": resolution.requested_revision,
        "resolved_revision": resolution.resolved_revision,
        "required": resolution.required,
        "status": resolution.status,
        "fingerprint": resolution.fingerprint,
    }


__all__ = [
    "SkillCapability", "SkillCapabilityBinding", "SkillCapabilityResolver", "SkillCatalogSnapshot",
    "SkillCatalogView", "SkillDescriptor", "SkillProvider", "snapshot_skill_catalog",
]
