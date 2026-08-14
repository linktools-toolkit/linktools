#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill binding contracts and Pydantic AI integration."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from ..asset import AssetRef, AssetRepository
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef, SkillSpec
from ._contract import (
    CapabilityBinding,
    CapabilityRefResolution,
    CapabilityMaterializationContext,
)

SKILL_TOOL_NAMES = ("list_skills", "load_skill")


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
    catalog: SkillCatalogSnapshot
    fingerprint: str

    @property
    def id(self) -> str:
        return "skill"

    @property
    def provider(self) -> str:
        return "skill"

    async def materialize(self, context: CapabilityMaterializationContext) -> "tuple[AbstractCapability[None], ...]":
        selected_ids = frozenset(
            resolution.id
            for resolution in self.resolutions
            if resolution.status == "resolved" and _allowed(resolution.id, context.allow_skills)
        )
        selected_tools = tuple(name for name in SKILL_TOOL_NAMES if _allowed(name, context.allow_tools))
        if not selected_ids or not selected_tools:
            return ()
        catalog = SkillCatalogSnapshot(
            tuple(item for item in self.catalog.descriptors if item.id in selected_ids),
            tuple(item for item in self.catalog.specifications if item.id in selected_ids),
        )
        return (SkillCapability(catalog, selected_tools),)


def bind_skill_capability(
    refs: "Sequence[AgentCapabilityRef]",
    specifications: "Sequence[SkillSpec | None]",
) -> SkillCapabilityBinding:
    """Compile resolved Skill declarations into one immutable capability binding."""
    if len(refs) != len(specifications):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    resolutions: list[CapabilityRefResolution] = []
    resolved: list[SkillSpec] = []
    for ref, specification in zip(refs, specifications):
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
        resolved.append(specification)
    binding_resolutions = tuple(resolutions)
    descriptors = tuple(SkillDescriptor(item.id, item.revision, f"Authorized skill {item.id}") for item in resolved)
    return SkillCapabilityBinding(
        binding_resolutions,
        SkillCatalogSnapshot(descriptors, tuple(resolved)),
        canonical_sha256(
            {
                "provider": "skill",
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
    selected_tools: "tuple[str, ...]" = SKILL_TOOL_NAMES

    def get_instructions(self) -> "Callable[[RunContext[None]], Awaitable[str | None]]":
        async def render(ctx: RunContext[None]) -> "str | None":
            del ctx
            descriptors = await self.catalog.list_skills()
            if not descriptors:
                return None
            lines = ["The following skills are available for this agent run."]
            if "load_skill" in self.selected_tools:
                lines.append("Use the `load_skill` tool to load the full instructions for a skill when it is relevant.")
            lines.extend(f"- {item.id}: {item.description}" for item in descriptors)
            return "\n".join(lines)

        return render

    def get_toolset(self) -> "FunctionToolset[None]":
        toolset = FunctionToolset[None](id="linktools-skill")

        @toolset.tool
        async def list_skills(ctx: RunContext[None]) -> "list[dict[str, str]]":
            """List skills available for this agent run."""
            del ctx
            descriptors = await self.catalog.list_skills()
            return [{"id": item.id, "description": item.description} for item in descriptors]

        @toolset.tool
        async def load_skill(ctx: RunContext[None], skill_id: str) -> "dict[str, str]":
            """Load the full instructions for a selected skill."""
            del ctx
            specification = await self.catalog.load_skill(skill_id)
            if specification is None:
                raise ValueError("skill not found")
            return {"id": specification.id, "content": specification.content}

        return toolset.filtered(lambda _ctx, definition: definition.name in self.selected_tools)

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


class SkillCapabilityProvider:
    """Resolve skills and bootstrap refs through the logical AssetRepository."""

    provider = "skill"

    async def bind(self, refs: "tuple[AgentCapabilityRef, ...]", *, assets: AssetRepository) -> CapabilityBinding:
        values: list[SkillSpec | None] = []
        for ref in refs:
            try:
                resolved = await assets.resolve(AssetRef("skill", ref.id))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND and not ref.required:
                    values.append(None)
                    continue
                raise
            if not isinstance(resolved.spec, SkillSpec):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            values.append(resolved.spec)
        return bind_skill_capability(refs, values)


async def merge_skill_catalogs(catalogs: "tuple[SkillCatalogView, ...]") -> SkillCatalogSnapshot:
    descriptors: list[SkillDescriptor] = []
    specifications: list[SkillSpec] = []
    for catalog in catalogs:
        snapshot = await snapshot_skill_catalog(catalog)
        descriptors.extend(snapshot.descriptors)
        specifications.extend(snapshot.specifications)
    return SkillCatalogSnapshot(tuple(descriptors), tuple(specifications))


def _resolution_payload(resolution: CapabilityRefResolution) -> "dict[str, object]":
    return {
        "id": resolution.id,
        "requested_revision": resolution.requested_revision,
        "resolved_revision": resolution.resolved_revision,
        "required": resolution.required,
        "status": resolution.status,
        "fingerprint": resolution.fingerprint,
    }


def _allowed(value: str, allowlist: "tuple[str, ...]") -> bool:
    return "*" in allowlist or value in allowlist


__all__ = [
    "SkillCapabilityProvider", "SKILL_TOOL_NAMES", "SkillCapability", "SkillCapabilityBinding", "SkillCatalogSnapshot", "SkillCatalogView", "SkillDescriptor",
    "bind_skill_capability", "merge_skill_catalogs", "snapshot_skill_catalog",
]
