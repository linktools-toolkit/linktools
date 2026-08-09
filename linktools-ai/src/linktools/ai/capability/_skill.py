#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-authorized Linktools skill capability."""

from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext

@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    revision: int
    content: str

    def __post_init__(self) -> None:
        if not self.id.strip() or self.revision < 1:
            raise ValueError("skill spec is incomplete")

    @property
    def asset_kind(self) -> str:
        return "skill"

    @property
    def asset_id(self) -> str:
        return self.id


class SkillProvider(Protocol):
    def manifest(self) -> str: ...
    def resolve_ref(self, skill_id: str, revision: 'int | None' = None) -> SkillSpec: ...


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
    descriptors: tuple[SkillDescriptor, ...]
    specifications: tuple[SkillSpec, ...]

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


@dataclass
class SkillCapability(AbstractCapability[None]):
    catalog: SkillCatalogView

    def get_toolset(self) -> "FunctionToolset[None]":
        toolset = FunctionToolset[None](id="linktools-skill")

        @toolset.tool
        async def list_skills(ctx: RunContext[None]) -> list[dict[str, str]]:
            """List skills authorized for this agent run."""
            del ctx
            descriptors = await self.catalog.list_skills()
            return [{"id": item.id, "description": item.description} for item in descriptors]

        @toolset.tool
        async def load_skill(ctx: RunContext[None], skill_id: str) -> dict[str, str]:
            """Load the selected authorized skill by id."""
            del ctx
            specification = await self.catalog.load_skill(skill_id)
            if specification is None:
                raise ValueError("skill unavailable")
            return {"id": specification.id, "content": specification.content}

        return toolset

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None


__all__ = [
    "SkillCatalogSnapshot", "SkillCatalogView", "SkillCapability", "SkillDescriptor",
    "SkillProvider", "SkillSpec",
]
