#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill domain models.

SkillSpec is the immutable skill DECLARATION (parsed from {name}.md frontmatter
+ body); SkillSummary / SkillContent are the lightweight runtime forms injected
into the prompt catalog / returned by read_skill."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """An immutable skill declaration parsed from a ``{name}.md`` item
    (frontmatter + markdown body). Previously defined in registry/skill.py;
    moved here so the skill domain owns its spec type."""

    id: str
    name: str
    description: str = ""
    instructions: str = ""
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class SkillSpecProvider(Protocol):
    """Provides SkillSpec objects from any configuration source.

    Lives in the skill domain (co-located with SkillSpec) so the skill package
    does not import providers back just to reference its own provider surface
    (the providers package re-exports it for RuntimeDependencies)."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, skill_id: str) -> SkillSpec: ...



class SkillSummary(BaseModel):
    id: str
    name: str
    description: "str | None" = None
    tags: "list[str]" = Field(default_factory=list)
    extension_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)


class SkillContent(BaseModel):
    id: str
    name: str
    description: "str | None" = None
    content: str
    extension_id: "str | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)
