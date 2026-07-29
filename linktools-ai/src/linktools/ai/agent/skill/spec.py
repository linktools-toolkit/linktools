"""Skill declaration and provider contract."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """An immutable skill declaration parsed from Markdown frontmatter."""

    id: str
    name: str
    description: str = ""
    instructions: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SkillSpecProvider(Protocol):
    async def list_ids(self) -> tuple[str, ...]: ...

    async def get(self, skill_id: str) -> SkillSpec: ...


__all__ = ["SkillSpec", "SkillSpecProvider"]
