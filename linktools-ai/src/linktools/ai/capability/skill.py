#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill provider boundary."""

from dataclasses import dataclass
from typing import Protocol


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


__all__ = ["SkillProvider", "SkillSpec"]
