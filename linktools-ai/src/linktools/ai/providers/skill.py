#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillSpecProvider: source-agnostic surface for SkillSpec objects."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..registry.skill import SkillSpec


@runtime_checkable
class SkillSpecProvider(Protocol):
    """Provides SkillSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]":
        ...

    async def get(self, skill_id: str) -> "SkillSpec":
        ...
