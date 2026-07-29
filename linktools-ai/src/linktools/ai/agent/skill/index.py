#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill specification index."""

from __future__ import annotations

from ...spec import SpecSource
from ...spec.index import SpecIndex
from ...spec.parsing import SpecLoader
from ...spec.source import SpecLoaderSource
from .codec import SkillSpecCodec
from .spec import SkillSpec


class SkillSpecIndex(SpecIndex[SkillSpec]):

    def __init__(
        self,
        source: SpecSource,
        *,
        codec: "SkillSpecCodec | None" = None,
        suffix: str = ".md",
        source_name: "str | None" = None,
    ) -> None:
        super().__init__(
            source,
            codec or SkillSpecCodec(),
            suffix=suffix,
            source_name=source_name,
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".md"
    ) -> "SkillSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, skill_id: str) -> SkillSpec:
        return await self._cache.get(skill_id)


__all__: "list[str]" = ["SkillSpecIndex"]
