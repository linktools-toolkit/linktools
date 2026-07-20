#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillCatalog: the skill domain Catalog over the generic Catalog contracts.

Same pattern as AgentCatalog: composes a CatalogSource (the raw {name}.md
origin), a SkillSpecCodec (strict decode), and a RevisionCache[SkillSpec].
Reuses the SpecLoaderSource adapter shape (the loader's int revision is
stringified; RegistryNotFoundError propagates).
"""

from __future__ import annotations

from ..catalog import CatalogSource, RevisionCache
from ..catalog.parsing import SpecLoader
from ..catalog.source import SpecLoaderSource
from .codec import SkillSpecCodec
from .models import SkillSpec


class SkillCatalog:
    """The skill domain Catalog: list/get SkillSpec over a CatalogSource with
    revision-cached, single-flight refresh and strict decode."""

    def __init__(
        self,
        source: CatalogSource,
        *,
        codec: "SkillSpecCodec | None" = None,
        suffix: str = ".md",
        source_name: "str | None" = None,
    ) -> None:
        self._source = source
        self._codec = codec or SkillSpecCodec()
        self._suffix = suffix
        self._cache: RevisionCache[SkillSpec] = RevisionCache(
            source, self._codec, suffix=suffix, source_name=source_name
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".md"
    ) -> "SkillCatalog":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, skill_id: str) -> SkillSpec:
        return await self._cache.get(skill_id)


__all__: "list[str]" = ["SkillCatalog"]
