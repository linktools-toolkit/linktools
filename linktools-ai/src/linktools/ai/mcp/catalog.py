#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPCatalog: the mcp domain Catalog over the generic Catalog contracts.

Same pattern as AgentCatalog / SkillCatalog (here items are ``{name}.yaml``).
"""

from __future__ import annotations

from ..catalog import CatalogSource, RevisionCache
from ..catalog.parsing import SpecLoader
from ..catalog.source import SpecLoaderSource
from .codec import MCPSpecCodec
from .spec import MCPServerSpec


class MCPCatalog:
    """The mcp domain Catalog: list/get MCPServerSpec over a CatalogSource with
    revision-cached, single-flight refresh and strict decode."""

    def __init__(
        self,
        source: CatalogSource,
        *,
        codec: "MCPSpecCodec | None" = None,
        suffix: str = ".yaml",
        source_name: "str | None" = None,
    ) -> None:
        self._source = source
        self._codec = codec or MCPSpecCodec()
        self._suffix = suffix
        self._cache: RevisionCache[MCPServerSpec] = RevisionCache(
            source, self._codec, suffix=suffix, source_name=source_name
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "MCPCatalog":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, mcp_id: str) -> MCPServerSpec:
        return await self._cache.get(mcp_id)


__all__: "list[str]" = ["MCPCatalog"]
