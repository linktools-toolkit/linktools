#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmCatalog: the swarm domain Catalog over the generic Catalog contracts.

Same pattern as the other domain catalogs (items are ``{name}.yaml``).
"""

from __future__ import annotations

from ..catalog import CatalogSource, RevisionCache
from ..catalog.parsing import SpecLoader
from ..catalog.source import SpecLoaderSource
from .codec import SwarmSpecCodec
from .spec import SwarmSpec


class SwarmCatalog:
    """The swarm domain Catalog: list/get SwarmSpec over a CatalogSource with
    revision-cached, single-flight refresh and strict decode."""

    def __init__(
        self,
        source: CatalogSource,
        *,
        codec: "SwarmSpecCodec | None" = None,
        suffix: str = ".yaml",
        source_name: "str | None" = None,
    ) -> None:
        self._source = source
        self._codec = codec or SwarmSpecCodec()
        self._suffix = suffix
        self._cache: RevisionCache[SwarmSpec] = RevisionCache(
            source, self._codec, suffix=suffix, source_name=source_name
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "SwarmCatalog":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, swarm_id: str) -> SwarmSpec:
        return await self._cache.get(swarm_id)


__all__: "list[str]" = ["SwarmCatalog"]
