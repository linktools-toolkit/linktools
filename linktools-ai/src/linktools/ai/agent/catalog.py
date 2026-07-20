#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentCatalog: the agent domain Catalog over the generic Catalog contracts.

Composes a CatalogSource (the raw {name}.md origin), an AgentSpecCodec (strict
decode), and a RevisionCache[AgentSpec] (revision-keyed, atomic invalidation,
single-flight refresh). This is the agent-domain instance of the pattern the
plan (§4.3) prescribes; it replaces the old monolithic per-domain registry that
coupled Source + cache + parse.

``SpecLoaderSource`` adapts the existing ``SpecLoader`` (catalog/parsing) to the
CatalogSource Protocol -- the loader's revision is an int monotonic clock; the
adapter stringifies it (CatalogSource.revision() returns str per §4.3). The
source and codec propagate their native errors (RegistryNotFoundError /
InvalidSpecError / RegistryParseError), so the agent domain keeps its existing
rich error hierarchy after the migration.
"""

from __future__ import annotations

from ..catalog import CatalogSource, RevisionCache
from ..catalog.parsing import SpecLoader
from ..catalog.source import SpecLoaderSource
from .codec import AgentSpecCodec
from .spec import AgentSpec


class AgentCatalog:
    """The agent domain Catalog: list/get AgentSpec over a CatalogSource with
    revision-cached, single-flight refresh and strict decode.

    Construct directly with a CatalogSource + AgentSpecCodec, or via
    ``from_specloader`` for the common filesystem/loader case.
    """

    def __init__(
        self,
        source: CatalogSource,
        *,
        codec: "AgentSpecCodec | None" = None,
        suffix: str = ".md",
        source_name: "str | None" = None,
    ) -> None:
        self._source = source
        self._codec = codec or AgentSpecCodec()
        self._suffix = suffix
        self._cache: RevisionCache[AgentSpec] = RevisionCache(
            source, self._codec, suffix=suffix, source_name=source_name
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".md"
    ) -> "AgentCatalog":
        """Build an AgentCatalog over a SpecLoader (the common case: filesystem
        or resource-backed loader)."""
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, agent_id: str) -> AgentSpec:
        return await self._cache.get(agent_id)


__all__: "list[str]" = ["AgentCatalog"]
