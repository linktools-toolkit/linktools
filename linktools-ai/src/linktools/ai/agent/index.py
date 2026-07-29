#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent specification index."""

from __future__ import annotations

from ..spec import SpecSource
from ..spec.index import SpecIndex
from ..spec.parsing import SpecLoader
from ..spec.source import SpecLoaderSource
from .codec import AgentSpecCodec
from .spec import AgentSpec


class AgentSpecIndex(SpecIndex[AgentSpec]):

    def __init__(
        self,
        source: SpecSource,
        *,
        codec: "AgentSpecCodec | None" = None,
        suffix: str = ".md",
        source_name: "str | None" = None,
    ) -> None:
        super().__init__(
            source,
            codec or AgentSpecCodec(),
            suffix=suffix,
            source_name=source_name,
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".md"
    ) -> "AgentSpecIndex":
        """Build an AgentSpecIndex over a SpecLoader (the common case: filesystem
        or asset-backed loader)."""
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, agent_id: str) -> AgentSpec:
        return await self._cache.get(agent_id)


__all__: "list[str]" = ["AgentSpecIndex"]
