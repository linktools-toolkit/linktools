#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm specification index."""

from __future__ import annotations

from ...spec import SpecSource
from ...spec.index import SpecIndex
from ...spec.parsing import SpecLoader
from ...spec.source import SpecLoaderSource
from .codec import SwarmSpecCodec
from .spec import SwarmSpec


class SwarmSpecIndex(SpecIndex[SwarmSpec]):

    def __init__(
        self,
        source: SpecSource,
        *,
        codec: "SwarmSpecCodec | None" = None,
        suffix: str = ".yaml",
        source_name: "str | None" = None,
    ) -> None:
        super().__init__(
            source,
            codec or SwarmSpecCodec(),
            suffix=suffix,
            source_name=source_name,
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "SwarmSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, swarm_id: str) -> SwarmSpec:
        return await self._cache.get(swarm_id)


__all__: "list[str]" = ["SwarmSpecIndex"]
