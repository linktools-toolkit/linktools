#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP server specification index."""

from __future__ import annotations

from ...spec import SpecSource
from ...spec.index import SpecIndex
from ...spec.parsing import SpecLoader
from ...spec.source import SpecLoaderSource
from .codec import MCPSpecCodec
from .spec import MCPServerSpec


class MCPServerSpecIndex(SpecIndex[MCPServerSpec]):

    def __init__(
        self,
        source: SpecSource,
        *,
        codec: "MCPSpecCodec | None" = None,
        suffix: str = ".yaml",
        source_name: "str | None" = None,
    ) -> None:
        super().__init__(
            source,
            codec or MCPSpecCodec(),
            suffix=suffix,
            source_name=source_name,
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "MCPServerSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, mcp_id: str) -> MCPServerSpec:
        return await self._cache.get(mcp_id)


__all__: "list[str]" = ["MCPServerSpecIndex"]
