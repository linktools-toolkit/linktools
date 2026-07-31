#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP server specification index."""


from ...spec.index import SpecIndex
from ...spec.source import SpecLoaderSource
from .codec import MCPSpecCodec
from .spec import MCPServerSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...spec import SpecSource
    from ...spec.parsing import SpecLoader

class MCPServerSpecIndex(SpecIndex[MCPServerSpec]):

    def __init__(
        self,
        source: "SpecSource",
        *,
        codec: "MCPSpecCodec | None" = None,
        suffix: str = ".yaml",
    ) -> None:
        super().__init__(
            source,
            codec or MCPSpecCodec(),
            suffix=suffix,
        )

    @classmethod
    def from_specloader(
        cls, loader: "SpecLoader", *, suffix: str = ".yaml"
    ) -> "MCPServerSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)


__all__: "list[str]" = ["MCPServerSpecIndex"]
