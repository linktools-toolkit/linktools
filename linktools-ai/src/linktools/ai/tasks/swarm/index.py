#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm specification index."""


from ...spec.index import SpecIndex
from ...spec.source import SpecLoaderSource
from .codec import SwarmSpecCodec
from .spec import SwarmSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...spec import SpecSource
    from ...spec.parsing import SpecLoader

class SwarmSpecIndex(SpecIndex[SwarmSpec]):

    def __init__(
        self,
        source: "SpecSource",
        *,
        codec: "SwarmSpecCodec | None" = None,
        suffix: str = ".yaml",
    ) -> None:
        super().__init__(
            source,
            codec or SwarmSpecCodec(),
            suffix=suffix,
        )

    @classmethod
    def from_specloader(
        cls, loader: "SpecLoader", *, suffix: str = ".yaml"
    ) -> "SwarmSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)


__all__: "list[str]" = ["SwarmSpecIndex"]
