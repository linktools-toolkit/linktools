#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent specification index."""


from ..spec.index import SpecIndex
from ..spec.source import SpecLoaderSource
from .codec import AgentSpecDocumentCodec
from .spec import AgentSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..spec import SpecSource
    from ..spec.parsing import SpecLoader

class AgentSpecIndex(SpecIndex[AgentSpec]):

    def __init__(
        self,
        source: "SpecSource",
        *,
        codec: "AgentSpecDocumentCodec | None" = None,
        suffix: str = ".md",
    ) -> None:
        super().__init__(
            source,
            codec or AgentSpecDocumentCodec(),
            suffix=suffix,
        )

    @classmethod
    def from_specloader(
        cls, loader: "SpecLoader", *, suffix: str = ".md"
    ) -> "AgentSpecIndex":
        """Build an AgentSpecIndex over a SpecLoader (the common case: filesystem
        or asset-backed loader)."""
        return cls(SpecLoaderSource(loader), suffix=suffix)


__all__ = ["AgentSpecIndex"]
