#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill specification index."""


from ...spec.index import SpecIndex
from ...spec.source import SpecLoaderSource
from .codec import SkillSpecCodec
from .spec import SkillSpec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...spec import SpecSource
    from ...spec.parsing import SpecLoader

class SkillSpecIndex(SpecIndex[SkillSpec]):

    def __init__(
        self,
        source: "SpecSource",
        *,
        codec: "SkillSpecCodec | None" = None,
        suffix: str = ".md",
    ) -> None:
        super().__init__(
            source,
            codec or SkillSpecCodec(),
            suffix=suffix,
        )

    @classmethod
    def from_specloader(
        cls, loader: "SpecLoader", *, suffix: str = ".md"
    ) -> "SkillSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)


__all__: "list[str]" = ["SkillSpecIndex"]
