#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where a container came from -- attached to each loaded ``BaseContainer``
as ``container._repository`` by ``ContainerLoader``."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linktools.core import LinktoolsManifest
    from linktools.types import PathType


@dataclass(frozen=True)
class ContainerRepositoryContext:
    """Private container-side record of where a container came from.

    Not exposed as public BaseContainer API -- Plan and the runtime
    requirement gate read it directly off ``container._repository``.
    """
    url: "str | None"
    root_path: "PathType | None"
    manifest: "LinktoolsManifest | None"
    builtin: bool
