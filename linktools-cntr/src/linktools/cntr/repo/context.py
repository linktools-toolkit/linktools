#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where a container came from and its resolved per-repository config --
attached to each loaded ``BaseContainer`` as
``container._repository_context`` by ``ContainerLoader``."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linktools.core import Config, ResolvedLinktoolsFileConfig
    from linktools.types import PathType


class RepositoryConfigContext(object):
    """Private container-side record of where a container came from and the
    Config it resolves fields through.

    Not exposed as public BaseContainer API beyond ``repository_context``/
    ``env_config`` -- Plan and the compose template context read it off
    ``container.repository_context``.

    ``file_config`` is this repository's ``ResolvedLinktoolsFileConfig``
    (``None`` for the shared builtin context, which has no third-party
    ``.linktools.json`` of its own). ``config`` is the Config every
    container loaded from this same repository shares --
    ``env``/``runtime``/``persistent`` state is shared process-wide (spec
    §33), only the local-file layer is unique to this repository.
    """

    def __init__(
            self,
            root_path: "PathType | None",
            file_config: "ResolvedLinktoolsFileConfig | None",
            config: "Config",
            url: "str | None" = None,
            builtin: bool = False,
    ):
        self.root_path = root_path
        self.file_config = file_config
        self.config = config
        self.url = url
        self.builtin = builtin
