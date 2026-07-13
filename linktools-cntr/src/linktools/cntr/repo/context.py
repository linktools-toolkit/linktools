#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where a container came from -- attached to each loaded ``BaseContainer``
as ``container.repo_context`` by ``ContainerLoader``."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linktools.types import PathType
    from linktools.core import ProjectProfile


class RepositoryConfigContext(object):
    """Private container-side record of where a container came from.

    Not exposed as public BaseContainer API beyond ``repo_context``
    -- Plan and the compose template context read it off
    ``container.repo_context``.

    ``file_config`` is this repository's own local-only ``ProjectProfile``
    (``None`` for the shared builtin context, which has no third-party
    ``.linktools.json`` of its own) -- consulted only for the
    ``requires.linktools-cntr`` compatibility gate, never for config field
    resolution (every container, builtin or third-party, resolves fields
    through the manager's own shared ``env_config`` -- see
    ``BaseContainer.env_config``). ``repo_name`` is the short,
    credential-free name for this repository (as opposed to ``url``, which
    may embed a Git credential) -- the only one safe to show a user (e.g.
    as a ``config list`` owner label).
    """

    def __init__(
            self,
            root_path: "PathType | None",
            file_config: "ProjectProfile | None",
            url: "str | None" = None,
            builtin: bool = False,
            repo_name: "str | None" = None,
    ):
        self.root_path = root_path
        self.file_config = file_config
        self.url = url
        self.builtin = builtin
        self.repo_name = repo_name
