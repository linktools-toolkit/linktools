#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where a container came from and its resolved per-repository config --
attached to each loaded ``BaseContainer`` as
``container._repository_context`` by ``ContainerLoader``."""
from typing import TYPE_CHECKING

from linktools.core import ConfigSource
from linktools.errors import ConfigNotFoundError
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any
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
    container loaded from this same repository shares -- ``env``/
    ``runtime``/``persistent`` state is shared process-wide, only the
    local-file layer is unique to this repository. ``repo_name`` is the
    short, credential-free name for this repository (as opposed to
    ``url``, which may embed a Git credential) -- the only one safe to
    show a user (e.g. as a ``config list`` owner label).
    """

    def __init__(
            self,
            root_path: "PathType | None",
            file_config: "ResolvedLinktoolsFileConfig | None",
            config: "Config",
            url: "str | None" = None,
            builtin: bool = False,
            repo_name: "str | None" = None,
    ):
        self.root_path = root_path
        self.file_config = file_config
        self.config = config
        self.url = url
        self.builtin = builtin
        self.repo_name = repo_name


class ManagerConfigSource(ConfigSource):
    """Expose a fixed set of manager-owned field values to a repository's
    Config, always through the manager's own Config resolution.

    Inserted into a repository Config's source chain (ahead of that
    repository's local/global file, see
    ``ContainerManager.build_repository_config``) so a value this source
    covers always wins over anything a third-party ``.linktools.json``
    declares for the same key -- the manager, not the repository, owns
    these fields (HOST, DOCKER_APP_PATH, ...). A key outside ``keys`` is
    reported absent, so it falls through untouched to the repository's own
    local file / provider / default -- this source only ever answers for
    the fixed key set it was built with.
    """

    name = "manager-config"
    before_provider = True

    def __init__(self, manager_config: "Config", keys: "Any") -> None:
        self._config = manager_config
        self._keys = frozenset(keys)

    def get(self, key: str) -> "tuple[Any, bool]":
        if key not in self._keys:
            return (MISSING, False)
        try:
            return (self._config.get(key), True)
        except ConfigNotFoundError:
            return (MISSING, False)

    def keys(self) -> "list[str]":
        return sorted(self._keys)
