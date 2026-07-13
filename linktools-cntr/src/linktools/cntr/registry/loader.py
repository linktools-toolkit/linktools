#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container discovery and loading.

Scans builtin assets (depth 1) and each configured repo (depth 2), importing
the first concrete BaseContainer subclass in a ``container.py`` or falling
back to a SimpleContainer for a compose file.
"""
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from linktools.core import ProjectProfile
from linktools.errors import ConfigError, ConfigValidationError
from linktools.runtime import import_module_file

from ..container import BaseContainer, SimpleContainer
from ..repo.context import RepositoryConfigContext
from ..repo.requirements import ensure_requirement
from ..repo.service import safe_display_url
from ...capabilities.cntr import __cap_cntr__

if TYPE_CHECKING:
    from collections.abc import Iterator
    from linktools.types import PathType
    from ..manager import ContainerManager


@dataclass(frozen=True)
class ContainerLoadError:
    """One ``container.py``/compose-file discovery that failed to load --
    import error, ``on_init()`` error, or any other exception raised while
    constructing the container. ``expected_name`` is a best-effort guess
    (the directory's name with any numeric order prefix stripped, the same
    convention ``BaseContainer.__init__`` uses) -- ``None`` when the
    failure happened before a name could even be determined (e.g. a syntax
    error during import)."""
    path: str
    message: str
    expected_name: "str | None" = None


@dataclass(frozen=True)
class ContainerLoadResult:
    """``ContainerLoader.load_all()``'s return value: the containers that
    loaded successfully, plus every load failure as a structured
    ``ContainerLoadError`` instead of a log-only warning that leaves
    callers unable to tell "not installed" apart from "failed to load"."""
    containers: "list[BaseContainer]"
    errors: "list[ContainerLoadError]" = field(default_factory=list)


def _guess_container_name(path: "PathType") -> "str | None":
    import re
    basename = os.path.basename(str(path).rstrip(os.sep))
    match = re.match(r"^(\d{1,3})-(.*)$", basename, re.M | re.I)
    return match.group(2) if match else (basename or None)


class ContainerLoader:
    """Discover and instantiate containers behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def load_all(self) -> "ContainerLoadResult":
        manager = self.manager
        containers: "list[BaseContainer]" = []
        errors: "list[ContainerLoadError]" = []

        manager.logger.debug("Load containers from assets")
        asset_path = __cap_cntr__.get_asset_path("containers")
        builtin_context = RepositoryConfigContext(
            root_path=asset_path, file_config=None, url=None, builtin=True,
        )
        for container in self._walk(asset_path, max_level=1, repository=builtin_context, errors=errors):
            containers.append(container)

        # Builtin container fields form the manager's base schema. Register
        # them before loading any third-party repository so repo providers
        # (for example get_nginx_domain()) can resolve shared builtin fields
        # such as NGINX_ROOT_DOMAIN.
        for container in containers:
            container.register_configs()

        for url, meta in manager.repos.get_all().items():
            # Log messages only ever show the credential-free display form
            # -- a repository persisted before P1-07 rejected credential
            # URLs at add() time may still carry one. `url` (the real,
            # possibly credential-bearing value) is still what's stored on
            # RepositoryConfigContext below, since real Git operations need it.
            display_url = safe_display_url(url)
            manager.logger.debug(f"Load containers from repository `{display_url}`")
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path) or not os.path.isdir(repo_path):
                manager.logger.warning(f"Repository `{display_url}` not found, skip.")
                continue

            # requires.linktools-cntr compatibility is checked before this
            # repository's container.py is ever imported. A repository
            # without a .linktools.json loads exactly as before -- no
            # warning required. A repository whose local requirement isn't
            # satisfied is skipped: warned, not imported.
            try:
                file_config = ProjectProfile(ProjectProfile.local_path(repo_path))
                ensure_requirement(file_config, "linktools-cntr", __cap_cntr__.version)
            except (ConfigError, ConfigValidationError) as exc:
                manager.logger.warning(f"Repository `{display_url}` failed compatibility check, skip: {exc}")
                continue

            repo_context = RepositoryConfigContext(
                root_path=repo_path, file_config=file_config,
                url=url, builtin=False,
                repo_name=meta.get("repo_name"),
            )
            for container in self._walk(repo_path, max_level=2, repository=repo_context, errors=errors):
                containers.append(container)

        return ContainerLoadResult(containers=containers, errors=errors)

    def _walk(
            self, path: "PathType", max_level: int, repository: "RepositoryConfigContext",
            errors: "list[ContainerLoadError] | None" = None,
    ) -> "Iterator[BaseContainer]":
        if errors is None:
            errors = []
        if not os.path.isdir(path):
            return
        yield from self._load_one(path, repository, errors)
        if max_level <= 0:
            return
        for name in os.listdir(path):
            yield from self._walk(os.path.join(path, name), max_level - 1, repository, errors)

    def _load_one(
            self, path: "PathType", repository: "RepositoryConfigContext",
            errors: "list[ContainerLoadError] | None" = None,
    ) -> "Iterator[BaseContainer]":
        if errors is None:
            errors = []
        manager = self.manager
        container_path = os.path.join(path, manager.docker_container_name)
        if os.path.exists(container_path):
            try:
                name = path.replace(os.sep, ".")
                module = import_module_file(name, container_path)
                for value in module.__dict__.values():
                    if isinstance(value, type) and issubclass(value, BaseContainer):
                        # A container.py that imports a shared concrete base
                        # class (e.g. `from repo.common import CommonContainer`)
                        # must not have that imported class picked up instead
                        # of the subclass this file actually defines -- dict
                        # iteration order follows import order, so the import
                        # would otherwise be found first.
                        if not value.__abstract__ and value.__module__ == module.__name__:
                            container = value(manager, path)
                            container.repo_context = repository
                            manager.logger.debug(f"Load container {container.name} in {path}")
                            container.on_init()
                            yield container
                            return
            except Exception as e:
                manager.logger.warning(f"Failed to load container from `{path}`: {e}")
                errors.append(ContainerLoadError(
                    path=str(path), message=str(e), expected_name=_guess_container_name(path)))
                return

        for compose_name in manager.docker_compose_names:
            compose_path = os.path.join(path, compose_name)
            if os.path.exists(compose_path):
                container = SimpleContainer(manager, path)
                container.repo_context = repository
                manager.logger.debug(f"Load container {container.name} in {path}")
                container.on_init()
                yield container
                return
