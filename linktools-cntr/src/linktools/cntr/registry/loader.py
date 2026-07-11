#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container discovery and loading.

Scans builtin assets (depth 1) and each configured repo (depth 2), importing
the first concrete BaseContainer subclass in a ``container.py`` or falling
back to a SimpleContainer for a compose file.
"""
import os
from typing import TYPE_CHECKING

from linktools.core import ensure_requirement
from linktools.errors import ConfigError, ConfigValidationError
from linktools.runtime import import_module_file

from ..container import BaseContainer, SimpleContainer
from ..repo.context import RepositoryConfigContext
from ...capabilities.cntr import __cap_cntr__

if TYPE_CHECKING:
    from collections.abc import Iterator
    from linktools.types import PathType
    from ..container import BaseContainer as _BaseContainer
    from ..manager import ContainerManager


class ContainerLoader:
    """Discover and instantiate containers behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def load_all(self) -> "list[BaseContainer]":
        manager = self.manager
        containers: "list[BaseContainer]" = []

        manager.logger.debug("Load containers from assets")
        asset_path = __cap_cntr__.get_asset_path("containers")
        # Builtin containers share the manager's own Config (cwd-scoped
        # local file + global file) -- same as before this repository
        # config-context split existed.
        builtin_context = RepositoryConfigContext(
            root_path=asset_path, file_config=None, config=manager.env_config, url=None, builtin=True,
        )
        for container in self._walk(asset_path, max_level=1, repository=builtin_context):
            containers.append(container)

        # Builtin container fields form the manager's base schema. Register
        # them before constructing any third-party repository Config so repo
        # providers (for example get_nginx_domain()) can resolve shared
        # builtin fields such as NGINX_ROOT_DOMAIN.
        for container in containers:
            container.env_config.update_defaults(**container.configs)

        for url, meta in manager.repo_store.get_all().items():
            manager.logger.debug(f"Load containers from repository `{url}`")
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path) or not os.path.isdir(repo_path):
                manager.logger.warning(f"Repository `{url}` not found, skip.")
                continue

            # requires.linktools-cntr compatibility is checked before this
            # repository's container.py is ever imported. A repository
            # without a .linktools.json loads exactly as before -- no
            # warning required. A repository whose local requirement isn't
            # satisfied is skipped: warned, not imported.
            try:
                file_config = manager.environ.load_file_config(local_root=repo_path)
                ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
            except (ConfigError, ConfigValidationError) as exc:
                manager.logger.warning(f"Repository `{url}` failed compatibility check, skip: {exc}")
                continue

            repo_context = RepositoryConfigContext(
                root_path=repo_path, file_config=file_config,
                config=manager.build_repository_config(repo_path), url=url, builtin=False,
            )
            for container in self._walk(repo_path, max_level=2, repository=repo_context):
                containers.append(container)

        return containers

    def _walk(
            self, path: "PathType", max_level: int, repository: "RepositoryConfigContext",
    ) -> "Iterator[BaseContainer]":
        if not os.path.isdir(path):
            return
        yield from self._load_one(path, repository)
        if max_level <= 0:
            return
        for name in os.listdir(path):
            yield from self._walk(os.path.join(path, name), max_level - 1, repository)

    def _load_one(
            self, path: "PathType", repository: "RepositoryConfigContext",
    ) -> "Iterator[BaseContainer]":
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
                            container._repository_context = repository
                            manager.logger.debug(f"Load container {container.name} in {path}")
                            container.on_init()
                            yield container
                            return
            except Exception as e:
                manager.logger.warning(f"Failed to load container from `{path}`: {e}")
                return

        for compose_name in manager.docker_compose_names:
            compose_path = os.path.join(path, compose_name)
            if os.path.exists(compose_path):
                container = SimpleContainer(manager, path)
                container._repository_context = repository
                manager.logger.debug(f"Load container {container.name} in {path}")
                container.on_init()
                yield container
                return
