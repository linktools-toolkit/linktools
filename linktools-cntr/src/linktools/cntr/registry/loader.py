#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container discovery and loading.

Scans builtin assets (depth 1) and each configured repo (depth 2), importing
the first concrete BaseContainer subclass in a ``container.py`` or falling
back to a SimpleContainer for a compose file.
"""
import os
from typing import TYPE_CHECKING

from linktools.runtime import import_module_file

from ..container import BaseContainer, SimpleContainer
from ..repo.context import ContainerRepositoryContext
from ..repo.manifest import ContainerManifestError
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
        builtin_context = ContainerRepositoryContext(
            url=None, root_path=asset_path, manifest=None, builtin=True,
        )
        for container in self._walk(asset_path, max_level=1, repository=builtin_context):
            containers.append(container)

        for url, meta in manager.repo_store.get_all().items():
            manager.logger.debug(f"Load containers from repository `{url}`")
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path) or not os.path.isdir(repo_path):
                manager.logger.warning(f"Repository `{url}` not found, skip.")
                continue

            # Manifest schema/kind/components.cntr/host-requirement
            # compatibility is validated before this repository's
            # container.py is ever imported. A repository without a
            # .linktools.json (a project without a manifest) loads exactly
            # as before -- no warning, no migration required. A manifest
            # present but missing (or not opted into) the cntr component is
            # skipped the same way: it simply hasn't declared cntr
            # capability for this project.
            try:
                manifest = manager.manifest_policy.load(repo_path)
                manager.manifest_policy.ensure_loadable(manifest)
            except ContainerManifestError as exc:
                manager.logger.warning(f"Repository `{url}` failed manifest validation, skip: {exc}")
                continue

            repo_context = ContainerRepositoryContext(
                url=url, root_path=repo_path, manifest=manifest, builtin=False,
            )
            for container in self._walk(repo_path, max_level=2, repository=repo_context):
                containers.append(container)

        return containers

    def _walk(
            self, path: "PathType", max_level: int, repository: "ContainerRepositoryContext",
    ) -> "Iterator[BaseContainer]":
        if not os.path.isdir(path):
            return
        yield from self._load_one(path, repository)
        if max_level <= 0:
            return
        for name in os.listdir(path):
            yield from self._walk(os.path.join(path, name), max_level - 1, repository)

    def _load_one(
            self, path: "PathType", repository: "ContainerRepositoryContext",
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
                            container._repository = repository
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
                container._repository = repository
                manager.logger.debug(f"Load container {container.name} in {path}")
                container.on_init()
                yield container
                return
