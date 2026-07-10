#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container discovery and loading (refactor spec Phase 4).

Extracted verbatim from ContainerManager._load_containers / _walk_containers /
_load_container. Behavior is unchanged: scan builtin assets (depth 1) and each
configured repo (depth 2), importing the first concrete BaseContainer subclass
in a ``container.py`` or falling back to a SimpleContainer for a compose file.
"""
import os
from typing import TYPE_CHECKING

from linktools.runtime import import_module_file

from ..container import BaseContainer, SimpleContainer
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
        for container in self._walk(asset_path, max_level=1):
            containers.append(container)

        for url, meta in manager.get_all_repos().items():
            manager.logger.debug(f"Load containers from repository `{url}`")
            repo_path = meta.get("repo_path")
            if not repo_path or not os.path.exists(repo_path) or not os.path.isdir(repo_path):
                manager.logger.warning(f"Repository `{url}` not found, skip.")
                continue
            for container in self._walk(repo_path, max_level=2):
                containers.append(container)

        return containers

    def _walk(self, path: "PathType", max_level: int) -> "Iterator[BaseContainer]":
        if not os.path.isdir(path):
            return
        yield from self._load_one(path)
        if max_level <= 0:
            return
        for name in os.listdir(path):
            yield from self._walk(os.path.join(path, name), max_level - 1)

    def _load_one(self, path: "PathType") -> "Iterator[BaseContainer]":
        manager = self.manager
        container_path = os.path.join(path, manager.docker_container_name)
        if os.path.exists(container_path):
            try:
                name = path.replace(os.sep, ".")
                module = import_module_file(name, container_path)
                for value in module.__dict__.values():
                    if isinstance(value, type) and issubclass(value, BaseContainer):
                        if not value.__abstract__:
                            container = value(manager, path)
                            manager.logger.debug(f"Load container {container.name} in {path}")
                            manager._callback(container.on_init)
                            yield container
                            return
            except Exception as e:
                manager.logger.warning(f"Failed to load container from `{path}`: {e}")
                return

        for compose_name in manager.docker_compose_names:
            compose_path = os.path.join(path, compose_name)
            if os.path.exists(compose_path):
                container = SimpleContainer(manager, path)
                manager.logger.debug(f"Load container {container.name} in {path}")
                manager._callback(container.on_init)
                yield container
                return
