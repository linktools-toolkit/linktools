#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Installed-container state (refactor spec Phase 4).

Extracted verbatim from ContainerManager: get/add/remove installed containers and
the INSTALLED_CONTAINERS load/dump helpers. Behavior is unchanged, including the
dependency-aware removal (compare by name; refuse without --force; force removes
dependents) fixed in §5.2.
"""
from typing import TYPE_CHECKING

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..container import BaseContainer
    from ..manager import ContainerManager


_INSTALLED_KEY = "INSTALLED_CONTAINERS"


class InstalledStateStore:
    """Owns the persisted installed-container set behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def get(self, resolve: bool = True) -> "list[BaseContainer]":
        with self.manager.environ.locks.process_lock("cntr:settings"):
            containers = self._load(reload=False)
        if resolve:
            containers = self.manager.resolver.resolve_dependencies(containers)
        return containers

    def add(self, *names: str) -> "list[BaseContainer]":
        with self.manager.environ.locks.process_lock("cntr:settings"):
            result = set()
            for name in names:
                container = self.manager.containers.get(name, None)
                if container:
                    result.add(container)
            containers = self._load(reload=True)
            containers.extend(result)
            self._dump(containers)
            return list(result)

    def remove(self, *names: str, force: bool = False) -> "list[BaseContainer]":
        with self.manager.environ.locks.process_lock("cntr:settings"):
            containers = self._load(reload=True)

            result = set()
            remove_names = set(names)
            for name in set(names):
                if name not in self.manager.containers:
                    continue
                for container in containers:
                    if not container.is_depend_on(name):
                        continue
                    if container.name in remove_names:
                        continue
                    if force:
                        remove_names.add(container.name)
                    elif container.name not in remove_names:
                        raise ContainerError(
                            f"{container} depends on {self.manager.containers[name]}, "
                            f"cannot remove {self.manager.containers[name]}"
                        )

            for name in remove_names:
                container = self.manager.containers.get(name, None)
                if container and container in containers:
                    result.add(container)
                    containers.remove(container)

            self._dump(containers)

            return list(result)

    def _load(self, reload: bool = False) -> "list[BaseContainer]":
        result = set()
        for name in self.manager._load_setting(_INSTALLED_KEY, reload=reload, default=[]):
            if name in self.manager.containers:
                result.add(self.manager.containers[name])
        return list(result)

    def _dump(self, containers: "Iterable[BaseContainer]") -> None:
        self.manager._dump_setting(
            _INSTALLED_KEY, list(set([container.name for container in containers])))
