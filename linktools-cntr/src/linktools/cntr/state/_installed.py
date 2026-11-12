#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Installed-container state: get/add/remove installed containers and the
dependency-aware removal (compare by name; refuse without --force unless a
dependent is also being removed; --force removes dependents too).
"""
from typing import TYPE_CHECKING

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Mapping
    from ..container import BaseContainer
    from ..manager import ContainerManager


_INSTALLED_KEY = "INSTALLED_CONTAINERS"


class InstalledStateStore:
    """Owns the persisted installed-container set behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def get(self, resolve: bool = True) -> "list[BaseContainer]":
        available = self.manager.containers
        with self.manager.environ.locks.process_lock("cntr:settings"):
            containers = self._load(available)
        if resolve:
            containers = self.manager.resolver.resolve_dependencies(containers)
        return containers

    def add(self, *names: str) -> "list[BaseContainer]":
        available = self.manager.containers
        with self.manager.environ.locks.process_lock("cntr:settings"):
            # The lock only serializes writers; each process's persistent
            # store still holds whatever it last read, possibly before
            # another process's write. Reload now, under the lock, so this
            # read-modify-write is based on the latest data instead of
            # clobbering a concurrent writer's change on flush.
            self.manager.settings.reload()
            result = set()
            for name in names:
                container = available.get(name, None)
                if container:
                    result.add(container)
            containers = self._load(available)
            containers.extend(result)
            self._dump(containers, available)
            return list(result)

    def remove(self, *names: str, force: bool = False) -> "list[BaseContainer]":
        available = self.manager.containers
        with self.manager.environ.locks.process_lock("cntr:settings"):
            self.manager.settings.reload()
            containers = self._load(available)

            result = set()
            remove_names = set(names)
            for name in set(names):
                if name not in available:
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
                            f"{container} depends on {available[name]}, "
                            f"cannot remove {available[name]}"
                        )

            for name in remove_names:
                container = available.get(name, None)
                if container and container in containers:
                    result.add(container)
                    containers.remove(container)

            self._dump(containers, available)

            return list(result)

    def load_names(self) -> "list[str]":
        """Return the raw persisted installed-container names, unresolved.

        Used directly by container discovery -- resolving these
        names to container objects here would recurse into discovery.
        """
        return list(self.manager.settings.get(_INSTALLED_KEY, []) or [])

    def _dump_names(self, names: "Iterable[str]") -> None:
        self.manager.settings.set(_INSTALLED_KEY, list(names))

    def _load(self, available: "Mapping[str, BaseContainer]") -> "list[BaseContainer]":
        result = set()
        for name in self.load_names():
            if name in available:
                result.add(available[name])
        return list(result)

    def _dump(self, containers: "Iterable[BaseContainer]", available: "Mapping[str, BaseContainer]") -> None:
        # A temporary load/compatibility failure does not uninstall a container.
        unresolved = set(self.load_names()).difference(available)
        self._dump_names(unresolved.union(container.name for container in containers))
