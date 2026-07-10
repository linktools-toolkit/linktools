#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container dependency resolution (refactor spec Phase 4).

Extracted verbatim from ContainerManager.resolve_depend_containers so the
manager can delegate to it. Behavior is unchanged: topological sort by declared
dependencies (stable by (order, name)), with cycle and missing-dependency errors
identical to the legacy implementation.
"""
from typing import TYPE_CHECKING

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..container import BaseContainer
    from ..manager import ContainerManager


class ContainerResolver:
    """Resolve inter-container dependencies behind the ContainerManager facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def resolve_dependencies(self, containers: "Iterable[BaseContainer]") -> "list[BaseContainer]":
        result: "list[BaseContainer]" = []
        visited: "set[BaseContainer]" = set()
        visiting: "set[BaseContainer]" = set()

        def visit(container: "BaseContainer"):
            if container in visited:
                return
            if container in visiting:
                raise ContainerError(f"Circular dependency detected for container {container}")
            visiting.add(container)
            for dependency in container.dependencies:
                if dependency not in self.manager.containers:
                    raise ContainerError(f"Dependency container {dependency} not found")
                visit(self.manager.containers[dependency])
            visiting.remove(container)
            visited.add(container)
            result.append(container)

        for container in sorted(containers, key=lambda o: (o.order, o.name)):
            visit(container)
        return result
