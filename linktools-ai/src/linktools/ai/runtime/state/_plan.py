#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative Runtime domain routing."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from pathlib import Path

from ...storage import StorageRoute, StorageRetentionMode


from ...core import RuntimeDomain


_OBJECT_STORE_DOMAINS = frozenset(
    {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.TASK,
        RuntimeDomain.EVALUATION,
        RuntimeDomain.RECOVERY,
        RuntimeDomain.TRACE,
        RuntimeDomain.CHECKPOINT,
    }
)


def runtime_domain_uses_object_store(domain: RuntimeDomain) -> bool:
    """Return whether a Runtime domain owns object-store data."""
    if not isinstance(domain, RuntimeDomain):
        raise TypeError("domain must be RuntimeDomain")
    return domain in _OBJECT_STORE_DOMAINS


@dataclass(frozen=True, slots=True)
class RuntimeStoragePlan:
    conversation: StorageRoute = field(default_factory=StorageRoute.memory)
    execution: StorageRoute = field(default_factory=StorageRoute.memory)
    memory: StorageRoute = field(default_factory=StorageRoute.memory)
    artifact: StorageRoute = field(default_factory=StorageRoute.memory)
    task: StorageRoute = field(default_factory=StorageRoute.memory)
    evaluation: StorageRoute = field(default_factory=StorageRoute.memory)
    _module_routes: Mapping[RuntimeDomain, StorageRoute] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        routes = tuple(self.route(domain) for domain in RuntimeDomain)
        if any(not isinstance(route, StorageRoute) for route in routes):
            raise ValueError("RuntimeStoragePlan contains an invalid route")
        grouped: dict[Path, list[Path]] = {}
        for domain, route in zip(RuntimeDomain, routes):
            if (
                route.kind != "filesystem"
                or route.transaction_root is None
            ):
                continue
            member_path = self.filesystem_path(domain)
            try:
                relative = member_path.relative_to(route.transaction_root)
            except ValueError as error:
                raise ValueError("filesystem route must be under transaction_root") from error
            if relative.parts and (
                relative.parts[0] in {".state-groups"}
                or relative.parts[0].startswith(".txn-")
            ):
                raise ValueError("filesystem route points at a reserved transaction path")
            grouped.setdefault(route.transaction_root, []).append(member_path)
        for paths in grouped.values():
            for index, left in enumerate(sorted(paths)):
                for right in sorted(paths)[index + 1 :]:
                    if left in right.parents or right in left.parents:
                        raise ValueError("filesystem group member paths must be disjoint")

    def with_module_routes(
        self, routes: Mapping[RuntimeDomain, StorageRoute],
    ) -> "RuntimeStoragePlan":
        """Assemble module declarations without adding authoring route fields."""
        allowed = {RuntimeDomain.TRACE, RuntimeDomain.CHECKPOINT}
        if not routes.keys() <= allowed:
            raise ValueError("module routes require persistence archive domains")
        effective = replace(self)
        object.__setattr__(effective, "_module_routes", MappingProxyType({**self._module_routes, **routes}))
        effective.__post_init__()
        return effective

    def route(self, domain: RuntimeDomain) -> StorageRoute:
        if not isinstance(domain, RuntimeDomain):
            raise TypeError("RuntimeDomain is required")
        if domain in self._module_routes:
            return self._module_routes[domain]
        return {
            RuntimeDomain.CONVERSATION: self.conversation,
            RuntimeDomain.EXECUTION: self.execution,
            RuntimeDomain.MEMORY: self.memory,
            RuntimeDomain.ARTIFACT: self.artifact,
            RuntimeDomain.TASK: self.task,
            RuntimeDomain.EVALUATION: self.evaluation,
            RuntimeDomain.RECOVERY: self.execution,
            RuntimeDomain.TRACE: self.execution,
            RuntimeDomain.CHECKPOINT: self.execution,
        }[domain]

    def filesystem_path(self, domain: RuntimeDomain) -> Path:
        """Return the physical member root for one filesystem domain."""
        route = self.route(domain)
        if route.path is None:
            raise ValueError("filesystem route path is required")
        if domain in {RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY, RuntimeDomain.TRACE, RuntimeDomain.CHECKPOINT}:
            return route.path / domain.value
        return route.path

    @property
    def durable_domains(self) -> frozenset[RuntimeDomain]:
        return frozenset(domain for domain in RuntimeDomain if self.route(domain).retention is StorageRetentionMode.DURABLE)


__all__ = [
    "RuntimeDomain",
    "RuntimeStoragePlan",
    "runtime_domain_uses_object_store",
]
