#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative persistence domain routing."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path

from linktools.ai.storage import CacheBackend, ObjectStore, StorageRoute, StorageRetentionMode


from linktools.ai.core import PersistenceDomain


_OBJECT_STORE_DOMAINS = frozenset(
    {
        PersistenceDomain.CONVERSATION,
        PersistenceDomain.EXECUTION,
        PersistenceDomain.MEMORY,
        PersistenceDomain.ARTIFACT,
        PersistenceDomain.TASK,
        PersistenceDomain.EVALUATION,
        PersistenceDomain.RECOVERY,
        PersistenceDomain.TRACE,
        PersistenceDomain.CHECKPOINT,
    }
)


def runtime_domain_uses_object_store(domain: PersistenceDomain) -> bool:
    """Return whether a Runtime domain owns object-store data."""
    if not isinstance(domain, PersistenceDomain):
        raise TypeError("domain must be PersistenceDomain")
    return domain in _OBJECT_STORE_DOMAINS


@dataclass(frozen=True, slots=True)
class PersistencePlan:
    conversation: StorageRoute = field(default_factory=StorageRoute.memory)
    execution: StorageRoute = field(default_factory=StorageRoute.memory)
    memory: StorageRoute = field(default_factory=StorageRoute.memory)
    artifact: StorageRoute = field(default_factory=StorageRoute.memory)
    task: StorageRoute = field(default_factory=StorageRoute.memory)
    evaluation: StorageRoute = field(default_factory=StorageRoute.memory)
    trace: StorageRoute | None = None
    checkpoint: StorageRoute | None = None
    capability_routes: Mapping[str, StorageRoute] = field(default_factory=dict)

    @classmethod
    def from_default(cls, route: StorageRoute) -> "PersistencePlan":
        """Apply one default resource to the independently routed domains."""
        return cls(**{domain.value: route for domain in PersistenceDomain
                      if domain not in {PersistenceDomain.RECOVERY, PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT}})

    @classmethod
    def filesystem(cls, root: Path, *, cache: CacheBackend | None = None) -> "PersistencePlan":
        """Keep domain members distinct within one filesystem transaction group."""
        return cls(**{
            domain.value: StorageRoute.filesystem(
                root if domain is PersistenceDomain.EXECUTION else root / domain.value,
                transaction_root=root, cache=cache,
            )
            for domain in PersistenceDomain
            if domain not in {PersistenceDomain.RECOVERY, PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT}
        })

    def __post_init__(self) -> None:
        if any(not isinstance(identity, str) or not identity or not isinstance(route, StorageRoute)
               for identity, route in self.capability_routes.items()):
            raise ValueError("capability routes require nonempty identities and StorageRoute values")
        object.__setattr__(self, "capability_routes", MappingProxyType(dict(self.capability_routes)))
        routes = tuple(self.route(domain) for domain in PersistenceDomain)
        if any(not isinstance(route, StorageRoute) for route in routes):
            raise ValueError("PersistencePlan contains an invalid route")
        grouped: dict[Path, list[Path]] = {}
        for domain, route in zip(PersistenceDomain, routes):
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
                relative.parts[0] in {".state-groups", ".state-partitions"}
                or relative.parts[0].startswith(".txn-")
            ):
                raise ValueError("filesystem route points at a reserved transaction path")
            grouped.setdefault(route.transaction_root, []).append(member_path)
        for paths in grouped.values():
            for index, left in enumerate(sorted(paths)):
                for right in sorted(paths)[index + 1 :]:
                    if left in right.parents or right in left.parents:
                        raise ValueError("filesystem group member paths must be disjoint")

    @property
    def resolved_routes(self) -> Mapping[PersistenceDomain, StorageRoute]:
        return MappingProxyType({domain: self.route(domain) for domain in PersistenceDomain})

    def validate_object_store(self, object_store: ObjectStore | None) -> None:
        if object_store is not None and not any(
            runtime_domain_uses_object_store(domain)
            and self.route(domain).retention is StorageRetentionMode.DURABLE
            for domain in PersistenceDomain
        ):
            raise ValueError(
                "object_store requires at least one durable object-capable PersistenceDomain"
            )
        if (
            self.route(PersistenceDomain.CONVERSATION).retention
            is StorageRetentionMode.DURABLE
        ):
            if (
                self.route(PersistenceDomain.EXECUTION).retention
                is not StorageRetentionMode.DURABLE
            ):
                raise ValueError("durable conversation requires durable execution")
            if (
                self.route(PersistenceDomain.RECOVERY).retention
                is not StorageRetentionMode.DURABLE
            ):
                raise ValueError("durable conversation requires durable recovery")
        if (
            self.route(PersistenceDomain.EVALUATION).retention
            is StorageRetentionMode.DURABLE
            and self.route(PersistenceDomain.EXECUTION).retention
            is not StorageRetentionMode.DURABLE
        ):
            raise ValueError("durable evaluation requires durable execution")

    def route(self, domain: PersistenceDomain) -> StorageRoute:
        if not isinstance(domain, PersistenceDomain):
            raise TypeError("PersistenceDomain is required")
        return {
            PersistenceDomain.CONVERSATION: self.conversation,
            PersistenceDomain.EXECUTION: self.execution,
            PersistenceDomain.MEMORY: self.memory,
            PersistenceDomain.ARTIFACT: self.artifact,
            PersistenceDomain.TASK: self.task,
            PersistenceDomain.EVALUATION: self.evaluation,
            PersistenceDomain.RECOVERY: self.execution,
            PersistenceDomain.TRACE: self.trace or self.execution,
            PersistenceDomain.CHECKPOINT: self.checkpoint or self.execution,
        }[domain]

    def filesystem_path(self, domain: PersistenceDomain) -> Path:
        """Return the physical member root for one filesystem domain."""
        route = self.route(domain)
        if route.path is None:
            raise ValueError("filesystem route path is required")
        if domain in {PersistenceDomain.EXECUTION, PersistenceDomain.RECOVERY, PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT}:
            return route.path / domain.value
        return route.path

    @property
    def durable_domains(self) -> frozenset[PersistenceDomain]:
        return frozenset(domain for domain in PersistenceDomain if self.route(domain).retention is StorageRetentionMode.DURABLE)


__all__ = [
    "PersistenceDomain",
    "PersistencePlan",
    "runtime_domain_uses_object_store",
]
