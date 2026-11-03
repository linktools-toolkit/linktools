#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative persistence domain routing."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..storage import ObjectStore, StorageRoute, StorageRetentionMode


from ..core import PersistenceDomain


def runtime_domain_uses_object_store(domain: PersistenceDomain) -> bool:
    """All Runtime persistence addresses support object-store data."""
    if not isinstance(domain, PersistenceDomain):
        raise TypeError("domain must be PersistenceDomain")
    return True


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
                      if domain not in {PersistenceDomain.CHECKPOINT_CONTROL, PersistenceDomain.TRACE, PersistenceDomain.CHECKPOINT}})


    def __post_init__(self) -> None:
        if any(not isinstance(identity, str) or not identity or not isinstance(route, StorageRoute)
               for identity, route in self.capability_routes.items()):
            raise ValueError("capability routes require nonempty identities and StorageRoute values")
        object.__setattr__(self, "capability_routes", MappingProxyType(dict(self.capability_routes)))
        routes = tuple(self.route(domain) for domain in PersistenceDomain)
        if any(not isinstance(route, StorageRoute) for route in routes):
            raise ValueError("PersistencePlan contains an invalid route")

    @property
    def resolved_routes(self) -> Mapping[PersistenceDomain, StorageRoute]:
        return MappingProxyType({domain: self.route(domain) for domain in PersistenceDomain
                                 if domain is not PersistenceDomain.CHECKPOINT_CONTROL})

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
            PersistenceDomain.CHECKPOINT_CONTROL: self.execution,
            PersistenceDomain.TRACE: self.trace or self.execution,
            PersistenceDomain.CHECKPOINT: self.checkpoint or self.execution,
        }[domain]

    @property
    def durable_domains(self) -> frozenset[PersistenceDomain]:
        return frozenset(domain for domain in PersistenceDomain if self.route(domain).retention is StorageRetentionMode.DURABLE)


__all__ = [
    "PersistenceDomain",
    "PersistencePlan",
    "runtime_domain_uses_object_store",
]
