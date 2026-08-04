#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmSpec: declarative multi-Agent orchestration. Names its member agents
(AgentRef), an optional coordinator agent, a strategy declaration, governance
limits, a context-sharing policy, and an aggregation policy."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ...agent.spec import MiddlewareRef
from ...json import JsonValue
from .models import AgentRef

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aggregation import AggregationPolicy
    from .limits import SwarmLimits


@dataclass(frozen=True, slots=True)
class SwarmContextPolicy:
    coordinator_reads_session: bool = True
    worker_reads_session: bool = False
    worker_reads_summary: bool = True
    write_aggregate_to_session: bool = True


@dataclass(frozen=True, slots=True)
class SwarmStrategySpec:
    kind: str
    config: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SwarmSpec:
    id: str
    name: str
    agents: "tuple[AgentRef, ...]"
    strategy: SwarmStrategySpec
    limits: "SwarmLimits"
    context_policy: SwarmContextPolicy
    aggregation: "AggregationPolicy"
    coordinator: "AgentRef | None" = None
    middleware: "tuple[MiddlewareRef, ...]" = ()
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy.kind == "task_graph":
            _validate_task_graph(self)


def _validate_task_graph(spec: "SwarmSpec") -> None:
    """task_graph carries no strategy config, no coordinator, a single
    collect aggregation pass, and a single round with no delegation/depth."""
    from ...errors import InvalidSpecError
    from .aggregation import AggregationMode

    if dict(spec.strategy.config):
        raise InvalidSpecError(
            f"swarm {spec.id!r}: task_graph strategy carries no config"
        )
    if spec.coordinator is not None:
        raise InvalidSpecError(
            f"swarm {spec.id!r}: task_graph strategy has no coordinator"
        )
    if spec.aggregation.mode is not AggregationMode.COLLECT:
        raise InvalidSpecError(
            f"swarm {spec.id!r}: task_graph requires collect aggregation"
        )
    if spec.limits.max_rounds != 1:
        raise InvalidSpecError(
            f"swarm {spec.id!r}: task_graph requires max_rounds == 1"
        )
    if spec.limits.max_delegations != 0:
        raise InvalidSpecError(
            f"swarm {spec.id!r}: task_graph forbids delegation"
        )
    if spec.limits.max_depth != 0:
        raise InvalidSpecError(
            f"swarm {spec.id!r}: task_graph forbids delegation depth"
        )


@runtime_checkable
class SwarmSpecProvider(Protocol):
    """Provides SwarmSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, swarm_id: str) -> SwarmSpec: ...
