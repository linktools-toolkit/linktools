#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmSpec: declarative multi-Agent orchestration . Names its member
agents (AgentRef), a coordinator agent, a strategy declaration, governance limits,
a context-sharing policy, and an aggregation policy."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..agent.spec import MiddlewareRef
from .aggregation import AggregationPolicy
from .limits import SwarmLimits
from .models import AgentRef


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
    coordinator: AgentRef
    strategy: SwarmStrategySpec
    limits: SwarmLimits
    context_policy: SwarmContextPolicy
    aggregation: AggregationPolicy
    middleware: "tuple[MiddlewareRef, ...]" = ()
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class SwarmSpecProvider(Protocol):
    """Provides SwarmSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, swarm_id: str) -> "SwarmSpec": ...
