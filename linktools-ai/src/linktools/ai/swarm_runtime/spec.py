#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmSpec: declarative multi-Agent orchestration (spec 22.2). Names its member
agents (AgentRef), a coordinator agent, a strategy declaration, governance limits,
a context-sharing policy, and an aggregation policy."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..agent_runtime.spec import MiddlewareRef
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
