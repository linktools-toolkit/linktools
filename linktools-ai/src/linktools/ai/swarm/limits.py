#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmLimits: the resource/governance caps a SwarmRun enforces (spec 22.3)."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class SwarmLimits:
    max_rounds: int
    max_tasks: int
    max_delegations: int
    max_depth: int
    max_concurrency: int
    max_total_tokens: "int | None"
    max_total_cost: "Decimal | None"
    timeout_seconds: "float | None"


DEFAULT_SWARM_LIMITS = SwarmLimits(
    max_rounds=10, max_tasks=50, max_delegations=20, max_depth=5,
    max_concurrency=4, max_total_tokens=None, max_total_cost=None, timeout_seconds=None,
)
