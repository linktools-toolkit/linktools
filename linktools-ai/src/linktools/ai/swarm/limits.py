#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmLimits: the resource/governance caps a SwarmRun enforces ."""

from dataclasses import dataclass
from decimal import Decimal
import math


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

    def __post_init__(self) -> None:
        integer_fields = (
            "max_rounds",
            "max_tasks",
            "max_delegations",
            "max_depth",
            "max_concurrency",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if self.max_total_tokens is not None and (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
        ):
            raise ValueError("max_total_tokens must be an integer")
        if self.max_total_cost is not None and not isinstance(
            self.max_total_cost, Decimal
        ):
            raise ValueError("max_total_cost must be a Decimal")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
        ):
            raise ValueError("timeout_seconds must be a number")
        if self.max_rounds <= 0 or self.max_tasks <= 0 or self.max_concurrency <= 0:
            raise ValueError("rounds, tasks, and concurrency must be positive")
        if self.max_delegations < 0 or self.max_depth < 0:
            raise ValueError("delegations and depth must not be negative")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.max_total_cost is not None and (
            not self.max_total_cost.is_finite() or self.max_total_cost < 0
        ):
            raise ValueError("max_total_cost must be finite and non-negative")
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")


DEFAULT_SWARM_LIMITS = SwarmLimits(
    max_rounds=10,
    max_tasks=50,
    max_delegations=20,
    max_depth=5,
    max_concurrency=4,
    max_total_tokens=None,
    max_total_cost=None,
    timeout_seconds=None,
)
