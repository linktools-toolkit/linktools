#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The persisted, resumable execution snapshot and the engine-return snapshot data."""


from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from ..errors import UsageRegressionError
from ..json import JsonValue
from .domain import MessageCaptureState, RunUsage

if TYPE_CHECKING:
    from datetime import datetime
    from .domain import RunStatus


@dataclass(frozen=True, slots=True)
class RequestUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost: "Decimal | None" = None

    def __post_init__(self) -> None:
        if self.total_cost is not None and not isinstance(self.total_cost, Decimal):
            object.__setattr__(self, "total_cost", Decimal(str(self.total_cost)))


@dataclass(frozen=True, slots=True)
class ModelUsageObservation:
    request_usage: RequestUsage
    cumulative_usage: RunUsage
    resolved_model_id: "str | None"
    provider_total_cost: "Decimal | None" = None

    def __post_init__(self) -> None:
        if self.provider_total_cost is not None and not isinstance(
            self.provider_total_cost, Decimal
        ):
            object.__setattr__(
                self,
                "provider_total_cost",
                Decimal(str(self.provider_total_cost)),
            )


@dataclass(slots=True)
class RunUsageCapture:
    """Per-run sink for authoritative cumulative model usage observations."""

    current: "RunUsage" = field(default_factory=RunUsage)
    cost_known: bool = True
    _last_observation_key: object = field(default=None, init=False, repr=False)

    def observe(self, observation: ModelUsageObservation) -> None:
        key = (
            observation.request_usage,
            observation.cumulative_usage,
            observation.resolved_model_id,
            observation.provider_total_cost,
        )
        if key == self._last_observation_key:
            return
        self._last_observation_key = key
        cumulative = observation.cumulative_usage
        if cumulative.input_tokens < self.current.input_tokens:
            raise UsageRegressionError("input_tokens decreased")
        if cumulative.output_tokens < self.current.output_tokens:
            raise UsageRegressionError("output_tokens decreased")
        if cumulative.total_tokens < self.current.total_tokens:
            raise UsageRegressionError("total_tokens decreased")
        if cumulative.cache_write_tokens < self.current.cache_write_tokens:
            raise UsageRegressionError("cache_write_tokens decreased")
        if cumulative.cache_read_tokens < self.current.cache_read_tokens:
            raise UsageRegressionError("cache_read_tokens decreased")
        total_cost = self._resolve_total_cost(observation)
        self.current = RunUsage(
            input_tokens=cumulative.input_tokens,
            output_tokens=cumulative.output_tokens,
            total_tokens=cumulative.total_tokens,
            cache_write_tokens=cumulative.cache_write_tokens,
            cache_read_tokens=cumulative.cache_read_tokens,
            total_cost=total_cost,
        )

    def _resolve_total_cost(
        self, observation: ModelUsageObservation
    ) -> "Decimal | None":
        provider_total = observation.provider_total_cost
        if provider_total is not None:
            if not provider_total.is_finite() or provider_total < 0:
                raise ValueError("provider_total_cost must be finite and non-negative")
            if (
                self.current.total_cost is not None
                and provider_total < self.current.total_cost
            ):
                raise UsageRegressionError("total_cost decreased")
            self.cost_known = True
            return provider_total
        if observation.resolved_model_id is None:
            self.cost_known = False
            return None
        request_cost = observation.request_usage.total_cost
        if request_cost is None or not self.cost_known:
            self.cost_known = False
            return None
        if not request_cost.is_finite() or request_cost < 0:
            raise ValueError("request cost must be finite and non-negative")
        return (self.current.total_cost or Decimal("0")) + request_cost

    def snapshot(self) -> "RunUsage":
        return self.current


@dataclass(frozen=True, slots=True)
class AgentSnapshotData:
    # TURN_DELTA material: this run's new_messages(), window-policy immune.
    # Persisted on the turn row at every terminal state (audit record).
    delta_messages: "tuple[JsonValue, ...]"
    final_output: "JsonValue | None"
    usage: "RunUsage"
    trace_end_sequence: int
    # Honesty marker for the turn's delta (see MessageCaptureState).
    capture_state: MessageCaptureState = MessageCaptureState.COMPLETE
    # RESUME_CHECKPOINT material: all_messages() -- the exact post-window-policy
    # pause context. Non-empty ONLY for PAUSED; the store clears it on every
    # terminal state so no cumulative history is retained (O(N^2) eliminated).
    checkpoint_messages: "tuple[JsonValue, ...]" = ()


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    schema: "Literal['run-snapshot.v1']"
    run_id: str
    revision: int
    # RESUME_CHECKPOINT (column name kept): non-empty only for PAUSED; the store
    # clears it on every terminal state so steady-state snapshots hold no
    # cumulative history.
    resume_messages: "tuple[JsonValue, ...]"
    final_output: "JsonValue | None"
    status: "RunStatus"
    usage: "RunUsage"
    trace_end_sequence: int
    created_at: "datetime"


__all__: "list[str]" = [
    "AgentSnapshotData",
    "ModelUsageObservation",
    "RequestUsage",
    "RunSnapshot",
    "RunUsageCapture",
]
