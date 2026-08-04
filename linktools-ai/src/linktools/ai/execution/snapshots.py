#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The persisted, resumable execution snapshot and the engine-return snapshot data."""


from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from ..errors import UsageObservationConflictError
from ..json import JsonValue
from .domain import MessageCaptureState, RunUsage

if TYPE_CHECKING:
    from datetime import datetime
    from ..model.pricing import ModelPricing
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
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
            ("cache_write_tokens", self.cache_write_tokens),
            ("cache_read_tokens", self.cache_read_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.total_cost is not None and not isinstance(self.total_cost, Decimal):
            object.__setattr__(self, "total_cost", Decimal(str(self.total_cost)))
        if self.total_cost is not None and (
            not self.total_cost.is_finite() or self.total_cost < 0
        ):
            raise ValueError("total_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ModelRequestUsageObservation:
    request_key: str
    usage: RequestUsage
    provider_name: "str | None"
    response_model_name: "str | None"
    provider_request_cost: "Decimal | None" = None

    def __post_init__(self) -> None:
        if not self.request_key:
            raise ValueError("request_key must not be empty")
        if self.provider_request_cost is not None and not isinstance(
            self.provider_request_cost, Decimal
        ):
            object.__setattr__(
                self,
                "provider_request_cost",
                Decimal(str(self.provider_request_cost)),
            )
        if self.provider_request_cost is not None and (
            not self.provider_request_cost.is_finite()
            or self.provider_request_cost < 0
        ):
            raise ValueError(
                "provider_request_cost must be finite and non-negative"
            )


@dataclass(slots=True)
class RunUsageCapture:
    """Accumulate one request-local observation per model response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: Decimal = Decimal("0")
    cost_known: bool = True
    observations: "dict[str, ModelRequestUsageObservation]" = field(
        default_factory=dict
    )

    def observe_request(
        self,
        observation: ModelRequestUsageObservation,
        *,
        pricing_id: "str | None",
        pricing: "ModelPricing | None",
    ) -> None:
        existing = self.observations.get(observation.request_key)
        if existing is not None:
            if existing != observation:
                raise UsageObservationConflictError(
                    "request key contains a different usage observation"
                )
            return
        request_cost = observation.provider_request_cost
        if request_cost is None and pricing_id is not None and pricing is not None:
            request_cost = pricing.cost(
                input_tokens=observation.usage.input_tokens,
                output_tokens=observation.usage.output_tokens,
                cache_read_tokens=observation.usage.cache_read_tokens,
                cache_write_tokens=observation.usage.cache_write_tokens,
            )
        if request_cost is None:
            self.cost_known = False
        else:
            self.total_cost += request_cost
        self.input_tokens += observation.usage.input_tokens
        self.output_tokens += observation.usage.output_tokens
        self.cache_read_tokens += observation.usage.cache_read_tokens
        self.cache_write_tokens += observation.usage.cache_write_tokens
        self.observations[observation.request_key] = observation

    def snapshot(self) -> "RunUsage":
        return RunUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            cache_write_tokens=self.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens,
            total_cost=self.total_cost if self.cost_known else None,
        )


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
    "ModelRequestUsageObservation",
    "RequestUsage",
    "RunSnapshot",
    "RunUsageCapture",
]
