#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared run admission limits and their observed consumption."""

from dataclasses import dataclass
from datetime import datetime, timezone

from ._json import JsonValue


def _nonnegative(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunBudget:
    """Limits shared by an execution tree or task graph.

    Token limits stop new requests at the observed usage threshold. Already
    admitted concurrent requests may finish above it. A deadline only gates
    admission; it does not cancel an effect that has already started.
    """

    model_requests: int | None = None
    tool_calls: int | None = None
    total_tokens: int | None = None
    deadline_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("model_requests", "tool_calls", "total_tokens"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative(value, name)
        if self.deadline_at is not None and (
            not isinstance(self.deadline_at, datetime)
            or self.deadline_at.utcoffset() is None
        ):
            raise ValueError("deadline_at must be a timezone-aware datetime")

    def digest_payload(self) -> dict[str, JsonValue]:
        """Return the semantic budget projection used by request identities."""
        return {
            "model_requests": self.model_requests,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "deadline_at": (
                None if self.deadline_at is None
                else self.deadline_at.astimezone(timezone.utc).isoformat()
            ),
        }


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """Observed usage, with unresolved model usage explicitly represented.

    Counts record admitted dispatches. ``total_tokens`` is the settled known
    subtotal, not an estimate of in-flight or terminal unknown consumption.
    """

    scope_id: str
    limits: RunBudget
    model_requests: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    in_flight_model_requests: int = 0
    unknown_model_requests: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("scope_id is required")
        if not isinstance(self.limits, RunBudget):
            raise TypeError("limits must be a RunBudget")
        for name in (
            "model_requests", "tool_calls", "total_tokens",
            "in_flight_model_requests", "unknown_model_requests",
        ):
            _nonnegative(getattr(self, name), name)
        if self.in_flight_model_requests + self.unknown_model_requests > self.model_requests:
            raise ValueError("unresolved model usage exceeds admitted requests")


__all__ = ["RunBudget", "BudgetUsage"]
