#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""observability.metrics: counter/histogram/gauge abstraction for linktools.ai.

ObservabilityMetrics is the Protocol boundary an OpenTelemetry adapter can
later plug into; it has no OTel dependency. A single class can
implement both this and ObservabilitySink -- LoggingObservabilitySink does."""

from typing import Any, Mapping, Protocol, runtime_checkable


# Stable low-cardinality instrument names used by the hardening paths. IDs and
# user data must remain event/log fields, never metric labels.
HARDENING_METRICS = (
    "run_cancellation_requested_total",
    "run_cancellation_completed_total",
    "run_cancellation_timeout_total",
    "run_resume_manifest_mismatch_total",
    "run_cross_tenant_denied_total",
    "tool_commit_retry_total",
    "tool_side_effect_unknown_total",
    "tool_idempotency_lease_lost_total",
    "retrieval_scope_missing_total",
    "cross_tenant_retrieval_denied_total",
    "memory_index_pending_total",
    "memory_index_failed_total",
    "context_injection_flagged_total",
)


class InMemoryMetrics:
    """Small default sink useful for tests and local diagnostics."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def counter(self, name: str, *, value: int = 1, attributes=None) -> None:
        if name not in HARDENING_METRICS:
            return
        self.counters[name] = self.counters.get(name, 0) + value

    def histogram(self, name: str, *, value: float, attributes=None) -> None:
        return None

    def gauge(self, name: str, *, value: float, attributes=None) -> None:
        return None


@runtime_checkable
class ObservabilityMetrics(Protocol):
    """Metrics boundary: named numeric instruments with optional attributes."""

    def counter(
        self,
        name: str,
        *,
        value: int = 1,
        attributes: "Mapping[str, Any] | None" = None,
    ) -> None:
        """Increment a monotonically-increasing counter by `value` (>= 0)."""
        ...

    def histogram(
        self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        """Record an observation into a distribution (e.g. latency)."""
        ...

    def gauge(
        self, name: str, *, value: float, attributes: "Mapping[str, Any] | None" = None
    ) -> None:
        """Set a instantaneous gauge to `value` (may go up or down)."""
        ...
